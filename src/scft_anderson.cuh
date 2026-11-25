constexpr int maxMixHistory = 10;
constexpr int mixThreads = 128;
constexpr int mixPartials = 12;
constexpr int parallelNormPartials = 96;

enum class GramLayout { Square, Ordered, Extend };
enum class MixTrial { Parent, Reset, Period2, Period4, RidgeFixed, RidgeAdaptive, Rpm1, Rpm2,
                      BlockDiag, Block2, Block4 };

struct AndersonMixer;
struct MixWindow;
struct MixIntervention {
    virtual bool attempt(AndersonMixer& mixer, MixWindow window, int update, int next) = 0;
    virtual ~MixIntervention() = default;
};

MixTrial requestedTrial() {
    const char* option = std::getenv("SCFT_MATILDA_MIX_TRIAL");
    if (!option || std::string_view(option) == "parent") return MixTrial::Parent;
    const std::string_view name(option);
    if (name == "reset") return MixTrial::Reset;
    if (name == "period2") return MixTrial::Period2;
    if (name == "period4") return MixTrial::Period4;
    if (name == "ridge_fixed") return MixTrial::RidgeFixed;
    if (name == "ridge_adapt") return MixTrial::RidgeAdaptive;
    if (name == "rpm1") return MixTrial::Rpm1;
    if (name == "rpm2") return MixTrial::Rpm2;
    if (name == "block_diag") return MixTrial::BlockDiag;
    if (name == "block2") return MixTrial::Block2;
    if (name == "block4") return MixTrial::Block4;
    throw std::runtime_error("Unsupported Anderson experiment: " + std::string(name));
}

struct RidgeAgreement {
    static constexpr double initial = 1e-4;
    double lambda = initial;
    double currentSquare = 0.0;
    double prediction = 0.0;
    double actual = 0.0;

    void update(double nextSquare, bool rejected) {
        actual = currentSquare - nextSquare;
        if (rejected || !(prediction > 1e-12 * currentSquare) || actual / prediction < 0.25) {
            lambda = std::min(1.0, 4 * lambda);
        } else if (actual / prediction > 0.75) {
            lambda = std::max(1e-8, 0.5 * lambda);
        }
    }
};

struct MixFields {
    Complex* field[3];
    const Complex* force[3];
    double timestep[3];
};

struct MixWindow {
    int history;
    int slot[maxMixHistory + 1];
    double gamma[maxMixHistory];
};

void checkMixCuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

thrust::device_vector<double> pressureMultiplier(FTS_Box& box) {
    const char* option = std::getenv("SCFT_MATILDA_MIX_PRESSURE_CAP");
    if (!option) return {};
    char* end = nullptr;
    const double cap = std::strtod(option, &end);
    if (end == option || *end || !std::isfinite(cap) || cap < 1.0 || cap > 4.0) {
        throw std::runtime_error("Anderson pressure multiplier requires a cap from 1 to 4");
    }
    if (cap == 1.0) return {};
    std::vector<const LinearMolec*> chains;
    double zeroResponse = 0.0;
    for (const auto* molecule : box.Molecs) {
        if (molecule->molec_type == "HParticle") continue;
        const auto* chain = dynamic_cast<const LinearMolec*>(molecule);
        if (!chain || chain->doSmear || chain->Ntot < 1 || chain->activity > 0 || chain->phi <= 0) {
            throw std::runtime_error("Pressure response requires canonical, unsmeared linear chains");
        }
        chains.push_back(chain);
        zeroResponse += chain->phi * chain->Ntot;
    }
    if (chains.empty() || box.Nr <= 1) throw std::runtime_error("Missing pressure response model");
    const double inverseSites = 1.0 / box.M;
    thrust::host_vector<double> multiplier(box.M, inverseSites);
    std::vector<double> wavevector(box.returnDimension());
    for (int i = 1; i < box.M; ++i) {
        const double bond = std::exp(-box.get_kD(i, wavevector.data()) / (box.Nr - 1.0));
        double response = 0.0;
        for (const auto* chain : chains) {
            const int sites = chain->Ntot;
            double sum = sites;
            double power = bond;
            for (int distance = 1; distance < sites; ++distance, power *= bond) {
                sum += 2 * (sites - distance) * power;
            }
            response += chain->phi * sum / sites;
        }
        multiplier[i] = inverseSites * std::clamp(zeroResponse / response, 1.0, cap);
    }
    std::cout << "Anderson pressure response cap " << cap << ", uniform multiplier 1\n";
    return thrust::device_vector<double>(multiplier);
}

__global__ void multiplyPressure(Complex* values, const double* multiplier, int sites) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) values[i] *= multiplier[i];
}

__global__ void captureMixState(MixFields fields, int sites, Complex* x, Complex* r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    for (int component = 0; component < 3; ++component) {
        const int offset = component * sites + i;
        x[offset] = fields.field[component][i];
        r[offset] = -fields.timestep[component] * fields.force[component][i];
    }
}

__device__ Complex residualColumn(const Complex* r, int width, int i, MixWindow window, int column) {
    if (column == window.history) return r[window.slot[column] * width + i];
    return r[window.slot[column + 1] * width + i] - r[window.slot[column] * width + i];
}

struct MixGramEntry {
    int row;
    int column;
};

template<GramLayout layout>
__host__ __device__ MixGramEntry mixGramEntry(int index, int history) {
    if constexpr (layout == GramLayout::Extend) {
        if (index < history) return {history - 1, index};
        return {index - history, history};
    }
    if constexpr (layout == GramLayout::Ordered) {
        const int triangle = history * (history + 1) / 2;
        if (index >= triangle) return {index - triangle, history};
        int row = 0;
        while (index > row) index -= ++row;
        return {row, index};
    }
    const int dimensions = history + 1;
    return {index / dimensions, index % dimensions};
}

template<GramLayout layout>
__global__ void mixGram(const Complex* r, int sites, MixWindow window, double* parts) {
    __shared__ double totals[mixThreads];
    const int width = 3 * sites;
    const auto [row, column] = mixGramEntry<layout>(blockIdx.x, window.history);
    double sum = 0.0;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < width; i += blockDim.x * gridDim.y) {
        const Complex a = residualColumn(r, width, i, window, row);
        const Complex b = residualColumn(r, width, i, window, column);
        // Compensate for the smaller exchange EM step in the fit.
        const double weight = i / sites == 2 ? 4.0 : 1.0;
        sum += weight * (a.real() * b.real() + a.imag() * b.imag());
    }
    totals[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = mixThreads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) totals[threadIdx.x] += totals[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) parts[blockIdx.x * gridDim.y + blockIdx.y] = totals[0] / sites;
}

__global__ void applyMix(MixFields fields, int sites, const Complex* x, const Complex* r, MixWindow window) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    for (int component = 0; component < 3; ++component) {
        const auto index = [&](int j) { return (3 * window.slot[j] + component) * sites + i; };
        int current = index(window.history);
        Complex next = x[current] + r[current];
        for (int h = 0; h < window.history; ++h) {
            const int a = index(h), b = index(h + 1);
            next -= window.gamma[h] * ((x[b] - x[a]) + (r[b] - r[a]));
        }
        fields.field[component][i] = next;
    }
}

struct MixStepOutcome {
    bool fitPhase = false;
    bool eligible = false;
    bool attempted = false;
    bool accepted = false;
    bool rejected = false;
    bool rankReset = false;
    bool scheduledReset = false;
};

struct AndersonMixer {
    FTS_Box& box;
    PotentialIncompress& pressure;
    PotentialFlory& flory;
    const int history;
    const bool orderedGram;
    const bool cacheGram;
    const bool parallelNorm;
    const MixTrial trial;
    const int cadence;
    const bool ridgeEnabled;
    const bool verifyCache = std::getenv("SCFT_MATILDA_MIX_VERIFY_CACHE") != nullptr;
    DeviceField x, r;
    thrust::device_vector<double> partials;
    thrust::device_vector<double> pressureWeights;
    using GramMatrix = std::array<double, (maxMixHistory + 1) * (maxMixHistory + 1)>;
    MixWindow cachedWindow{};
    GramMatrix cachedMatrix{};
    int reusedGramEntries = 0;
    int head = 0;
    int retained = 1;
    int mixedSteps = 0;
    int rejections = 0;
    int rankRestarts = 0;
    int evictions = 0;
    int clippedSteps = 0;
    int fitAttempts = 0;
    int explicitResets = 0;
    int skippedFits = 0;
    int corrections = 0;
    RidgeAgreement agreement;
    std::ofstream trialTrace;
    std::ofstream fitOracle;
    double norm = 0.0;

    AndersonMixer(FTS_Box& box_, PotentialIncompress& pressure_, PotentialFlory& flory_,
                  int history_, bool orderedGram_, bool cacheGram_, bool parallelNorm_, MixTrial trial_)
        : box(box_), pressure(pressure_), flory(flory_), history(history_), orderedGram(orderedGram_),
          cacheGram(cacheGram_), parallelNorm(parallelNorm_),
          trial(trial_), cadence(trial == MixTrial::Period4 ? 4 : trial == MixTrial::Period2 ? 2 : 1),
          ridgeEnabled(trial == MixTrial::RidgeFixed || trial == MixTrial::RidgeAdaptive),
          x(size_t(history + 1) * 3 * box.M), r(x.size()),
          partials(std::max(gramEntryCount(history) * mixPartials, parallelNorm ? parallelNormPartials : 0)),
          pressureWeights(pressureMultiplier(box)) {
        const char* requestedTrace = std::getenv("SCFT_MATILDA_MIX_TRACE");
        if (requestedTrace && std::string_view(requestedTrace) == "1") {
            trialTrace = openOutput(Path(std::getenv("SCFT_RUN_DIR")) / "mix-trial.csv");
            trialTrace << "updates,evals,fp64,eligible,fit_attempted,fit_accepted,raw_rejected,"
                          "scheduled_reset,rank_reset,lambda,next_lambda,prediction,actual,"
                          "raw_norm,fitted_norm,coefficient_s,fit_norm_s,subspace_accepted\n";
        }
        if (ridgeEnabled && verifyCache) {
            fitOracle = openOutput(Path(std::getenv("SCFT_RUN_DIR")) / "ridge-oracle.log");
        }
        resetFromCurrentState();
    }

    void resetFromCurrentState() {
        head = 0;
        retained = 1;
        cachedWindow.history = 0;
        capture(0);
        norm = residualNorm(0);
        if (!std::isfinite(norm)) throw std::runtime_error("Nonfinite residual at mixing restart");
        preconditionAcceptedPressure(0);
        if (ridgeEnabled) {
            agreement = {};
            const double fitted = pressureWeights.empty() ? norm : residualNorm(0);
            agreement.currentSquare = fitted * fitted;
        }
    }

    MixFields fields() const {
        return {{thrust::raw_pointer_cast(pressure.d_wpl.data()),
                 thrust::raw_pointer_cast(flory.d_wpl.data()),
                 thrust::raw_pointer_cast(flory.d_wmi.data())},
                {thrust::raw_pointer_cast(pressure.d_dHdw.data()),
                 thrust::raw_pointer_cast(flory.d_dHdwpl.data()),
                 thrust::raw_pointer_cast(flory.d_dHdwmi.data())},
                {pressure.delt, flory.deltPlus, flory.deltMinus}};
    }

    int blocks() const { return (box.M + mixThreads - 1) / mixThreads; }

    void preconditionAcceptedPressure(int slot) {
        if (pressureWeights.empty()) return;
        // The current residual is raw through the growth test. Only accepted
        // states enter the preconditioned fit and update.
        auto* values = thrust::raw_pointer_cast(r.data()) + size_t(slot) * 3 * box.M;
        box.cufftTransformDouble(values, values, 1, false);
        multiplyPressure<<<blocks(), mixThreads>>>(
            values, thrust::raw_pointer_cast(pressureWeights.data()), box.M);
        box.cufftTransformDouble(values, values, -1, false);
    }

    void capture(int slot) {
        pressure.computeForce();
        flory.computeForce();
        const size_t offset = size_t(slot) * 3 * box.M;
        captureMixState<<<blocks(), mixThreads>>>(fields(), box.M,
            thrust::raw_pointer_cast(x.data() + offset), thrust::raw_pointer_cast(r.data() + offset));
    }

    int gramEntryCount(int differences) const {
        if (orderedGram && differences > 0) {
            return differences * (differences + 1) / 2 + differences;
        }
        return (differences + 1) * (differences + 1);
    }

    template<GramLayout layout>
    void gramWithLayout(MixWindow window, double* result) {
        const int elements = layout == GramLayout::Extend ? 2 * window.history : gramEntryCount(window.history);
        std::array<double, (maxMixHistory + 1) * (maxMixHistory + 1) * mixPartials> host{};
        auto* scratch = thrust::raw_pointer_cast(partials.data());
        mixGram<layout><<<dim3(elements, mixPartials), mixThreads>>>(
            thrust::raw_pointer_cast(r.data()), box.M, window, scratch);
        checkMixCuda(cudaMemcpy(host.data(), scratch, elements * mixPartials * sizeof(double), cudaMemcpyDeviceToHost));
        for (int element = 0; element < elements; ++element) {
            // Retain the published operand order and host reduction for every consumed entry.
            const auto [row, column] = mixGramEntry<layout>(element, window.history);
            double& value = result[row * (window.history + 1) + column];
            value = 0.0;
            for (int i = 0; i < mixPartials; ++i) value += host[element * mixPartials + i];
        }
    }

    void gram(MixWindow window, double* result) {
        if (orderedGram) gramWithLayout<GramLayout::Ordered>(window, result);
        else gramWithLayout<GramLayout::Square>(window, result);
    }

    void coefficientGram(MixWindow window, GramMatrix& result) {
        if (!cacheGram) {
            gram(window, result.data());
            return;
        }
        const int n = window.history;
        const int offset = cachedWindow.history - n + 1;
        // Each coefficient fit is followed by one accepted state. A matching
        // suffix contains unchanged residual differences, even through wrap.
        if (n == 1 || (cachedWindow.history > 0 && offset >= 0 &&
                       std::equal(window.slot, window.slot + n, cachedWindow.slot + offset))) {
            const int oldWidth = cachedWindow.history + 1;
            for (int row = 0; row < n - 1; ++row) {
                for (int column = 0; column <= row; ++column) {
                    result[row * (n + 1) + column] =
                        cachedMatrix[(offset + row) * oldWidth + offset + column];
                }
            }
            reusedGramEntries += n * (n - 1) / 2;
            gramWithLayout<GramLayout::Extend>(window, result.data());
        } else {
            gram(window, result.data());
        }
        cachedWindow = window;
        cachedMatrix = result;
        if (verifyCache) {
            GramMatrix full{};
            gram(window, full.data());
            for (int row = 0; row < n; ++row) {
                for (int column = 0; column <= row; ++column) {
                    if (result[row * (n + 1) + column] != full[row * (n + 1) + column]) {
                        throw std::runtime_error("Anderson cached Gram differs from full rebuild");
                    }
                }
                if (result[row * (n + 1) + n] != full[row * (n + 1) + n]) {
                    throw std::runtime_error("Anderson cached right-hand side differs from full rebuild");
                }
            }
        }
    }

    double residualNorm(int slot, const Complex* values = nullptr) {
        double square = 0.0;
        const bool scratchState = values != nullptr;
        if (!values) values = thrust::raw_pointer_cast(r.data());
        const MixWindow window{0, {slot}, {}};
        if (parallelNorm) {
            std::array<double, parallelNormPartials> host{};
            auto* scratch = thrust::raw_pointer_cast(partials.data());
            mixGram<GramLayout::Ordered><<<dim3(1, parallelNormPartials), mixThreads>>>(
                values, box.M, window, scratch);
            checkMixCuda(cudaMemcpy(host.data(), scratch, sizeof(host), cudaMemcpyDeviceToHost));
            for (double part : host) square += part;
        } else if (!scratchState) {
            gram(window, &square);
        } else {
            std::array<double, mixPartials> host{};
            auto* scratch = thrust::raw_pointer_cast(partials.data());
            mixGram<GramLayout::Ordered><<<dim3(1, mixPartials), mixThreads>>>(values, box.M, window, scratch);
            checkMixCuda(cudaMemcpy(host.data(), scratch, sizeof(host), cudaMemcpyDeviceToHost));
            for (double part : host) square += part;
        }
        return std::sqrt(square);
    }

    bool coefficients(MixWindow& window) {
        const int n = window.history;
        const int width = n + 1;
        GramMatrix gramMatrix{};
        double lower[maxMixHistory][maxMixHistory]{};
        coefficientGram(window, gramMatrix);
        double scale = 0.0;
        for (int i = 0; i < n; ++i) scale = std::max(scale, gramMatrix[i * width + i]);
        if (!std::isfinite(scale) || scale == 0) return false;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double value = gramMatrix[i * width + j];
                for (int k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k];
                if (i == j) {
                    if (ridgeEnabled) value += agreement.lambda * scale;
                    if (!(value > scale * 1e-12)) return false;
                    lower[i][j] = std::sqrt(value);
                } else {
                    lower[i][j] = value / lower[j][j];
                }
            }
            double value = gramMatrix[i * width + n];
            for (int j = 0; j < i; ++j) value -= lower[i][j] * window.gamma[j];
            window.gamma[i] = value / lower[i][i];
        }
        double magnitude = 0.0;
        for (int i = n - 1; i >= 0; --i) {
            double value = window.gamma[i];
            for (int j = i + 1; j < n; ++j) value -= lower[j][i] * window.gamma[j];
            window.gamma[i] = value / lower[i][i];
            magnitude += std::abs(window.gamma[i]);
        }
        if (!std::isfinite(magnitude)) return false;
        if (magnitude > 60.0) {
            const double scale = 60.0 / magnitude;
            for (int i = 0; i < n; ++i) window.gamma[i] *= scale;
            ++clippedSteps;
        }
        if (ridgeEnabled) {
            agreement.prediction = 0.0;
            for (int i = 0; i < n; ++i) {
                agreement.prediction += 2 * gramMatrix[i * width + n] * window.gamma[i];
                for (int j = 0; j <= i; ++j) {
                    agreement.prediction -= (i == j ? 1.0 : 2.0) *
                        gramMatrix[i * width + j] * window.gamma[i] * window.gamma[j];
                }
            }
            if (fitOracle.is_open()) {
                fitOracle << n << ' ' << agreement.lambda << ' ' << agreement.currentSquare << ' '
                          << agreement.prediction;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j <= i; ++j) fitOracle << ' ' << gramMatrix[i * width + j];
                    fitOracle << ' ' << gramMatrix[i * width + n] << ' ' << window.gamma[i];
                }
                fitOracle << '\n';
            }
        }
        return true;
    }

    void evaluate(MixWindow window, int next) {
        applyMix<<<blocks(), mixThreads>>>(fields(), box.M,
            thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(r.data()), window);
        for (auto& species : box.Species) {
            species.zeroDensity();
            species.buildPotentialField();
        }
        box.calcMoleculeDensities();
        capture(next);
    }

    MixStepOutcome advance(int step, MixIntervention* intervention) {
        const int slots = history + 1;
        const MixWindow damped{0, {head}, {}};
        auto window = damped;
        const bool scheduledReset = trial == MixTrial::Reset && step >= 79 && (step - 79) % 160 == 0;
        if (scheduledReset) {
            retained = 1;
            cachedWindow.history = 0;
            ++explicitResets;
        }
        const bool eligible = step >= 8 && retained > 1;
        const bool fitPhase = step >= 8 && (step - 8) % cadence == 0;
        const bool fit = eligible && fitPhase;
        bool rankReset = false;
        double coefficientSeconds = 0.0, normSeconds = 0.0;
        const double lambda = agreement.lambda;
        if (fit) {
            ++fitAttempts;
            window.history = retained - 1;
            for (int i = 0; i < retained; ++i) {
                window.slot[i] = (head + slots - window.history + i) % slots;
            }
            const auto begin = trialTrace.is_open() ? Clock::now() : Clock::time_point{};
            if (!coefficients(window)) {
                window = damped;
                retained = 1;
                cachedWindow.history = 0;
                ++rankRestarts;
                rankReset = true;
            }
            if (trialTrace.is_open()) coefficientSeconds = secondsSince(begin);
        } else if (eligible) {
            ++skippedFits;
            // Consecutive-fit reuse is invalid after advancing the ring without a fit.
            cachedWindow.history = 0;
        }
        const int next = (head + 1) % slots;
        const bool corrected = intervention && intervention->attempt(*this, window, step + 1, next);
        if (!corrected) evaluate(window, next);
        double nextNorm = residualNorm(next);
        const bool rejected = !corrected && window.history && (!std::isfinite(nextNorm) || nextNorm > 1.35 * norm);
        if (corrected) {
            ++corrections;
            cachedWindow.history = 0;
        } else if (rejected) {
            ++rejections;
            retained = 1;
            cachedWindow.history = 0;
            evaluate(damped, next);
            nextNorm = residualNorm(next);
        } else if (window.history) {
            ++mixedSteps;
        }
        if (!std::isfinite(nextNorm)) throw std::runtime_error("Nonfinite damped mixing residual");
        preconditionAcceptedPressure(next);
        if (ridgeEnabled) {
            const auto begin = Clock::now();
            const double fitted = pressureWeights.empty() ? nextNorm : residualNorm(next);
            normSeconds = secondsSince(begin);
            const double nextSquare = fitted * fitted;
            if (window.history) {
                if (trial == MixTrial::RidgeAdaptive) agreement.update(nextSquare, rejected);
                else agreement.actual = agreement.currentSquare - nextSquare;
            }
            agreement.currentSquare = nextSquare;
        }
        if (trialTrace.is_open()) {
            trialTrace << step + 1 << ',' << box.densityEvaluations << ',' << observedBondsInDouble(box) << ','
                       << eligible << ',' << fit << ',' << (window.history && !rejected && !corrected) << ',' << rejected << ','
                       << scheduledReset << ',' << rankReset << ',';
            if (ridgeEnabled) {
                trialTrace << lambda << ',' << agreement.lambda << ',';
                if (window.history) {
                    trialTrace << agreement.prediction << ',';
                    if (!rejected) trialTrace << agreement.actual;
                } else {
                    trialTrace << ',';
                }
            } else {
                trialTrace << ",,,";
            }
            trialTrace << ',' << nextNorm << ',';
            if (ridgeEnabled) trialTrace << std::sqrt(agreement.currentSquare);
            trialTrace << ',' << coefficientSeconds << ',' << normSeconds << ',' << corrected << '\n' << std::flush;
        }
        norm = nextNorm;
        if (history && retained == slots) ++evictions;
        retained = corrected ? 1 : std::min(slots, retained + 1);
        head = next;
        if (step % box.logFreq == 0) box.writeData(step);
        return {fitPhase, eligible, fit, window.history && !rejected && !corrected, rejected,
                rankReset, scheduledReset};
    }
};

int andersonDepth(std::string_view mode) {
    if (mode == "damped") return 0;
    int history = 0;
    const auto result = std::from_chars(mode.data(), mode.data() + mode.size(), history);
    if (result.ec != std::errc{} || result.ptr != mode.data() + mode.size() ||
        history < 1 || history > maxMixHistory) return -1;
    return history;
}

std::unique_ptr<AndersonMixer> createAnderson(FTS_Box& box, const BinaryModel& model) {
    const auto trial = requestedTrial();
    const char* value = std::getenv("SCFT_MATILDA_ANDERSON");
    if (!value || !model.canonical || box.densityFieldFreq != 0 || box.chemFieldFreq != 0) return nullptr;
    const int history = andersonDepth(value);
    if (history < 0) return nullptr;
    PotentialIncompress* pressure = nullptr;
    PotentialFlory* flory = nullptr;
    for (auto* potential : box.Potentials) {
        if (potential->zeroMean) return nullptr;
        if (auto* candidate = dynamic_cast<PotentialIncompress*>(potential)) pressure = candidate;
        if (auto* candidate = dynamic_cast<PotentialFlory*>(potential)) flory = candidate;
    }
    if (!pressure || !flory) return nullptr;
    // The mixer evaluates forces directly. EMPC recreates these snapshots by
    // assignment if a later run selects the predictor/corrector iterator.
    for (auto* snapshot : {&pressure->d_wplo, &pressure->d_dHdwplo,
                           &flory->d_wplo, &flory->d_wmio, &flory->d_dHdwplo, &flory->d_dHdwmio}) {
        DeviceField{}.swap(*snapshot);
    }
    std::cout << "Single-evaluation controller, Anderson history " << history << '\n';
    const char* backend = std::getenv("SCFT_MATILDA_MIX_GRAM");
    const bool orderedGram = backend && std::string_view(backend) == "ordered";
    const char* reuse = std::getenv("SCFT_MATILDA_MIX_CACHE");
    const bool cacheGram = reuse && std::string_view(reuse) == "1";
    const char* normMode = std::getenv("SCFT_MATILDA_MIX_NORM");
    const bool parallelNorm = normMode && std::string_view(normMode) == "parallel";
    if (cacheGram && !orderedGram) throw std::runtime_error("Incremental Gram requires the ordered backend");
    std::cout << "Anderson Gram backend " << (orderedGram ? "ordered" : "original") << std::endl;
    if (cacheGram) std::cout << "Anderson Gram cache enabled\n";
    if (parallelNorm) std::cout << "Anderson norm uses 96 partials\n";
    const char* experiment = std::getenv("SCFT_MATILDA_MIX_TRIAL");
    std::cout << "Anderson experiment " << (experiment ? experiment : "parent") << std::endl;
    return std::make_unique<AndersonMixer>(box, *pressure, *flory, history, orderedGram, cacheGram, parallelNorm, trial);
}
