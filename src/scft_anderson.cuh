constexpr int maxMixHistory = 5;
constexpr int mixThreads = 128;
constexpr int mixPartials = 12;

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

__global__ void mixGram(const Complex* r, int sites, MixWindow window, double* parts) {
    __shared__ double totals[mixThreads];
    const int width = 3 * sites;
    const int dimensions = window.history + 1;
    const int row = blockIdx.x / dimensions;
    const int column = blockIdx.x % dimensions;
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
    if (threadIdx.x == 0) parts[blockIdx.x * mixPartials + blockIdx.y] = totals[0] / sites;
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

struct AndersonMixer {
    FTS_Box& box;
    PotentialIncompress& pressure;
    PotentialFlory& flory;
    const int history;
    DeviceField x, r;
    thrust::device_vector<double> partials;
    int head = 0;
    int retained = 1;
    int mixedSteps = 0;
    int rejections = 0;
    int restarts = 0;
    int clippedSteps = 0;
    double norm = 0.0;

    AndersonMixer(FTS_Box& box_, PotentialIncompress& pressure_, PotentialFlory& flory_, int history_)
        : box(box_), pressure(pressure_), flory(flory_), history(history_),
          x(size_t(history + 1) * 3 * box.M), r(x.size()),
          partials((history + 1) * (history + 1) * mixPartials) {
        resetFromCurrentState();
    }

    void resetFromCurrentState() {
        head = 0;
        retained = 1;
        capture(0);
        norm = residualNorm(0);
        if (!std::isfinite(norm)) throw std::runtime_error("Nonfinite residual at mixing restart");
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

    void capture(int slot) {
        pressure.computeForce();
        flory.computeForce();
        const size_t offset = size_t(slot) * 3 * box.M;
        captureMixState<<<blocks(), mixThreads>>>(fields(), box.M,
            thrust::raw_pointer_cast(x.data() + offset), thrust::raw_pointer_cast(r.data() + offset));
    }

    void gram(MixWindow window, double* result) {
        const int elements = (window.history + 1) * (window.history + 1);
        std::array<double, (maxMixHistory + 1) * (maxMixHistory + 1) * mixPartials> host{};
        auto* scratch = thrust::raw_pointer_cast(partials.data());
        mixGram<<<dim3(elements, mixPartials), mixThreads>>>(
            thrust::raw_pointer_cast(r.data()), box.M, window, scratch);
        checkMixCuda(cudaMemcpy(host.data(), scratch, elements * mixPartials * sizeof(double), cudaMemcpyDeviceToHost));
        for (int element = 0; element < elements; ++element) {
            result[element] = 0.0;
            for (int i = 0; i < mixPartials; ++i) result[element] += host[element * mixPartials + i];
        }
    }

    double residualNorm(int slot) {
        double square = 0.0;
        gram(MixWindow{0, {slot}, {}}, &square);
        return std::sqrt(square);
    }

    bool coefficients(MixWindow& window) {
        const int n = window.history;
        const int width = n + 1;
        double gramMatrix[(maxMixHistory + 1) * (maxMixHistory + 1)]{};
        double lower[maxMixHistory][maxMixHistory]{};
        gram(window, gramMatrix);
        double scale = 0.0;
        for (int i = 0; i < n; ++i) scale = std::max(scale, gramMatrix[i * width + i]);
        if (!std::isfinite(scale) || scale == 0) return false;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double value = gramMatrix[i * width + j];
                for (int k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k];
                if (i == j) {
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

    void advance(int step) {
        const int slots = history + 1;
        const MixWindow damped{0, {head}, {}};
        auto window = damped;
        if (step >= 8 && retained > 1) {
            window.history = retained - 1;
            for (int i = 0; i < retained; ++i) {
                window.slot[i] = (head + slots - window.history + i) % slots;
            }
            if (!coefficients(window)) {
                window = damped;
                retained = 1;
                ++restarts;
            }
        }
        const int next = (head + 1) % slots;
        evaluate(window, next);
        double nextNorm = residualNorm(next);
        if (window.history && (!std::isfinite(nextNorm) || nextNorm > 1.35 * norm)) {
            ++rejections;
            ++restarts;
            retained = 1;
            evaluate(damped, next);
            nextNorm = residualNorm(next);
        } else if (window.history) {
            ++mixedSteps;
        }
        if (!std::isfinite(nextNorm)) throw std::runtime_error("Nonfinite damped mixing residual");
        norm = nextNorm;
        retained = std::min(slots, retained + 1);
        head = next;
        if (step % box.logFreq == 0) box.writeData(step);
    }
};

std::unique_ptr<AndersonMixer> createAnderson(FTS_Box& box, const BinaryModel& model) {
    const char* value = std::getenv("SCFT_MATILDA_ANDERSON");
    if (!value || !model.canonical || box.densityFieldFreq != 0 || box.chemFieldFreq != 0) return nullptr;
    const std::string mode(value);
    const int history = mode == "damped" ? 0 : mode == "3" ? 3 : mode == "5" ? 5 : -1;
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
    return std::make_unique<AndersonMixer>(box, *pressure, *flory, history);
}
