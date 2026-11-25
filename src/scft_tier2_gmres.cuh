__global__ void gmresAxpy(Complex* target, const Complex* source, double scale, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) target[i] += scale * source[i];
}

__global__ void gmresScale(Complex* target, const Complex* source, double scale, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) target[i] = scale * source[i];
}

struct GmresLeastSquares {
    std::array<std::array<double, 3>, 4> h{};
    std::array<double, 3> cosines{}, sines{};
    std::array<double, 4> rhs{};
    int rank = 0;

    bool column(int j) {
        for (int i = 0; i < j; ++i) {
            const double a = h[i][j], b = h[i + 1][j];
            h[i][j] = cosines[i] * a + sines[i] * b;
            h[i + 1][j] = -sines[i] * a + cosines[i] * b;
        }
        const double length = std::hypot(h[j][j], h[j + 1][j]);
        if (!(length > 0) || !std::isfinite(length)) return false;
        cosines[j] = h[j][j] / length;
        sines[j] = h[j + 1][j] / length;
        h[j][j] = length;
        h[j + 1][j] = 0;
        rhs[j + 1] = -sines[j] * rhs[j];
        rhs[j] *= cosines[j];
        ++rank;
        return true;
    }

    std::array<double, 3> solve() const {
        std::array<double, 3> answer{};
        for (int i = rank - 1; i >= 0; --i) {
            double value = rhs[i];
            for (int j = i + 1; j < rank; ++j) value -= h[i][j] * answer[j];
            answer[i] = value / h[i][i];
        }
        return answer;
    }
};

class GmresExperiment final : public MixIntervention {
    const BinaryModel& model;
    const int sites, width, blocks, vectorBlocks, first;
    const bool diagnostic = std::getenv("SCFT_MATILDA_RESIDUAL_TRACE") != nullptr;
    std::array<DeviceField, 3> basis;
    DeviceField raw, work, correction, half;
    std::array<double, 8> stepNorms{};
    int stepCount = 0;
    bool fp64;

    void syncPrecision(AndersonMixer& mix) {
        const bool current = observedBondsInDouble(mix.box);
        if (current != fp64) stepCount = 0;
        fp64 = current;
    }

    double cap() const {
        const int count = std::min(stepCount, int(stepNorms.size()));
        if (!count) return 0;
        auto sorted = stepNorms;
        std::sort(sorted.begin(), sorted.begin() + count);
        return sorted[count / 2] + sorted[(count - 1) / 2];
    }

    void probe(AndersonMixer& mix, const Complex* base, double epsilon, int column, DeviceField& result) {
        const ComplexDirection direction{{deviceData(basis[column]), nullptr}, {epsilon, 0}, 1};
        assignMixFields<<<blocks, mixThreads>>>(mix.fields(), sites, base, direction);
        rebuildDensity(mix);
        storeRaw<<<blocks, mixThreads>>>(mix.fields(), sites, mixData(result));
        // J of F = -raw; the Arnoldi right hand side is raw at the base.
        differenceQuotient<<<vectorBlocks, mixThreads>>>(mixData(result), deviceData(raw), width, -1 / epsilon);
    }

public:
    GmresExperiment(const BinaryModel& model_, AndersonMixer& mix)
        : model(model_), sites(mix.box.M), width(3 * sites), blocks(mix.blocks()),
          vectorBlocks((width + mixThreads - 1) / mixThreads),
          first(std::getenv("SCFT_MATILDA_GMRES_FIRST_AT")
                ? observeInteger("SCFT_MATILDA_GMRES_FIRST_AT") : 80),
          raw(width), work(width), correction(width), fp64(observedBondsInDouble(mix.box)) {
        if (first < 10 || first % 2) throw std::runtime_error("GMRES first attempt requires a scheduled plain update");
        for (auto& v : basis) v.resize(width);
        if (diagnostic) half.resize(width);
    }

    bool attempt(AndersonMixer& mix, MixWindow window, int update, int next) override {
        if (window.history || update < first || (update - first) % 160) return false;
        syncPrecision(mix);
        const double radius = cap();
        const auto begin = mix.box.densityEvaluations;
        const auto finish = [&](bool accepted, const std::string& detail) {
            mix.researchEvent(update, "gmres3", accepted, int(mix.box.densityEvaluations - begin), detail);
            return accepted;
        };
        if (!(radius > 0)) return finish(false, "no_recent_step_norms");

        const auto* base = deviceData(mix.x) + size_t(mix.head) * width;
        storeRaw<<<blocks, mixThreads>>>(mix.fields(), sites, mixData(raw));
        const double beta = std::sqrt(mixDot(deviceData(raw), deviceData(raw), sites));
        if (!(beta > 0) || !std::isfinite(beta)) return finish(false, "unusable_base_force");
        const auto baseState = checkProbeState(mix.box, model);
        const double physical = normalizedPhysicalSquare(baseState, model, sites);
        if (!validProbeState(baseState, model, sites) || !std::isfinite(physical)) {
            return finish(false, "unusable_base_physical");
        }
        const double epsilon = 1e-3 * std::max(1.0, std::sqrt(mixDot(base, base, sites)));
        gmresScale<<<vectorBlocks, mixThreads>>>(mixData(basis[0]), deviceData(raw), 1 / beta, width);
        GmresLeastSquares fit;
        fit.rhs[0] = beta;
        double maxOrtho = 0, halfError = 0;
        for (int j = 0; j < 3; ++j) {
            probe(mix, base, epsilon, j, work);
            if (diagnostic && mix.researchAttempts == 0) {
                probe(mix, base, epsilon / 2, j, half);
                const auto p = reduceLag(width, MixLagAt{deviceData(work), deviceData(half), sites});
                halfError = std::max(halfError, std::sqrt(std::max(0.0, p.current + p.past - 2 * p.dot) /
                                                        std::max(1e-300, p.current)));
            }
            const double rawColumn = std::sqrt(mixDot(deviceData(work), deviceData(work), sites));
            for (int i = 0; i <= j; ++i) {
                fit.h[i][j] = mixDot(deviceData(work), deviceData(basis[i]), sites);
                gmresAxpy<<<vectorBlocks, mixThreads>>>(
                    mixData(work), deviceData(basis[i]), -fit.h[i][j], width);
            }
            const double tail = std::sqrt(mixDot(deviceData(work), deviceData(work), sites));
            fit.h[j + 1][j] = tail;
            if (!std::isfinite(tail) || !fit.column(j) || !(tail > 1e-12 * rawColumn) || j == 2) break;
            gmresScale<<<vectorBlocks, mixThreads>>>(mixData(basis[j + 1]), deviceData(work), 1 / tail, width);
            if (diagnostic) {
                for (int i = 0; i <= j; ++i) {
                    maxOrtho = std::max(maxOrtho, std::abs(mixDot(deviceData(basis[i]),
                                                                 deviceData(basis[j + 1]), sites)));
                }
            }
        }
        if (fit.rank == 0) return finish(false, "singular_arnoldi");
        const auto coefficient = fit.solve();
        double length = 0;
        for (int i = 0; i < fit.rank; ++i) length = std::hypot(length, coefficient[i]);
        if (!(length > 0) || !std::isfinite(length)) return finish(false, "invalid_step");
        const double scale = std::min(1.0, radius / length);
        gmresScale<<<vectorBlocks, mixThreads>>>(mixData(correction), deviceData(basis[0]), scale * coefficient[0], width);
        for (int i = 1; i < fit.rank; ++i) {
            gmresAxpy<<<vectorBlocks, mixThreads>>>(mixData(correction), deviceData(basis[i]), scale * coefficient[i], width);
        }
        const ComplexDirection step{{deviceData(correction), nullptr}, {1, 0}, 1};
        assignMixFields<<<blocks, mixThreads>>>(mix.fields(), sites, base, step);
        rebuildDensity(mix);
        storeRaw<<<blocks, mixThreads>>>(mix.fields(), sites, mixData(work));
        const double rawAfter = mix.residualNorm(0, deviceData(work));
        const auto proposal = checkProbeState(mix.box, model);
        const double after = normalizedPhysicalSquare(proposal, model, sites);
        const bool accepted = rawAfter < mix.norm && std::isfinite(after) && after <= physical &&
            validProbeState(proposal, model, sites);
        if (accepted) mix.capture(next);
        std::ostringstream detail;
        detail << "rank=" << fit.rank << " epsilon=" << epsilon << " model_residual=" << std::abs(fit.rhs[fit.rank])
               << " half_error=" << halfError << " max_orthogonality=" << maxOrtho
               << " cap_scale=" << scale << " raw_before=" << mix.norm << " raw_after=" << rawAfter
               << " physical_before=" << physical << " physical_after=" << after;
        return finish(accepted, detail.str());
    }

    void committed(AndersonMixer& mix, int next, bool corrected) override {
        syncPrecision(mix);
        if (corrected) stepCount = 0;
        const auto* base = deviceData(mix.x) + size_t(mix.head) * width;
        const auto* accepted = deviceData(mix.x) + size_t(next) * width;
        difference<<<vectorBlocks, mixThreads>>>(accepted, base, mixData(work), width);
        const double length = std::sqrt(mixDot(deviceData(work), deviceData(work), sites));
        if (std::isfinite(length)) stepNorms[stepCount++ % stepNorms.size()] = length;
    }
};
