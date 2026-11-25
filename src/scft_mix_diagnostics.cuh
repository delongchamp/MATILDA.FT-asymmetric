struct PhysicalResidual {
    const Complex *a, *b, *mask, *wa, *wb;
    double densityScale, nr, chi, particleCoefficient;

    __host__ __device__ void operator()(int i, double& closure, double& exchange, double& pore) const {
        const double x = a[i].real() * densityScale;
        const double y = b[i].real() * densityScale;
        const double m = mask ? mask[i].real() * densityScale : 0.0;
        const double weight = fmin(fmax(1.0 - m, 0.0), 1.0);
        pore = weight >= 0.1 ? weight : 0.0;
        closure = x + y + m - 1.0;
        exchange = ((wa[i].real() * nr - wb[i].real() * nr) - chi * (y - x) -
                    particleCoefficient * m) / fmax(1.0, fabs(chi));
    }
};

PhysicalResidual residualMaps(const FTS_Box& box, const BinaryModel& model) {
    const auto& a = box.Species[model.a];
    const auto& b = box.Species[model.b];
    return {deviceData(a.d_density), deviceData(b.d_density),
            model.np >= 0 ? deviceData(box.Species[model.np].d_density) : nullptr,
            deviceData(a.d_w), deviceData(b.d_w), 1.0 / (box.C * box.Nr), box.Nr,
            model.chi, box.Nr * model.deltaParticleChi};
}

struct LagProducts {
    double dot = 0, current = 0, past = 0;

    __host__ __device__ LagProducts operator+(const LagProducts& b) const {
        return {dot + b.dot, current + b.current, past + b.past};
    }
};

struct AddLagProducts {
    __host__ __device__ LagProducts operator()(LagProducts a, LagProducts b) const { return a + b; }
};

struct MixLagAt {
    const Complex* current;
    const Complex* past;
    int sites;

    __host__ __device__ LagProducts operator()(int i) const {
        const auto a = current[i], b = past[i];
        const double weight = i / sites == 2 ? 4.0 : 1.0;
        return {weight * (a.real() * b.real() + a.imag() * b.imag()),
                weight * (a.real() * a.real() + a.imag() * a.imag()),
                weight * (b.real() * b.real() + b.imag() * b.imag())};
    }
};

struct PhysicalLagAt {
    const double* current;
    const double* past;
    PhysicalResidual maps;
    bool exchange;

    __host__ __device__ LagProducts operator()(int i) const {
        double weight = 1.0;
        if (exchange) {
            double unusedClosure, unusedExchange;
            maps(i, unusedClosure, unusedExchange, weight);
        }
        const double a = current[i], b = past[i];
        return {weight * a * b, weight * a * a, weight * b * b};
    }
};

__global__ void saveResidualLag(MixFields fields, PhysicalResidual maps, int sites,
                                Complex* raw, double* physical) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    for (int c = 0; c < 3; ++c) raw[c * sites + i] = -fields.timestep[c] * fields.force[c][i];
    double pore;
    if (physical) maps(i, physical[i], physical[sites + i], pore);
}

double signedCosine(LagProducts p) {
    return p.current > 0 && p.past > 0
        ? p.dot / std::sqrt(p.current * p.past)
        : std::numeric_limits<double>::quiet_NaN();
}

template<class Product>
LagProducts reduceLag(int count, Product product) {
    const auto first = thrust::make_counting_iterator(0);
    return thrust::transform_reduce(thrust::device, first, first + count, product, LagProducts{}, AddLagProducts{});
}

class MixLagTrace {
    std::vector<std::pair<int, int>> windows;
    DeviceField raw;
    thrust::device_vector<double> physical;
    std::ofstream stream;
    int previousUpdate = -1, previousRestarts = -1, count = 0;
    bool previousPrecision = false;

public:
    MixLagTrace(const Path& out, int sites, const char* option)
        : raw(size_t(9) * sites), physical(size_t(6) * sites),
          stream(openOutput(out / "mix-lags.csv")) {
        std::string_view ranges(option);
        while (!ranges.empty()) {
            const auto comma = ranges.find(',');
            const auto range = ranges.substr(0, comma);
            const auto colon = range.find(':');
            const auto first = range.data();
            const auto last = first + range.size();
            if (colon == std::string_view::npos) throw std::runtime_error("Expected lag window FIRST:LAST");
            int lo = 0, hi = 0;
            const auto minRead = std::from_chars(first, first + colon, lo);
            const auto maxRead = std::from_chars(first + colon + 1, last, hi);
            if (minRead.ec != std::errc{} || minRead.ptr != first + colon ||
                maxRead.ec != std::errc{} || maxRead.ptr != last || lo < 1 || hi < lo) {
                throw std::runtime_error("Invalid signed lag window");
            }
            windows.emplace_back(lo, hi);
            if (comma == std::string_view::npos) break;
            ranges.remove_prefix(comma + 1);
        }
        stream << "updates,evals,fp64,seconds,lag,raw_cos,fitted_cos,gram_cos,closure_cos,exchange_cos\n";
    }

    void record(int updates, AndersonMixer& mixer, const BinaryModel& model) {
        const bool inWindow = std::any_of(windows.begin(), windows.end(), [updates](auto range) {
            return updates >= range.first && updates <= range.second;
        });
        if (!inWindow) return;
        const int restarts = mixer.rejections + mixer.rankRestarts + mixer.explicitResets + mixer.corrections;
        const bool fp64 = mixer.box.fullPrecisionBonds;
        if (updates != previousUpdate + 1 || fp64 != previousPrecision || restarts != previousRestarts) count = 0;
        previousUpdate = updates;
        previousRestarts = restarts;
        previousPrecision = fp64;
        const int sites = mixer.box.M, slot = updates % 3;
        const int width = 3 * sites, ring = mixer.history + 1;
        auto* rawNow = thrust::raw_pointer_cast(raw.data()) + size_t(slot) * width;
        auto* physicalNow = thrust::raw_pointer_cast(physical.data()) + size_t(slot) * 2 * sites;
        const auto maps = residualMaps(mixer.box, model);
        saveResidualLag<<<mixer.blocks(), mixThreads>>>(mixer.fields(), maps, sites, rawNow, physicalNow);
        const double fittedNorm = mixer.pressureWeights.empty() ? mixer.norm : mixer.residualNorm(mixer.head);
        for (int lag = 1; lag <= std::min({count, 2, mixer.retained - 1}); ++lag) {
            const int oldSlot = (slot + 3 - lag) % 3;
            const auto* rawPast = deviceData(raw) + size_t(oldSlot) * width;
            const auto* physicalPast = thrust::raw_pointer_cast(physical.data()) + size_t(oldSlot) * 2 * sites;
            MixWindow window{lag, {}, {}};
            for (int j = 0; j <= lag; ++j) window.slot[j] = (mixer.head + ring - lag + j) % ring;
            AndersonMixer::GramMatrix matrix{};
            mixer.gram(window, matrix.data());
            double b = 0.0, h = 0.0;
            for (int i = 0; i < lag; ++i) {
                b += matrix[i * (lag + 1) + lag];
                for (int j = 0; j <= i; ++j) h += (i == j ? 1.0 : 2.0) * matrix[i * (lag + 1) + j];
            }
            const double c = fittedNorm * fittedNorm;
            const double derived = signedCosine({c - b, c, c - 2 * b + h});
            const auto* fNow = deviceData(mixer.r) + size_t(mixer.head) * width;
            const auto* fPast = deviceData(mixer.r) + size_t(window.slot[0]) * width;
            const double direct = signedCosine(reduceLag(width, MixLagAt{fNow, fPast, sites}));
            if (std::isfinite(direct) && (!std::isfinite(derived) || std::abs(direct - derived) > 1e-8)) {
                throw std::runtime_error("Fitted lag Gram disagrees with direct signed cosine");
            }
            stream << updates << ',' << mixer.box.densityEvaluations << ',' << fp64 << ','
                   << secondsSince(processStart) << ',' << lag << ','
                   << signedCosine(reduceLag(width, MixLagAt{rawNow, rawPast, sites})) << ','
                   << direct << ',' << derived << ','
                   << signedCosine(reduceLag(sites, PhysicalLagAt{physicalNow, physicalPast, maps, false})) << ','
                   << signedCosine(reduceLag(sites, PhysicalLagAt{physicalNow + sites, physicalPast + sites, maps, true}))
                   << '\n' << std::flush;
        }
        count = std::min(count + 1, 2);
    }
};
