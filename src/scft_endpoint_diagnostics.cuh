struct PhaseResidualAt {
    PhysicalResidual maps;
    const double* oldA;
    const double* oldB;

    __host__ __device__ Sums operator()(int i) const {
        double closure, exchange, pore;
        maps(i, closure, exchange, pore);
        const double da = (maps.a[i].real() - oldA[i]) * maps.densityScale;
        const double db = (maps.b[i].real() - oldB[i]) * maps.densityScale;
        return {pore, closure * closure, pore * exchange * exchange, pore * da * da, pore * db * db};
    }
};

__global__ void savePhaseDensities(PhysicalResidual maps, int sites, double* a, double* b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) {
        a[i] = maps.a[i].real();
        b[i] = maps.b[i].real();
    }
}

class EndpointPhaseTrace {
    static constexpr int window = 40;
    struct Row {
        int update = 0;
        size_t evals = 0;
        bool fp64 = false;
        MixStepOutcome mix{};
        double closure = 0, exchange = 0, drift = 0;
    };
    thrust::device_vector<double> history;
    std::array<Row, window + 1> before{};
    std::ofstream stream;
    int endpoint = 0;

    void write(const Row& r) {
        stream << r.update << ',' << r.update - endpoint << ',' << r.evals << ',' << r.fp64 << ','
               << r.mix.fitPhase << ',' << r.mix.eligible << ',' << r.mix.attempted << ','
               << r.mix.accepted << ',' << r.mix.rejected << ',' << r.mix.rankReset << ','
               << r.mix.scheduledReset << ',' << r.closure << ',' << r.exchange << ',' << r.drift << '\n';
    }

public:
    explicit EndpointPhaseTrace(const Path& out, int sites)
        : history(size_t(2) * checkEvery * sites), stream(openOutput(out / "endpoint-phases.csv")) {
        stream << "updates,from_endpoint,field_evals,fp64,fit_phase,eligible,fit_attempted,fit_accepted,"
                  "raw_rejected,rank_reset,scheduled_reset,closure_rms,exchange_rms,drift_10_rms\n";
    }

    void record(const FTS_Box& box, const BinaryModel& model, int update, MixStepOutcome mix) {
        if (endpoint && update > endpoint + window) return;
        const auto maps = residualMaps(box, model);
        const int sites = box.M;
        auto* oldA = thrust::raw_pointer_cast(history.data()) + size_t(2) * (update % checkEvery) * sites;
        auto* oldB = oldA + sites;
        if (update > checkEvery) {
            const auto first = thrust::make_counting_iterator(0);
            const auto sums = thrust::transform_reduce(thrust::device, first, first + sites,
                PhaseResidualAt{maps, oldA, oldB}, Sums{}, AddSums{});
            const Row row{update, box.densityEvaluations, observedBondsInDouble(box), mix,
                          std::sqrt(sums.pressure / sites), std::sqrt(sums.exchange / sums.weight),
                          std::sqrt(std::max(sums.driftA, sums.driftB) / sums.weight)};
            if (endpoint) {
                write(row);
                stream.flush();
            } else {
                before[update % before.size()] = row;
            }
        }
        savePhaseDensities<<<(sites + mixThreads - 1) / mixThreads, mixThreads>>>(maps, sites, oldA, oldB);
    }

    void markEndpoint(int update) {
        endpoint = update;
        for (int i = std::max(update - window, checkEvery + 1); i <= update; ++i) {
            const auto& row = before[i % before.size()];
            if (row.update == i) write(row);
        }
        stream.flush();
    }
};

int observeInteger(const char* name) {
    const char* setting = std::getenv(name);
    if (!setting) return 0;
    const std::string_view value(setting);
    int result = 0;
    const auto read = std::from_chars(value.data(), value.data() + value.size(), result);
    if (read.ec != std::errc{} || read.ptr != value.data() + value.size() || result < 1) {
        throw std::runtime_error(std::string(name) + " requires a positive integer");
    }
    return result;
}

void writeDensitySnapshot(const Path& destination, const FTS_Box& box, const BinaryModel& model,
                          int update, double closure, double exchange, double drift,
                          const AndersonMixer* mixer = nullptr) {
    std::filesystem::create_directories(destination);
    const double coldSeconds = secondsSince(processStart);
    const double scale = 1.0 / (box.C * box.Nr);
    writeNpy(destination / "phi_ps.npy", box, box.Species[model.a].d_density, scale);
    writeNpy(destination / "phi_pmma.npy", box, box.Species[model.b].d_density, scale);
    if (mixer) {
        writeNpy(destination / "w_pressure.npy", box, mixer->pressure.d_wpl, 1.0, true);
        writeNpy(destination / "w_flory_plus.npy", box, mixer->flory.d_wpl, 1.0, true);
        writeNpy(destination / "w_flory_minus.npy", box, mixer->flory.d_wmi, 1.0, true);
    }
    const Path temporary = destination / "state.tmp";
    {
        auto meta = openOutput(temporary);
        meta << "{\"updates\":" << update << ",\"field_evals\":" << box.densityEvaluations
             << ",\"seconds\":" << coldSeconds << ",\"fp64\":"
             << (observedBondsInDouble(box) ? "true" : "false") << ",\"closure_rms\":" << closure
             << ",\"exchange_rms\":" << exchange << ",\"drift_rms\":" << drift << "}\n";
    }
    std::filesystem::rename(temporary, destination / "state.json");
}
