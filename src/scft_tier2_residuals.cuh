void writeSignedNpy(const Path& path, const FTS_Box& box, const thrust::device_vector<double>& field) {
    const thrust::host_vector<double> host(field);
    auto output = openOutput(path);
    writeNpyHeader(output, box, false);
    output.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(double));
}

class Tier2ResidualTrace {
    const Path root;
    const std::string id, segment;
    const double affinity;
    const int budget;
    const int limit, requestedProbe;
    thrust::device_vector<double> history, closureMap, exchangeMap;
    std::ofstream phases;
    int epoch = 0, generationStart = 0, lateUpdate = 0;

    static std::string runId() {
        const char* value = std::getenv("SCFT_MATILDA_RUN_ID");
        if (!value || !*value || std::string_view(value).find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string_view::npos) {
            throw std::runtime_error("Residual tracing requires an alphanumeric SCFT_MATILDA_RUN_ID");
        }
        return value;
    }

    bool captureSample(int update, bool fitPhase) {
        if (requestedProbe && (update == requestedProbe - 1 || update == requestedProbe)) return true;
        if (update == 0 || update == 79 || update == 80 || update == 159 || update == 160) return true;
        if (!lateUpdate && fitPhase && update > 180 &&
            ((budget && secondsSince(processStart) >= budget - 4) || update >= limit - 4)) {
            lateUpdate = update;
        }
        return lateUpdate && (update == lateUpdate || update == lateUpdate + 1);
    }

public:
    Tier2ResidualTrace(const Path& out, int sites, int budgetSeconds, int nsteps)
        : root(out / "residual-states"), id(runId()),
          segment(std::getenv("SCFT_MATILDA_STAGE") ? std::getenv("SCFT_MATILDA_STAGE") : "target"),
          affinity(std::getenv("SCFT_MATILDA_AFFINITY_SCALE")
                    ? std::stod(std::getenv("SCFT_MATILDA_AFFINITY_SCALE")) : 1.0),
          budget(budgetSeconds), limit(nsteps),
          requestedProbe(observeInteger("SCFT_MATILDA_GMRES_FIRST_AT")),
          history(size_t(2) * checkEvery * sites), closureMap(sites), exchangeMap(sites),
          phases(openOutput(out / "residual-phases.csv")) {
        phases << "updates,field_evals,time_s,epoch,precision,phase,fit_attempted,fit_accepted,"
                  "raw_rejected,rank_reset,closure_rms,exchange_rms,drift_10_rms,raw_norm,fit_norm\n";
    }

    void record(const FTS_Box& box, const BinaryModel& model, AndersonMixer& mixer,
                int update, MixStepOutcome mix, bool switched) {
        if (switched) {
            ++epoch;
            generationStart = update;
        }
        const auto maps = residualMaps(box, model);
        const auto first = thrust::make_counting_iterator(0);
        const size_t offset = size_t(2) * (update % checkEvery) * box.M;
        auto* oldA = thrust::raw_pointer_cast(history.data()) + offset;
        auto* oldB = oldA + box.M;
        const bool haveLag = update > checkEvery && update - generationStart >= checkEvery;
        const auto sums = thrust::transform_reduce(thrust::device, first, first + box.M,
            PhaseResidualAt{maps, oldA, oldB}, Sums{}, AddSums{});
        const double closure = std::sqrt(sums.pressure / box.M);
        const double exchange = std::sqrt(sums.exchange / sums.weight);
        const auto phase = update == 0 ? "initial" : switched ? "handover" :
            mix.fitPhase ? "fit" : mix.eligible ? "plain" : "warmup";
        const auto precision = observedBondsInDouble(box) ? "fp64" : "fp32";
        const double now = secondsSince(processStart);
        const double fit = mixer.pressureWeights.empty() ? mixer.norm : mixer.residualNorm(mixer.head);
        phases << update << ',' << box.densityEvaluations << ',' << now << ',' << epoch << ','
               << precision << ',' << phase << ',' << mix.attempted << ',' << mix.accepted << ','
               << mix.rejected << ',' << mix.rankReset << ',' << closure << ',' << exchange << ',';
        if (haveLag) phases << std::sqrt(std::max(sums.driftA, sums.driftB) / sums.weight);
        phases << ',' << mixer.norm << ',' << fit << '\n' << std::flush;
        savePhaseDensities<<<(box.M + mixThreads - 1) / mixThreads, mixThreads>>>(maps, box.M, oldA, oldB);
        if (!captureSample(update, mix.fitPhase)) return;

        const auto directory = root / ("u" + std::to_string(update));
        std::filesystem::create_directories(directory);
        capturePhysicalMaps<<<(box.M + mixThreads - 1) / mixThreads, mixThreads>>>(maps, box.M,
            thrust::raw_pointer_cast(closureMap.data()), thrust::raw_pointer_cast(exchangeMap.data()));
        writeSignedNpy(directory / "closure.npy", box, closureMap);
        writeSignedNpy(directory / "exchange.npy", box, exchangeMap);
        auto meta = openOutput(directory / "sample.json");
        meta << "{\"run_id\":\"" << id << "\",\"segment\":\"" << segment << "\",\"update\":" << update
             << ",\"stage_update\":" << update << ",\"field_evals\":" << box.densityEvaluations
             << ",\"state_kind\":\"committed\",\"phase\":\"" << phase << "\",\"precision\":\""
             << precision << "\",\"wall_s\":" << now << ",\"affinity_scale\":" << affinity << ",\"epoch\":" << epoch
             << ",\"closure_rms\":" << closure << ",\"exchange_rms\":" << exchange << "}\n";
    }
};
