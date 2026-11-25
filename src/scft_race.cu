// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "scft_race.h"
#include "FTS_Box.h"
#include "fts_molecule_linear.h"
#include "fts_potential_flory.h"
#include "fts_potential_incompress.h"
#include "fts_potentialParticle.h"
#include "fts_species.h"
#include "git-version.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

void die(const char*);

namespace {
using Complex = thrust::complex<double>;
using DeviceField = thrust::device_vector<Complex>;
using Clock = std::chrono::steady_clock;
using Path = std::filesystem::path;

const auto processStart = Clock::now();
constexpr double maximumClosure = 0.002;
constexpr double maximumExchange = 0.002;
constexpr double maximumDrift = 0.001;
constexpr int checkEvery = 10;

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::ofstream openOutput(const Path& filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17);
    return out;
}

void writeNpy(const Path& filename, const FTS_Box& box,
              const thrust::host_vector<Complex>& field, double scale,
              bool complexValues = false) {
    // NumPy's v1 header uses a little-endian length. The Linux CUDA targets use
    // little-endian IEEE doubles. Reversing Nx preserves the x-fastest layout.
    std::string header = "{'descr': '" + std::string(complexValues ? "<c16" : "<f8") +
        "', 'fortran_order': False, 'shape': (";
    for (int i = static_cast<int>(box.Nx.size()) - 1; i >= 0; --i) {
        header += std::to_string(box.Nx[i]) + ", ";
    }
    header += "), }";
    header.resize(header.size() + (63 - (10 + header.size()) % 64), ' ');
    header += '\n';
    const char tag[10] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0,
        static_cast<char>(header.size() & 255), static_cast<char>(header.size() >> 8)};
    auto out = openOutput(filename);
    out.write(tag, sizeof(tag));
    out.write(header.data(), header.size());
    std::vector<double> values;
    values.reserve(field.size() * (complexValues ? 2 : 1));
    for (const auto& value : field) {
        values.push_back(value.real() * scale);
        if (complexValues) values.push_back(value.imag() * scale);
    }
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

struct BinaryModel {
    int a = -1, b = -1, np = -1;
    double chi = 0.0;
    double deltaParticleChi = 0.0;
    double fractionA = 0.0;
    double fractionB = 0.0;
    bool canonical = false;

    explicit BinaryModel(FTS_Box& box) {
        for (std::size_t i = 0; i < box.Species.size(); ++i) {
            const auto& name = box.Species[i].fts_species;
            if (name == "A") a = i;
            else if (name == "B") b = i;
            else if (name == "NP") np = i;
            else throw std::runtime_error("The race adapter expects A, B and optional NP");
        }
        if (box.returnFTSstyle() != "scft" || a < 0 || b < 0 || box.C <= 0 || box.Nr <= 0) {
            throw std::runtime_error("The race adapter requires a binary SCFT model");
        }
        int floryCount = 0, incompressCount = 0;
        for (auto* potential : box.Potentials) {
            if (const auto* flory = dynamic_cast<const PotentialFlory*>(potential)) {
                const bool ab = flory->typeI == "A" && flory->typeJ == "B";
                const bool ba = flory->typeI == "B" && flory->typeJ == "A";
                if (!ab && !ba) throw std::runtime_error("Expected a single A/B Flory term");
                chi = flory->chiN;
                ++floryCount;
            } else if (const auto* particle = dynamic_cast<const PotentialParticle*>(potential)) {
                if (particle->typeJ != "NP" || (particle->typeI != "A" && particle->typeI != "B")) {
                    throw std::runtime_error("Expected NP affinity for A or B");
                }
                deltaParticleChi += particle->typeI == "A" ? particle->chi : -particle->chi;
            } else if (potential->printStyle() == "Incompress") {
                ++incompressCount;
            } else {
                throw std::runtime_error("Unsupported force in the race adapter");
            }
        }
        if (floryCount != 1 || incompressCount != 1) {
            throw std::runtime_error("Expected one strict pressure and one Flory term");
        }

        bool grand = false;
        for (const auto* molecule : box.Molecs) {
            if (molecule->molec_type == "HParticle") continue;
            const auto* chain = dynamic_cast<const LinearMolec*>(molecule);
            if (chain == nullptr || chain->doSmear) {
                throw std::runtime_error("The race adapter expects unsmeared linear polymers");
            }
            if (chain->activity > 0) {
                grand = true;
                continue;
            }
            if (chain->phi < 0) throw std::runtime_error("Supply a canonical molecular fraction");
            canonical = true;
            for (int block = 0; block < chain->numBlocks; ++block) {
                const double fraction = chain->phi * chain->N[block] / chain->Ntot;
                if (chain->intSpecies[block] == a) fractionA += fraction;
                else if (chain->intSpecies[block] == b) fractionB += fraction;
                else throw std::runtime_error("Expected a polymer block of A or B");
            }
        }
        if (canonical == grand || (canonical && std::abs(fractionA + fractionB - 1.0) > 1e-8)) {
            throw std::runtime_error("Expected one ensemble with a complete polymer composition");
        }
    }
};

#include "scft_anderson.cuh"

struct Sums {
    double weight = 0.0, pressure = 0.0, exchange = 0.0, driftA = 0.0, driftB = 0.0;
    double a = 0.0, b = 0.0, mask = 0.0, minFraction = 0.0;
    double fieldImag = 0.0, densityImag = 0.0;
    int invalid = 0;
};

struct AddSums {
    __host__ __device__ Sums operator()(const Sums& x, const Sums& y) const {
        return {x.weight + y.weight, x.pressure + y.pressure, x.exchange + y.exchange,
            x.driftA + y.driftA, x.driftB + y.driftB, x.a + y.a, x.b + y.b, x.mask + y.mask,
            fmin(x.minFraction, y.minFraction), fmax(x.fieldImag, y.fieldImag),
            fmax(x.densityImag, y.densityImag), x.invalid | y.invalid};
    }
};

struct ObserveNode {
    const Complex *a, *b, *mask, *wa, *wb, *oldA, *oldB;
    double invDensity, nr, chi, particleCoefficient;

    __host__ __device__ Sums operator()(int i) const {
        const auto x = a[i] * invDensity;
        const auto y = b[i] * invDensity;
        const auto m = mask ? mask[i] * invDensity : Complex(0.0);
        const auto u = wa[i] * nr;
        const auto v = wb[i] * nr;
        const double di = fmax(fmax(fabs(x.imag()), fabs(y.imag())), fabs(m.imag()));
        const double wi = fmax(fabs(u.imag()), fabs(v.imag()));
        if (!(isfinite(x.real()) && isfinite(y.real()) && isfinite(m.real()) &&
              isfinite(u.real()) && isfinite(v.real()) && isfinite(x.imag()) &&
              isfinite(y.imag()) && isfinite(m.imag()) && isfinite(u.imag()) &&
              isfinite(v.imag()))) {
            Sums result;
            result.invalid = 1;
            return result;
        }
        const double weight = fmin(fmax(1.0 - m.real(), 0.0), 1.0);
        const double pore = weight >= 0.1 ? weight : 0.0;
        const double pressure = x.real() + y.real() + m.real() - 1.0;
        const double exchange = (u.real() - v.real()) - chi * (y.real() - x.real()) -
                                particleCoefficient * m.real();
        const double da = x.real() - oldA[i].real() * invDensity;
        const double db = y.real() - oldB[i].real() * invDensity;
        return {pore, pressure * pressure, pore * exchange * exchange,
                pore * da * da, pore * db * db, x.real(), y.real(), m.real(),
                fmin(x.real(), y.real()), wi, di, 0};
    }
};

const Complex* deviceData(const DeviceField& field) {
    return thrust::raw_pointer_cast(field.data());
}

void writePartitions(const Path& path, const FTS_Box& box) {
    auto out = openOutput(path);
    out << "{\"single_chain_partition\":[";
    bool first = true;
    for (const auto* molecule : box.Molecs) {
        if (molecule->molec_type == "HParticle") continue;
        if (!first) out << ',';
        first = false;
        out << molecule->Q.real();
    }
    out << "]}\n";
}

void readFrozenField(const Path& path, DeviceField& dest, bool complexField) {
    std::ifstream in(path, std::ios::binary);
    thrust::host_vector<Complex> values(dest.size());
    for (auto& value : values) {
        double real = 0.0, imaginary = 0.0;
        if (!in.read(reinterpret_cast<char*>(&real), sizeof(real)) || !std::isfinite(real)
            || (complexField && (!in.read(reinterpret_cast<char*>(&imaginary), sizeof(imaginary))
                                || !std::isfinite(imaginary)))) {
            throw std::runtime_error("Invalid frozen per-site field " + path.string());
        }
        value = Complex(real, imaginary);
    }
    if (in.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Unexpected trailing data in " + path.string());
    }
    dest = values;
}

void runFixed(FTS_Box& box, const BinaryModel& model, const Path& out) {
    const char* input = std::getenv("SCFT_MATILDA_FIXED_DIR");
    if (input == nullptr) throw std::runtime_error("Set SCFT_MATILDA_FIXED_DIR");
    const Path root(input);
    auto& a = box.Species[model.a];
    auto& b = box.Species[model.b];
    const char* complex = std::getenv("SCFT_MATILDA_FIXED_COMPLEX");
    const bool complexFields = complex && std::string_view(complex) == "1";
    readFrozenField(root / "w_ps.raw", a.d_w, complexFields);
    readFrozenField(root / "w_pmma.raw", b.d_w, complexFields);
    for (auto& species : box.Species) species.zeroDensity();
    box.calcMoleculeDensities();
    const double scale = 1.0 / (box.C * box.Nr);
    writeNpy(out / "phi_ps.npy", box, a.d_density, scale);
    writeNpy(out / "phi_pmma.npy", box, b.d_density, scale);
    writePartitions(out / "fixed.json", box);
}

void runMask(FTS_Box& box, const Path& out) {
    const auto particle = std::find_if(box.Species.begin(), box.Species.end(),
        [](const auto& species) { return species.fts_species == "NP"; });
    if (particle == box.Species.end() || box.C <= 0 || box.Nr <= 0) {
        throw std::runtime_error("Mask export requires a normalized NP species");
    }
    const thrust::host_vector<Complex> mask = particle->d_density;
    const double timeToMask = secondsSince(processStart);
    writeNpy(out / "np_fraction.npy", box, mask, 1.0 / (box.C * box.Nr));
    auto meta = openOutput(out / "mask.json");
    meta << "{\"native_sites\":" << box.M << ",\"accessible_fraction\":" << box.Vfree / box.V
         << ",\"time_to_mask_s\":" << timeToMask << "}\n";
}

void runObserved(FTS_Box& box, const BinaryModel& model, const Path& out, int nsteps,
                 bool stopAtNative) {
    const auto& a = box.Species[model.a];
    const auto& b = box.Species[model.b];
    const auto* particle = model.np >= 0 ? &box.Species[model.np] : nullptr;
    const double scale = 1.0 / (box.C * box.Nr);
    scftRaceStage("observer_snapshots_begin", 0, box.densityEvaluations);
    DeviceField lastA = a.d_density, lastB = b.d_density;
    thrust::host_vector<Complex> startA = a.d_w, startB = b.d_w;
    auto trace = openOutput(out / "observer.csv");
    trace << "updates,field_evals,closure_rms,exchange_rms,drift_rms,canonical_loading_error,"
             "phi_ps_mean,phi_pmma_mean,min_fraction,max_abs_imag_W,max_abs_imag_phi,check_pass\n";
    int updates = 0, nativeStop = 0, passing = 0;
    double closure = 0.0, exchange = 0.0, drift = 0.0;
    bool converged = false;
    const auto solveStart = Clock::now();
    const char* requested = std::getenv("SCFT_MATILDA_ANDERSON");
    const bool requireMixer = std::getenv("SCFT_MATILDA_REQUIRE_ANDERSON") != nullptr;
    if (requireMixer && (stopAtNative || !requested || andersonDepth(requested) < 0)) {
        throw std::runtime_error("Required Anderson controller has an unsupported request");
    }
    scftRaceStage("controller_begin", 0, box.densityEvaluations);
    auto mixer = stopAtNative ? nullptr : createAnderson(box, model);
    std::cout << "Controller requested " << (requested ? requested : "auto") << " resolved "
              << (mixer ? std::to_string(mixer->history) : "EMPC") << std::endl;
    if (requireMixer && !mixer) {
        throw std::runtime_error("Required Anderson controller did not resolve");
    }
    scftRaceStage("controller_ready", 0, box.densityEvaluations);
    const char* precisionMode = std::getenv("SCFT_MATILDA_FP32_BOND");
    const bool polish = precisionMode && std::string(precisionMode) == "polish" && !stopAtNative;
    const char* forcePolish = std::getenv("SCFT_MATILDA_DIAGNOSTIC_FP64_AT");
    const int forcedUpdate = forcePolish ? std::atoi(forcePolish) : 0;
    if (forcePolish && (!polish || forcedUpdate < checkEvery || forcedUpdate > nsteps ||
                        forcedUpdate % checkEvery || std::to_string(forcedUpdate) != forcePolish)) {
        throw std::runtime_error("Diagnostic FP64 handover requires polish at a common check");
    }
    int polishUpdate = 0;
    for (int step = 0; step < nsteps; ++step) {
        if (mixer) mixer->advance(step);
        else box.doTimeStep(step);
        ++updates;
        const bool native = step >= 50 && !nativeStop && box.converged(step);
        if (native) nativeStop = updates;
        if (updates % checkEvery == 0) {
            const auto first = thrust::make_counting_iterator(0);
            const ObserveNode perNode{deviceData(a.d_density), deviceData(b.d_density),
                particle ? deviceData(particle->d_density) : nullptr,
                deviceData(a.d_w), deviceData(b.d_w), deviceData(lastA), deviceData(lastB),
                scale, box.Nr, model.chi, box.Nr * model.deltaParticleChi};
            auto sums = thrust::transform_reduce(thrust::device, first, first + box.M,
                                                 perNode, Sums{}, AddSums{});
            if (sums.invalid || sums.weight <= 0) {
                throw std::runtime_error("The observer found nonfinite fields or an empty pore");
            }
            closure = std::sqrt(sums.pressure / box.M);
            const bool startPolish = polish && !box.fullPrecisionBonds &&
                (forcePolish ? updates == forcedUpdate : closure <= 0.004);
            if (startPolish) {
                if (forcePolish) {
                    auto fp32 = openOutput(out / "fp32_handover.json");
                    fp32 << "{\"updates\":" << updates
                         << ",\"field_evals\":" << box.densityEvaluations
                         << ",\"closure_rms\":" << closure
                         << ",\"exchange_rms\":"
                         << std::sqrt(sums.exchange / sums.weight) / std::max(1.0, std::abs(model.chi))
                         << ",\"drift_rms\":"
                         << std::sqrt(std::max(sums.driftA, sums.driftB) / sums.weight) << "}\n";
                }
                scftRaceStage("fp64_begin", updates, box.densityEvaluations);
                box.fullPrecisionBonds = true;
                polishUpdate = updates;
                passing = 0;
                for (auto& species : box.Species) species.zeroDensity();
                box.calcMoleculeDensities();
                scftRaceStage("fp64_density_done", updates, box.densityEvaluations);
                if (mixer) mixer->resetFromCurrentState();
                sums = thrust::transform_reduce(thrust::device, first, first + box.M, perNode, Sums{}, AddSums{});
                if (sums.invalid || sums.weight <= 0) throw std::runtime_error("Invalid FP64 polish state");
                closure = std::sqrt(sums.pressure / box.M);
            }
            exchange = std::sqrt(sums.exchange / sums.weight) / std::max(1.0, std::abs(model.chi));
            drift = std::sqrt(std::max(sums.driftA, sums.driftB) / sums.weight);
            const double available = 1.0 - sums.mask / box.M;
            const double meanA = sums.a / box.M, meanB = sums.b / box.M;
            const double loadingError = model.canonical
                ? std::max(std::abs(meanA - available * model.fractionA),
                           std::abs(meanB - available * model.fractionB)) : 0.0;
            const bool pass = (!polish || (box.fullPrecisionBonds && !startPolish)) &&
                std::isfinite(closure + exchange + drift + loadingError) &&
                loadingError <= 0.002 && sums.minFraction >= -0.005 &&
                closure <= maximumClosure && exchange <= maximumExchange && drift <= maximumDrift;
            passing = pass ? passing + 1 : 0;
            converged = passing >= 2;
            trace << updates << ',' << box.densityEvaluations << ',' <<
                closure << ',' << exchange << ',' << drift << ',' << loadingError << ',' <<
                meanA << ',' << meanB << ',' << sums.minFraction << ',' <<
                sums.fieldImag << ',' << sums.densityImag << ',' << pass << '\n' << std::flush;
            lastA = a.d_density;
            lastB = b.d_density;
            scftRaceStage(box.fullPrecisionBonds ? "fp64_check" : "initial_precision_check",
                          updates, box.densityEvaluations);
        } else {
            scftRaceStage("update", updates, box.densityEvaluations);
        }
        if ((stopAtNative && native) || (!stopAtNative && converged)) break;
        ++box.totSteps;
    }
    const auto result = cudaDeviceSynchronize();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    const double solveSeconds = secondsSince(solveStart);
    scftRaceStage("host_copy_begin", updates, box.densityEvaluations);
    const thrust::host_vector<Complex> finalA = a.d_density, finalB = b.d_density;
    const double timeToFields = secondsSince(processStart);
    scftRaceStage("fields_ready", updates, box.densityEvaluations);
    const auto exportStart = Clock::now();
    writeNpy(out / "phi_ps.npy", box, finalA, scale);
    writeNpy(out / "phi_pmma.npy", box, finalB, scale);
    writeNpy(out / "initial_w_ps.npy", box, startA, 1.0, true);
    writeNpy(out / "initial_w_pmma.npy", box, startB, 1.0, true);
    writeNpy(out / "w_ps.npy", box, a.d_w, 1.0, true);
    writeNpy(out / "w_pmma.npy", box, b.d_w, 1.0, true);
    if (particle) writeNpy(out / "np_fraction.npy", box, particle->d_density, scale);
    writePartitions(out / "fixed.json", box);
    scftRaceStage("export_done", updates, box.densityEvaluations);
    auto meta = openOutput(out / "solver.json");
    meta << "{\n  \"revision\": \"" << MY_GIT_VERSION << "\",\n  \"converged\": " <<
        (converged && !forcePolish && updates % checkEvery == 0 ? "true" : "false") <<
        ",\n  \"updates\": " << updates << ",\n  \"field_evals\": " <<
        box.densityEvaluations << ",\n  \"native_stop_update\": ";
    if (nativeStop) meta << nativeStop;
    else meta << "null";
    meta << ",\n  \"last_common_check_update\": " << updates / checkEvery * checkEvery <<
        ",\n  \"time_to_fields_s\": " << timeToFields <<
        ",\n  \"diagnostic_fp64_switch\": " << (forcePolish ? "true" : "false") <<
        ",\n  \"solve_s\": " << solveSeconds <<
        ",\n  \"export_s\": " << secondsSince(exportStart);
    if (updates >= checkEvery && std::isfinite(closure + exchange + drift)) {
        meta << ",\n  \"closure_rms\": " << closure << ",\n  \"exchange_rms\": " << exchange <<
            ",\n  \"drift_rms\": " << drift;
    }
    if (mixer) {
        meta << ",\n  \"anderson_history\": " << mixer->history
             << ",\n  \"anderson_mixed_steps\": " << mixer->mixedSteps
             << ",\n  \"anderson_rejections\": " << mixer->rejections
             << ",\n  \"anderson_restarts\": " << mixer->rankRestarts + mixer->rejections
             << ",\n  \"anderson_rank_restarts\": " << mixer->rankRestarts
             << ",\n  \"anderson_evictions\": " << mixer->evictions
             << ",\n  \"anderson_precision_resets\": " << (polishUpdate ? 1 : 0)
             << ",\n  \"anderson_coefficient_clips\": " << mixer->clippedSteps;
        if (mixer->cacheGram) meta << ",\n  \"anderson_gram_reused_entries\": " << mixer->reusedGramEntries;
    }
    if (polishUpdate) meta << ",\n  \"fp64_bond_from_update\": " << polishUpdate;
    meta << "\n}\n";
}
}

void scftRaceStage(const char* phase, int updates, int evaluations, std::size_t requestedBytes) {
    static const char* directory = std::getenv("SCFT_RUN_DIR");
    static const bool enabled = directory && std::getenv("SCFT_MATILDA_STAGES");
    if (!enabled) return;
    static auto trace = openOutput(Path(directory) / "stages.csv");
    static const bool header = static_cast<bool>(trace <<
        "phase,seconds,updates,evals,cuda_free_bytes,cuda_total_bytes,requested_bytes\n");
    (void) header;
    const double seconds = secondsSince(processStart);
    std::size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    trace << phase << ',' << seconds << ',' << updates << ',' << evaluations << ','
          << freeBytes << ',' << totalBytes << ',' << requestedBytes << '\n' << std::flush;
}

void runScftRace(FTS_Box& box, int nsteps, const char* mode) {
    try {
        const char* output = std::getenv("SCFT_RUN_DIR");
        if (output == nullptr) throw std::runtime_error("Set SCFT_RUN_DIR to the guarded run");
        const std::string operation(mode);
        if (operation == "mask") {
            runMask(box, output);
            return;
        }
        const BinaryModel model(box);
        if (operation == "fixed") runFixed(box, model, output);
        else if (operation == "observe" || operation == "native-observe") {
            runObserved(box, model, output, nsteps, operation == "native-observe");
        } else {
            throw std::runtime_error("Select fixed, mask, observe or native-observe");
        }
    } catch (const std::exception& error) {
        die(error.what());
    }
}
