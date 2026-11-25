#include "scft_small_system.cuh"

constexpr int observationSize = 8;

Complex* mixData(DeviceField& values) { return thrust::raw_pointer_cast(values.data()); }

__global__ void difference(const Complex* a, const Complex* b, Complex* dest, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) dest[i] = a[i] - b[i];
}

__global__ void differenceQuotient(Complex* values, const Complex* base, int width, double inverseStep) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) values[i] = (values[i] - base[i]) * inverseStep;
}

__global__ void storeMixFields(MixFields fields, int sites, Complex* values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) {
        for (int c = 0; c < 3; ++c) values[c * sites + i] = fields.field[c][i];
    }
}

struct ComplexDirection {
    const Complex* u[2]{};
    double coefficient[2]{};
    int rank = 0;
};

__global__ void assignMixFields(MixFields fields, int sites, const Complex* base, ComplexDirection dir) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    for (int c = 0; c < 3; ++c) {
        const int offset = c * sites + i;
        Complex value = base[offset];
        for (int j = 0; j < dir.rank; ++j) value += dir.coefficient[j] * dir.u[j][offset];
        fields.field[c][i] = value;
    }
}

__global__ void storeRaw(MixFields fields, int sites, Complex* values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) {
        for (int c = 0; c < 3; ++c) values[c * sites + i] = -fields.timestep[c] * fields.force[c][i];
    }
}

double mixDot(const Complex* a, const Complex* b, int sites) {
    return reduceLag(3 * sites, MixLagAt{a, b, sites}).dot / sites;
}

void rebuildDensity(AndersonMixer& mixer) {
    for (auto& species : mixer.box.Species) {
        species.zeroDensity();
        species.buildPotentialField();
    }
    mixer.box.calcMoleculeDensities();
    mixer.pressure.computeForce();
    mixer.flory.computeForce();
}

Sums checkProbeState(const FTS_Box& box, const BinaryModel& model) {
    const auto maps = residualMaps(box, model);
    const ObserveNode perNode{maps.a, maps.b, maps.mask, maps.wa, maps.wb, maps.a, maps.b,
                              maps.densityScale, maps.nr, maps.chi, maps.particleCoefficient};
    const auto first = thrust::make_counting_iterator(0);
    return thrust::transform_reduce(thrust::device, first, first + box.M, perNode, Sums{}, AddSums{});
}

bool validProbeState(const Sums& sums, const BinaryModel& model, int sites) {
    const double available = 1.0 - sums.mask / sites;
    return !sums.invalid && sums.weight > 0 && sums.minFraction >= -0.005 &&
           std::abs(sums.a / sites - available * model.fractionA) <= 0.002 &&
           std::abs(sums.b / sites - available * model.fractionB) <= 0.002;
}

double normalizedPhysicalSquare(const Sums& sums, const BinaryModel& model, int sites) {
    const double exchangeScale = std::max(1.0, std::abs(model.chi)) * maximumExchange;
    return sums.pressure / (sites * maximumClosure * maximumClosure) +
           sums.exchange / (sums.weight * exchangeScale * exchangeScale);
}

using HostComplex = std::vector<Complex>;

Complex hostMean(const Complex* a, int sites) {
    Complex total{};
    for (int i = 0; i < sites; ++i) total += a[i];
    return total / static_cast<double>(sites);
}

void removeUniformModes(HostComplex& a, int sites) {
    for (int c = 0; c < 3; ++c) {
        Complex* channel = a.data() + c * sites;
        const Complex uniform = hostMean(channel, sites);
        for (int i = 0; i < sites; ++i) channel[i] -= uniform;
    }
}

double hostMixDot(const HostComplex& a, const HostComplex& b, int sites) {
    double sum = 0.0;
    for (int c = 0; c < 3; ++c) {
        const double weight = c == 2 ? 4.0 : 1.0;
        double channel = 0.0;
        for (int i = c * sites; i < (c + 1) * sites; ++i) {
            channel += a[i].real() * b[i].real() + a[i].imag() * b[i].imag();
        }
        sum += weight * channel;
    }
    return sum / sites;
}

struct ObservationWindow {
    std::array<HostComplex, observationSize> differences;
    SmallVector norms{};
    int count = 0, epoch = -1;
    bool fp64 = false;

    void record(AndersonMixer& mixer, DeviceField& scratch, bool keepVectors) {
        const int restarts = mixer.rejections + mixer.rankRestarts + mixer.explicitResets;
        if (epoch != restarts || fp64 != observedBondsInDouble(mixer.box)) count = 0;
        epoch = restarts;
        fp64 = observedBondsInDouble(mixer.box);
        if (mixer.retained < 2 || count == observationSize) return;
        const int sites = mixer.box.M, width = 3 * sites;
        const int old = (mixer.head + mixer.history) % (mixer.history + 1);
        const auto* previous = deviceData(mixer.x) + size_t(old) * width;
        const auto* now = deviceData(mixer.x) + size_t(mixer.head) * width;
        difference<<<(width + mixThreads - 1) / mixThreads, mixThreads>>>(now, previous, mixData(scratch), width);
        norms[count] = std::sqrt(mixDot(deviceData(scratch), deviceData(scratch), sites));
        if (keepVectors) {
            auto& host = differences[count];
            host.resize(width);
            checkMixCuda(cudaMemcpy(host.data(), deviceData(scratch), width * sizeof(Complex), cudaMemcpyDeviceToHost));
            removeUniformModes(host, sites);
        }
        ++count;
    }

    double cap() const {
        auto sorted = norms;
        std::sort(sorted.begin(), sorted.end());
        return sorted[observationSize / 2 - 1] + sorted[observationSize / 2];
    }
};

struct PodBasis {
    ComplexDirection directions;
    SmallVector singular{};
    SmallVector persistence{};
    bool residualSupplement = false;
};

PodBasis buildPOD(const ObservationWindow& window, const DeviceField& raw,
                  std::array<DeviceField, 2>& vectors, int sites, int requestedRank) {
    const int width = 3 * sites;
    HostComplex residual(width), direction(width);
    checkMixCuda(cudaMemcpy(residual.data(), deviceData(raw), width * sizeof(Complex), cudaMemcpyDeviceToHost));
    removeUniformModes(residual, sites);
    const double residualNorm = std::sqrt(hostMixDot(residual, residual, sites));
    SmallMatrix gram{};
    SmallVector rhs{};
    for (int i = 0; i < observationSize; ++i) {
        const auto& a = window.differences[i];
        rhs[i] = hostMixDot(a, residual, sites);
        for (int j = 0; j <= i; ++j) gram[i][j] = gram[j][i] = hostMixDot(a, window.differences[j], sites);
    }
    const auto spectrum = symmetricSpectrum(gram, observationSize);
    PodBasis result;
    std::array<int, observationSize> order{};
    for (int i = 0; i < observationSize; ++i) {
        order[i] = i;
        const double sigma = std::sqrt(std::max(0.0, spectrum.values[i]));
        result.singular[i] = sigma;
        if (!(sigma > 1e-6 * result.singular[0])) continue;
        double energy[2]{}, alignment = 0.0;
        for (int row = 0; row < observationSize; ++row) {
            double projection = 0.0;
            for (int j = 0; j < observationSize; ++j) projection += gram[row][j] * spectrum.vectors[j][i];
            energy[row / (observationSize / 2)] += projection * projection;
            alignment += rhs[row] * spectrum.vectors[row][i];
        }
        const double balance = 2 * std::sqrt(energy[0] * energy[1]) / (energy[0] + energy[1]);
        result.persistence[i] = balance * (residualNorm > 0 ? std::abs(alignment) / (residualNorm * sigma) : 0);
    }
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) {
        return result.persistence[i] > result.persistence[j];
    });
    HostComplex firstBasis;
    for (int mode : order) {
        if (!(result.singular[mode] > 1e-6 * result.singular[0])) continue;
        std::fill(direction.begin(), direction.end(), Complex{});
        for (int j = 0; j < observationSize; ++j) {
            const double coefficient = spectrum.vectors[j][mode] / result.singular[mode];
            for (int i = 0; i < width; ++i) direction[i] += coefficient * window.differences[j][i];
        }
        const int col = result.directions.rank++;
        if (col == 0 && requestedRank == 2) firstBasis = direction;
        vectors[col].resize(width);
        checkMixCuda(cudaMemcpy(mixData(vectors[col]), direction.data(), width * sizeof(Complex), cudaMemcpyHostToDevice));
        result.directions.u[col] = deviceData(vectors[col]);
        if (result.directions.rank == requestedRank) break;
    }
    if (requestedRank == 2 && result.directions.rank == 1 && residualNorm > 0) {
        const double overlap = hostMixDot(residual, firstBasis, sites);
        for (int i = 0; i < width; ++i) residual[i] -= overlap * firstBasis[i];
        const double norm = std::sqrt(hostMixDot(residual, residual, sites));
        if (norm > 1e-6 * residualNorm) {
            for (auto& value : residual) value /= norm;
            vectors[1].resize(width);
            checkMixCuda(cudaMemcpy(mixData(vectors[1]), residual.data(), width * sizeof(Complex), cudaMemcpyHostToDevice));
            result.directions.u[1] = deviceData(vectors[1]);
            result.directions.rank = 2;
            result.residualSupplement = true;
        }
    }
    return result;
}

struct PhysicalDirections {
    const double* modes[2]{};
    double pressure[2]{};
    double exchange[2]{};
    int count = 0;
};

__global__ void assignPhysicalDirections(MixFields fields, int sites, const Complex* base,
                                         PhysicalDirections direction) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    double pressure = 0.0, exchange = 0.0;
    for (int j = 0; j < direction.count; ++j) {
        pressure += direction.pressure[j] * direction.modes[j][i];
        exchange += direction.exchange[j] * direction.modes[j][i];
    }
    fields.field[0][i] = base[i] + Complex(0.0, -pressure);
    fields.field[1][i] = base[sites + i];
    fields.field[2][i] = base[2 * sites + i] + Complex(exchange, 0.0);
}

__global__ void capturePhysicalMaps(PhysicalResidual maps, int sites, double* closure, double* exchange) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) {
        double e, pore;
        maps(i, closure[i], e, pore);
        if (exchange) exchange[i] = e;
    }
}

struct PatternValue {
    PhysicalResidual maps;
    const double* center;
    bool alternate;

    __device__ double operator()(int i) const {
        double closure, exchange, pore;
        maps(i, closure, exchange, pore);
        if (alternate) return exchange;
        return center ? closure - center[i] : closure;
    }
};

__global__ void smoothPattern(PatternValue value, double* mode, int nx, int ny, int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = nx * ny;
    if (i >= plane * nz) return;
    const int x = i % nx, y = i / nx % ny, z = i / plane;
    const int xp = x == nx - 1 ? 1 - nx : 1, xm = x == 0 ? nx - 1 : -1;
    const int yp = y == ny - 1 ? 1 - ny : 1, ym = y == 0 ? ny - 1 : -1;
    const int zp = z == nz - 1 ? 1 - nz : 1, zm = z == 0 ? nz - 1 : -1;
    mode[i] = (2 * value(i) + value(i + xp) + value(i + xm) + value(i + yp * nx) +
               value(i + ym * nx) + value(i + zp * plane) + value(i + zm * plane)) / 8;
}

struct RealPatternAt {
    const double* a;
    const double* b;

    __host__ __device__ LagProducts operator()(int i) const {
        const double x = a[i];
        return {x * (b ? b[i] : 1.0), x * x, 0.0};
    }
};

__global__ void adjustPattern(double* mode, const double* basis, double center, double overlap,
                              double scale, int sites) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) mode[i] = (mode[i] - center - (basis ? overlap * basis[i] : 0.0)) * scale;
}

double normalizePattern(double* mode, const double* basis, AndersonMixer& mixer) {
    const int sites = mixer.box.M;
    const double center = reduceLag(sites, RealPatternAt{mode, nullptr}).dot / sites;
    const double overlap = basis ? reduceLag(sites, RealPatternAt{mode, basis}).dot / sites : 0.0;
    adjustPattern<<<mixer.blocks(), mixThreads>>>(mode, basis, center, overlap, 1.0, sites);
    const double norm = std::sqrt(reduceLag(sites, RealPatternAt{mode, nullptr}).current / sites);
    if (std::isfinite(norm) && norm > 1e-14) {
        adjustPattern<<<mixer.blocks(), mixThreads>>>(mode, nullptr, 0.0, 0.0, 1.0 / norm, sites);
    }
    return norm;
}

struct PhysicalProjectionAt {
    PhysicalResidual maps;
    const double* mode;
    bool exchange;

    __host__ __device__ LagProducts operator()(int i) const {
        double c, e, pore;
        maps(i, c, e, pore);
        const double v = mode[i];
        const double weight = exchange ? pore : 1.0;
        return {weight * (exchange ? e : c) * v, weight * v * v, weight};
    }
};

struct PhysicalLeakageAt {
    PhysicalResidual maps;
    const double* previous;
    const double* first;
    const double* second;
    bool exchange;

    __host__ __device__ LagProducts operator()(int i) const {
        double c, e, pore;
        maps(i, c, e, pore);
        const double v = (exchange ? e : c) - previous[i];
        const double weight = exchange ? pore : 1.0;
        return {weight * v * v, weight * v * first[i], second ? weight * v * second[i] : 0.0};
    }
};

struct PatternOverlapAt {
    PhysicalResidual maps;
    const double* first;
    const double* second;
    bool exchange;

    __host__ __device__ LagProducts operator()(int i) const {
        double c, e, pore;
        maps(i, c, e, pore);
        const double a = first[i], b = second ? second[i] : 0.0;
        const double weight = exchange ? pore : 1.0;
        return {weight * a * b, weight * a * a, weight * b * b};
    }
};

struct ProposedProjectionAt {
    const Complex* before;
    const Complex* after;
    const double* mode;
    int sites;

    __host__ __device__ LagProducts operator()(int i) const {
        return {-(after[i] - before[i]).imag() * mode[i],
                (after[2 * sites + i] - before[2 * sites + i]).real() * mode[i], 0.0};
    }
};

struct SpeciesSignAt {
    PhysicalResidual maps;
    const Complex* baseA;
    const Complex* baseB;
    const double* mode;
    double epsilon;
    bool exchange;

    __host__ __device__ LagProducts operator()(int i) const {
        const auto da = (maps.wa[i] - baseA[i]) * maps.nr;
        const auto db = (maps.wb[i] - baseB[i]) * maps.nr;
        const double v = mode[i] * epsilon;
        const auto errorA = da - (exchange ? -v : v);
        const auto errorB = db - v;
        return {v * v, errorA.real() * errorA.real() + errorA.imag() * errorA.imag(),
                errorB.real() * errorB.real() + errorB.imag() * errorB.imag()};
    }
};

class SubspaceExperiment final : public MixIntervention {
    const BinaryModel& model;
    const bool projected;
    const bool diagonal;
    const int requestedRank;
    const int sites, width, blocks;
    const bool probeDiagnostic = std::getenv("SCFT_MATILDA_PROBE_DIAGNOSTIC") != nullptr;
    int nextEvent = 80;
    int nominal = 80;
    ObservationWindow history;
    DeviceField scratch, baseRaw;
    std::array<DeviceField, 2> u, ju;
    DeviceField complement, signsA, signsB;
    std::array<thrust::device_vector<double>, 2> patterns;
    thrust::device_vector<double> originalClosure, originalExchange;
    std::ofstream trace;

    double epsilon(const Complex* base) const {
        return 1e-3 * std::max(1.0, std::sqrt(mixDot(base, base, sites)));
    }

    bool examineProjection(AndersonMixer& mixer, const Complex* base, int next) {
        auto basis = buildPOD(history, baseRaw, u, sites, requestedRank);
        auto dir = basis.directions;
        for (double value : basis.singular) trace << " sigma=" << value;
        for (double value : basis.persistence) trace << " persistence=" << value;
        trace << " basis_rank=" << dir.rank << " residual_supplement=" << basis.residualSupplement;
        if (dir.rank == 0) return false;
        const double stepSize = epsilon(base);
        SmallMatrix gram{};
        SmallVector rhs{};
        for (int j = 0; j < dir.rank; ++j) {
            dir.coefficient[j] = stepSize;
            assignMixFields<<<blocks, mixThreads>>>(mixer.fields(), sites, base, dir);
            rebuildDensity(mixer);
            ju[j].resize(width);
            storeRaw<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(ju[j]));
            differenceQuotient<<<(width + mixThreads - 1) / mixThreads, mixThreads>>>(
                mixData(ju[j]), deviceData(baseRaw), width, 1 / stepSize);
            if (probeDiagnostic) {
                dir.coefficient[j] = stepSize / 2;
                assignMixFields<<<blocks, mixThreads>>>(mixer.fields(), sites, base, dir);
                rebuildDensity(mixer);
                storeRaw<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(scratch));
                differenceQuotient<<<(width + mixThreads - 1) / mixThreads, mixThreads>>>(
                    mixData(scratch), deviceData(baseRaw), width, 2 / stepSize);
                const auto compared = reduceLag(width, MixLagAt{deviceData(ju[j]), deviceData(scratch), sites});
                trace << " epsilon_delta=" << std::sqrt(std::max(0.0, compared.current + compared.past -
                    2 * compared.dot) / std::max(compared.current, 1e-300));
            }
            dir.coefficient[j] = 0.0;
            rhs[j] = -mixDot(deviceData(ju[j]), deviceData(baseRaw), sites);
            for (int i = 0; i <= j; ++i) {
                gram[i][j] = gram[j][i] = mixDot(deviceData(ju[i]), deviceData(ju[j]), sites);
            }
        }
        const auto fit = regularizedResponse(gram, rhs, dir.rank);
        trace << " response_rank=" << fit.rank << " condition=" << fit.condition << " epsilon=" << stepSize;
        if (!fit.rank) return false;
        double square = 0.0;
        for (int i = 0; i < dir.rank; ++i) square += fit.amplitudes[i] * fit.amplitudes[i];
        if (!(square > 0) || !std::isfinite(square)) return false;
        const double scale = std::min(1.0, history.cap() / std::sqrt(square));
        for (int j = 0; j < dir.rank; ++j) dir.coefficient[j] = fit.amplitudes[j] * scale;
        assignMixFields<<<blocks, mixThreads>>>(mixer.fields(), sites, base, dir);
        rebuildDensity(mixer);
        storeRaw<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(scratch));
        const double rawNorm = mixer.residualNorm(0, deviceData(scratch));
        trace << " cap_scale=" << scale << " raw_before=" << mixer.norm << " raw_after=" << rawNorm;
        if (!(rawNorm < mixer.norm) || !validProbeState(checkProbeState(mixer.box, model), model, sites)) {
            return false;
        }
        mixer.capture(next);
        return true;
    }

    SmallVector physicalOutputs(AndersonMixer& mixer, const PhysicalDirections& directions) const {
        SmallVector output{};
        const auto maps = residualMaps(mixer.box, model);
        for (int j = 0; j < directions.count; ++j) {
            for (int channel = 0; channel < 2; ++channel) {
                const auto p = reduceLag(sites, PhysicalProjectionAt{maps, directions.modes[j], channel == 1});
                output[2 * j + channel] = p.dot / (std::sqrt(p.current * p.past) *
                        (channel == 0 ? maximumClosure : maximumExchange));
            }
        }
        return output;
    }

    double responseLeakage(AndersonMixer& mixer, const PhysicalDirections& dirs, bool exchange) const {
        const auto maps = residualMaps(mixer.box, model);
        const auto* previous = thrust::raw_pointer_cast((exchange ? originalExchange : originalClosure).data());
        const auto values = reduceLag(sites, PhysicalLeakageAt{
            maps, previous, dirs.modes[0], dirs.modes[1], exchange});
        const auto gram = reduceLag(sites, PatternOverlapAt{
            maps, dirs.modes[0], dirs.modes[1], exchange});
        if (!(values.dot > 0 && gram.current > 0)) return 0.0;
        double retained = values.current * values.current / gram.current;
        if (dirs.count == 2) {
            const double independent = gram.past - gram.dot * gram.dot / gram.current;
            const double projection = values.past - gram.dot * values.current / gram.current;
            if (independent > 1e-12 * gram.past) retained += projection * projection / independent;
        }
        return std::sqrt(std::max(0.0, 1 - retained / values.dot));
    }

    void setPhysicalFields(AndersonMixer& mixer, PhysicalDirections dirs, bool density) {
        assignPhysicalDirections<<<blocks, mixThreads>>>(mixer.fields(), sites, deviceData(complement), dirs);
        if (density) rebuildDensity(mixer);
        else for (auto& species : mixer.box.Species) species.buildPotentialField();
    }

    double makePattern(AndersonMixer& mixer, int index, const double* center, bool alternate) {
        auto* values = thrust::raw_pointer_cast(patterns[index].data());
        const auto& n = mixer.box.Nx;
        smoothPattern<<<blocks, mixThreads>>>({residualMaps(mixer.box, model), center, alternate},
                                             values, n[0], n[1], n[2]);
        return normalizePattern(values, index == 0 ? nullptr : thrust::raw_pointer_cast(patterns[0].data()), mixer);
    }

    SmallVector parentProjection(const Complex* before, const PhysicalDirections& dirs,
                                 const Complex* after) const {
        SmallVector values{};
        for (int i = 0; i < dirs.count; ++i) {
            const auto p = reduceLag(sites, ProposedProjectionAt{before, after, dirs.modes[i], sites});
            values[2 * i] = p.dot / sites;
            values[2 * i + 1] = p.current / sites;
        }
        return values;
    }

    bool examineCoupling(AndersonMixer& mixer, MixWindow window, const Complex* base, int next) {
        const auto before = checkProbeState(mixer.box, model);
        const double physicalBefore = normalizedPhysicalSquare(before, model, sites);
        if (!(makePattern(mixer, 0, nullptr, false) > 1e-14) &&
            !(makePattern(mixer, 0, nullptr, true) > 1e-14)) return false;
        PhysicalDirections dirs{{thrust::raw_pointer_cast(patterns[0].data()), nullptr}, {}, {}, 1};
        applyMix<<<blocks, mixThreads>>>(mixer.fields(), sites, deviceData(mixer.x), deviceData(mixer.r), window);
        storeMixFields<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(scratch));
        const double stepSize = epsilon(base);
        SmallVector parent{};
        const auto prepareComplement = [&] {
            parent = parentProjection(base, dirs, deviceData(scratch));
            auto orthogonal = dirs;
            for (int i = 0; i < dirs.count; ++i) {
                orthogonal.pressure[i] = -parent[2 * i];
                orthogonal.exchange[i] = -parent[2 * i + 1];
            }
            assignPhysicalDirections<<<blocks, mixThreads>>>(mixer.fields(), sites, deviceData(scratch), orthogonal);
            storeMixFields<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(complement));
            rebuildDensity(mixer);
        };
        prepareComplement();
        if (requestedRank == 4) {
            auto* reference = thrust::raw_pointer_cast(originalClosure.data());
            capturePhysicalMaps<<<blocks, mixThreads>>>(residualMaps(mixer.box, model), sites, reference, nullptr);
            auto pressureProbe = dirs;
            pressureProbe.pressure[0] = stepSize;
            setPhysicalFields(mixer, pressureProbe, true);
            double leaked = makePattern(mixer, 1, reference, false);
            if (!(leaked > 1e-10)) leaked = makePattern(mixer, 1, nullptr, true);
            trace << " offmode_closure_norm=" << leaked;
            if (leaked > 1e-14) {
                dirs.count = 2;
                dirs.modes[1] = thrust::raw_pointer_cast(patterns[1].data());
            }
            prepareComplement();
        }
        signsA = mixer.box.Species[model.a].d_w;
        signsB = mixer.box.Species[model.b].d_w;
        const int columns = 2 * dirs.count;
        trace << " basis_rank=" << columns << " epsilon=" << stepSize;
        const auto outputs = physicalOutputs(mixer, dirs);
        if (probeDiagnostic) {
            auto identity = dirs;
            for (int j = 0; j < dirs.count; ++j) {
                identity.pressure[j] = parent[2 * j];
                identity.exchange[j] = parent[2 * j + 1];
            }
            assignPhysicalDirections<<<blocks, mixThreads>>>(mixer.fields(), sites, deviceData(complement), identity);
            storeMixFields<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(baseRaw));
            difference<<<(width + mixThreads - 1) / mixThreads, mixThreads>>>(
                deviceData(baseRaw), deviceData(scratch), mixData(baseRaw), width);
            const double identityError = std::sqrt(mixDot(deviceData(baseRaw), deviceData(baseRaw), sites));
            trace << " parent_identity_error=" << identityError;
            if (!(identityError <= 1e-12 * std::max(1.0, stepSize / 1e-3))) return false;
            assignPhysicalDirections<<<blocks, mixThreads>>>(mixer.fields(), sites, deviceData(complement), dirs);
            capturePhysicalMaps<<<blocks, mixThreads>>>(residualMaps(mixer.box, model), sites,
                thrust::raw_pointer_cast(originalClosure.data()), thrust::raw_pointer_cast(originalExchange.data()));
        }
        SmallMatrix jacobian{}, measured{};
        for (int j = 0; j < columns; ++j) {
            const int mode = j / 2;
            auto probe = dirs;
            (j % 2 ? probe.exchange[mode] : probe.pressure[mode]) = stepSize;
            setPhysicalFields(mixer, probe, false);
            const auto signs = reduceLag(sites, SpeciesSignAt{residualMaps(mixer.box, model),
                deviceData(signsA), deviceData(signsB), dirs.modes[mode], stepSize, j % 2 != 0});
            const double signError = std::sqrt((signs.current + signs.past) / signs.dot);
            trace << " sign_error=" << signError;
            if (!(signError <= 1e-7)) return false;
            rebuildDensity(mixer);
            const auto values = physicalOutputs(mixer, dirs);
            for (int i = 0; i < columns; ++i) {
                measured[i][j] = (values[i] - outputs[i]) / stepSize;
                jacobian[i][j] = diagonal && i != j ? 0.0 : measured[i][j];
                trace << " J" << i << j << '=' << measured[i][j];
            }
            if (probeDiagnostic) {
                trace << " closure_leakage=" << responseLeakage(mixer, dirs, false)
                      << " exchange_leakage=" << responseLeakage(mixer, dirs, true);
                (j % 2 ? probe.exchange[mode] : probe.pressure[mode]) = stepSize / 2;
                setPhysicalFields(mixer, probe, true);
                const auto half = physicalOutputs(mixer, dirs);
                double delta = 0.0, size = 0.0;
                for (int i = 0; i < columns; ++i) {
                    const double error = 2 * (half[i] - outputs[i]) / stepSize - measured[i][j];
                    delta += error * error;
                    size += measured[i][j] * measured[i][j];
                }
                trace << " epsilon_delta=" << std::sqrt(delta / std::max(size, 1e-300));
            }
        }
        SmallMatrix gram{};
        SmallVector rhs{};
        // Fit in orthonormal field coordinates, with an exchange norm of two.
        for (int i = 0; i < columns; ++i) {
            const double wi = i % 2 ? 2.0 : 1.0;
            for (int k = 0; k < columns; ++k) rhs[i] -= jacobian[k][i] * outputs[k] / wi;
            for (int j = 0; j <= i; ++j) {
                const double wj = j % 2 ? 2.0 : 1.0;
                for (int k = 0; k < columns; ++k) gram[i][j] += jacobian[k][i] * jacobian[k][j] / (wi * wj);
                gram[j][i] = gram[i][j];
            }
        }
        const auto fit = regularizedResponse(gram, rhs, columns);
        trace << " response_rank=" << fit.rank << " condition=" << fit.condition;
        if (!fit.rank) return false;
        SmallVector difference{};
        double square = 0.0;
        for (int i = 0; i < columns; ++i) {
            const double weight = i % 2 ? 2.0 : 1.0;
            difference[i] = fit.amplitudes[i] - weight * parent[i];
            square += difference[i] * difference[i];
        }
        if (!std::isfinite(square)) return false;
        const double scale = square > 0 ? std::min(1.0, history.cap() / std::sqrt(square)) : 1.0;
        for (int i = 0; i < dirs.count; ++i) {
            dirs.pressure[i] = parent[2 * i] + scale * difference[2 * i];
            dirs.exchange[i] = parent[2 * i + 1] + scale * difference[2 * i + 1] / 2;
        }
        setPhysicalFields(mixer, dirs, true);
        const auto after = checkProbeState(mixer.box, model);
        const double physicalAfter = normalizedPhysicalSquare(after, model, sites);
        storeRaw<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(baseRaw));
        const double rawNorm = mixer.residualNorm(0, deviceData(baseRaw));
        const auto actual = physicalOutputs(mixer, dirs);
        double predictionError = 0.0, actualChange = 0.0;
        for (int i = 0; i < columns; ++i) {
            double predicted = outputs[i];
            for (int j = 0; j < dirs.count; ++j) {
                predicted += measured[i][2 * j] * dirs.pressure[j] + measured[i][2 * j + 1] * dirs.exchange[j];
            }
            predictionError += (predicted - actual[i]) * (predicted - actual[i]);
            actualChange += (actual[i] - outputs[i]) * (actual[i] - outputs[i]);
        }
        trace << " cap_scale=" << scale << " raw_before=" << mixer.norm << " raw_after=" << rawNorm
              << " physical_before=" << physicalBefore << " physical_after=" << physicalAfter
              << " combined_error=" << std::sqrt(predictionError / std::max(actualChange, 1e-300));
        if (!(std::isfinite(rawNorm) && rawNorm <= 1.35 * mixer.norm && physicalAfter < physicalBefore) ||
            !validProbeState(after, model, sites)) return false;
        mixer.capture(next);
        return true;
    }

public:
    SubspaceExperiment(const BinaryModel& model_, AndersonMixer& mixer, const Path& out)
        : model(model_), projected(mixer.trial == MixTrial::Rpm1 || mixer.trial == MixTrial::Rpm2),
          diagonal(mixer.trial == MixTrial::BlockDiag),
          requestedRank(mixer.trial == MixTrial::Rpm1 ? 1 : mixer.trial == MixTrial::Block4 ? 4 : 2),
          sites(mixer.box.M), width(3 * sites), blocks(mixer.blocks()), scratch(width), baseRaw(width),
          trace(openOutput(out / "subspace-events.log")) {
        if (!projected) {
            if (mixer.box.returnDimension() != 3 || mixer.flory.typeI != "A" || mixer.flory.typeJ != "B") {
                throw std::runtime_error("Physical coupling requires a 3D A/B Flory direction");
            }
            complement.resize(width);
            patterns[0].resize(sites);
            if (requestedRank == 4) {
                patterns[1].resize(sites);
            }
            if (requestedRank == 4 || probeDiagnostic) originalClosure.resize(sites);
            if (probeDiagnostic) originalExchange.resize(sites);
        }
        trace << "requested_rank=" << requestedRank << " projected=" << projected << " diagonal=" << diagonal
              << " diagnostic=" << probeDiagnostic << '\n' << std::flush;
    }

    bool attempt(AndersonMixer& mixer, MixWindow window, int update, int next) override {
        if (nextEvent - update >= observationSize) return false;
        history.record(mixer, scratch, projected);
        if (update < nextEvent) return false;
        if (history.count < observationSize) {
            ++nextEvent;
            return false;
        }
        const auto start = Clock::now();
        const auto evals = mixer.box.densityEvaluations;
        const auto* base = deviceData(mixer.x) + size_t(mixer.head) * width;
        trace << "update=" << update << " nominal=" << nominal << " fp64=" << observedBondsInDouble(mixer.box)
              << " cap=" << history.cap();
        if (projected) storeRaw<<<blocks, mixThreads>>>(mixer.fields(), sites, mixData(baseRaw));
        const bool accepted = projected ? examineProjection(mixer, base, next)
                                        : examineCoupling(mixer, window, base, next);
        trace << " accepted=" << accepted << " probe_evals=" << mixer.box.densityEvaluations - evals
              << " seconds=" << secondsSince(start) << '\n' << std::flush;
        history.count = 0;
        do { nominal += 160; } while (nominal <= update);
        nextEvent = nominal;
        return accepted;
    }
};
