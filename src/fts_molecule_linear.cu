// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_molecule_linear.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include "FTS_Box.h"
#include "fts_species.h"
void die(const char*);

namespace {
bool optionEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string_view(value) == "1";
}
}

namespace {
using Complex = thrust::complex<double>;

void requireCuda(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void requireFft(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Stream bond cuFFT failure " + std::to_string(result));
    }
}

__global__ void bondBlockWeight(const Complex* field, Complex* weights, Complex* first, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Complex w = exp(-field[i]);
    weights[i] = w;
    if (first) first[i] = w;
}

__global__ void bondMultiply(const Complex* lhs, const Complex* rhs, Complex* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] * rhs[i];
}
}

struct StreamBondPlan {
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t complete = nullptr;
    cufftHandle fft = 0;
    size_t workspaceBytes = 0;

    explicit StreamBondPlan(FTS_Box& box, int batch = 1, cufftType type = CUFFT_Z2Z) {
        try {
            requireCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            requireCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            requireCuda(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming));
            requireFft(cufftCreate(&fft));
            if (batch == 2) {
                const int dim = box.returnDimension();
                int lengths[3];
                for (int j = 0; j < dim; ++j) lengths[j] = box.Nx[dim - 1 - j];
                requireFft(cufftMakePlanMany(fft, dim, lengths, lengths, 1, box.M,
                    lengths, 1, box.M, type, batch, &workspaceBytes));
            } else if (box.returnDimension() == 2) {
                requireFft(cufftMakePlan2d(fft, box.Nx[1], box.Nx[0], CUFFT_Z2Z, &workspaceBytes));
            } else {
                requireFft(cufftMakePlan3d(fft, box.Nx[2], box.Nx[1], box.Nx[0], CUFFT_Z2Z, &workspaceBytes));
            }
            requireFft(cufftSetStream(fft, stream));
            std::cout << "Stream bond cuFFT workspace bytes " << workspaceBytes << std::endl;
        } catch (...) {
            release();
            throw;
        }
    }

    ~StreamBondPlan() { release(); }

    void release() {
        if (stream) cudaStreamSynchronize(stream);
        if (fft) cufftDestroy(fft);
        if (ready) cudaEventDestroy(ready);
        if (complete) cudaEventDestroy(complete);
        if (stream) cudaStreamDestroy(stream);
    }

    void enqueueZ2Z(Complex* input, Complex* output, int direction) const {
        requireFft(cufftExecZ2Z(fft, reinterpret_cast<cufftDoubleComplex*>(input),
            reinterpret_cast<cufftDoubleComplex*>(output), direction));
    }

    void enqueueC2C(cufftComplex* input, cufftComplex* output, int direction) const {
        requireFft(cufftExecC2C(fft, input, output, direction));
    }
};

struct StreamBondExecution {
    std::shared_ptr<StreamBondPlan> plan;
    const bool capture;
    bool warmed = false;
    bool exclusive = false;
    bool captureFailed = false;
    cudaGraphExec_t graph = nullptr;
    std::vector<const Complex*> capturedFields;
    Complex* capturedForward = nullptr;
    const Complex* capturedReverse = nullptr;

    explicit StreamBondExecution(bool useGraph) : capture(useGraph) {}
    ~StreamBondExecution() {
        if (graph) cudaGraphExecDestroy(graph);
    }
};

namespace {
__global__ void seedBatch(const Complex* first, const Complex* second,
                          Complex* frontier, Complex* firstHistory, Complex* secondHistory, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    firstHistory[i] = frontier[i] = first[i];
    secondHistory[i] = frontier[count + i] = second[i];
}

__global__ void scaleBatch(const Complex* symbol, Complex* spectrum, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    spectrum[i] = symbol[i] * spectrum[i];
    spectrum[count + i] = symbol[i] * spectrum[count + i];
}

__global__ void finishBatch(const Complex* first, const Complex* second,
                            Complex* frontier, Complex* firstHistory, Complex* secondHistory, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    firstHistory[i] = frontier[i] = frontier[i] * first[i];
    secondHistory[i] = frontier[count + i] = frontier[count + i] * second[i];
}

__global__ void seedFloatBatch(const Complex* first, const Complex* second,
                               cufftComplex* frontier, Complex* firstHistory, Complex* secondHistory, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    firstHistory[i] = first[i];
    secondHistory[i] = second[i];
    frontier[i] = make_cuFloatComplex(static_cast<float>(first[i].real()), static_cast<float>(first[i].imag()));
    frontier[count + i] = make_cuFloatComplex(static_cast<float>(second[i].real()), static_cast<float>(second[i].imag()));
}

__global__ void scaleFloatBatch(const Complex* foldedSymbol, cufftComplex* spectrum, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float folded = static_cast<float>(foldedSymbol[i].real());
    spectrum[i].x *= folded;
    spectrum[i].y *= folded;
    spectrum[count + i].x *= folded;
    spectrum[count + i].y *= folded;
}

__global__ void finishFloatBatch(const Complex* first, const Complex* second, cufftComplex* frontier,
                                 Complex* firstHistory, Complex* secondHistory, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto a = frontier[i], b = frontier[count + i];
    const Complex left = Complex(a.x, a.y) * first[i];
    const Complex right = Complex(b.x, b.y) * second[i];
    firstHistory[i] = left;
    secondHistory[i] = right;
    frontier[i] = make_cuFloatComplex(static_cast<float>(left.real()), static_cast<float>(left.imag()));
    frontier[count + i] = make_cuFloatComplex(static_cast<float>(right.real()), static_cast<float>(right.imag()));
}
}

struct BatchedBondExecution {
    StreamBondPlan plan;
    thrust::device_vector<Complex> frontier, spectrum;
    thrust::device_vector<cufftComplex> floatFrontier, floatSpectrum;
    const bool mixed;
    bool warmed = false;
    bool captureFailed = false;
    cudaGraphExec_t graph = nullptr;
    std::vector<const Complex*> capturedInputs, inputs;
    BatchedBondExecution(FTS_Box& box, bool mixed_)
        : plan(box, 2, mixed_ ? CUFFT_C2C : CUFFT_Z2Z),
          frontier(mixed_ ? 0 : size_t(2) * box.M), spectrum(frontier.size()),
          floatFrontier(mixed_ ? size_t(2) * box.M : 0), floatSpectrum(floatFrontier.size()), mixed(mixed_) {}
    ~BatchedBondExecution() {
        if (graph) cudaGraphExecDestroy(graph);
    }
};

LinearMolec::~LinearMolec() = default;

LinearMolec::LinearMolec(std::istringstream& iss, FTS_Box* p_box)
    : FTS_Molec(iss, p_box),
      directBond(optionEnabled("SCFT_MATILDA_BOND_DIRECT")),
      foldBondScale(optionEnabled("SCFT_MATILDA_BOND_FOLD_SCALE")) {
    // iss comes into this routine having already passed "molecule" and "linear"
    // structure of iss should be:
    // numBlocks, integer
    // Ns for block 1, type for block 1. Type is a char, should be one of the species types
    // Ns for block 2, type for block 2...
    // Repeat for number of blocks


    iss >> numBlocks;

    Ntot = 0;
    doSmear = 0;
    blockSpecies.resize(numBlocks);
    intSpecies.resize(numBlocks);
    d_intSpecies.resize(numBlocks);
    d_N.resize(numBlocks);
    N.resize(numBlocks);

    // Resize density arrays for multiblocks
    density.resize(numBlocks*mybox->M);
    d_density.resize(numBlocks*mybox->M);
    d_cDensity.resize(numBlocks*mybox->M);

    for ( int j=0 ; j<numBlocks; j++ ) {
        iss >> N[j];
        iss >> blockSpecies[j];

        Ntot += N[j];
    }

    if ( iss.tellg() != -1 ) {
        std::string s1;
        iss >> s1;
        if ( s1 == "smear" || s1 == "Smear" ) {
            iss >> s1 ; 
            doSmear = 1;
            if ( s1 == "Gaussian" || s1 == "gaussian" ) {
                iss >> smearLength;
                std::cout << "Smearing with unit Gaussian with width " << smearLength << std::endl;

                smearFunc.resize(mybox->M);
                d_smearFunc.resize(mybox->M);

                // initializes Gaussian smear in k-space
                mybox->initSmearGaussian(smearFunc, 1.0, smearLength);
                mybox->writeTComplexGridData("smearGaussian.dat", smearFunc);
                
                // Send smear to the device
                d_smearFunc = smearFunc;
            }
        }
    }

    // Copy block lengths to device
    d_N = N;

    // Both ends use the same free-end condition. Compare the block sequence in reverse.
    for ( int j=0 ; j<numBlocks/2 ; j++ ) {
        if ( ( blockSpecies[j] != blockSpecies[numBlocks-1-j] ) ||    // Block Species are unequal
             (N[j] != N[numBlocks-1-j]) )  {                    // Block lengths are unequal
             
            isSymmetric = false;
            break;
        }
    }
    shareReverse = isSymmetric && optionEnabled("SCFT_MATILDA_SHARE_REVERSE");
    d_q.resize(Ntot * mybox->M);
    if (!shareReverse) d_qdag.resize(d_q.size());

    // Determine integer block Species
    for ( int j=0 ; j<numBlocks; j++ ) {
        intSpecies[j] = -55;
        for ( int i=0 ; i<mybox->Species.size(); i++ ) {
            if ( blockSpecies[j] == mybox->Species[i].fts_species ) {
                intSpecies[j] = i;
            }
        }
        if ( intSpecies[j] < 0 ) {
            std::string lastWords = blockSpecies[j] + " species not found!";
            die(lastWords.c_str());
        }
    }

    d_intSpecies = intSpecies;



    // Initialize bond potential
    thrust::host_vector<thrust::complex<double>> bond_fft(mybox->M);
    const double bondScale = foldBondScale ? 1.0 / double(mybox->M) : 1.0;

    double k2, *kv;
    kv = new double[mybox->returnDimension()];
    for ( int i=0 ; i<mybox->M ; i++ ) {
        k2 = mybox->get_kD(i, kv);

        //bond_fft[i] = exp(-k2/6.0);
        bond_fft[i] = exp(-k2/(mybox->Nr-1.0)) * bondScale;
    }
    delete kv;

    // Send bond potential to device
    d_bond_fft = bond_fft;


    // Allocate temp storage data
    if (doSmear) W.resize(mybox->M);
    if (!directBond) qf.resize(mybox->M);
    expW.resize(mybox->M);
    fuseDensity = !doSmear && optionEnabled("SCFT_MATILDA_FUSE_DENSITY");
    const char* streamMode = std::getenv("SCFT_MATILDA_BOND_GRAPH");
    const int dimensions = mybox->returnDimension();
    if (!doSmear && directBond && foldBondScale && Ntot > 0 && streamMode
        && (dimensions == 2 || dimensions == 3)) {
        const std::string_view mode(streamMode);
        if (mode == "1" || mode == "2") {
            streamBond = std::make_unique<StreamBondExecution>(mode == "2");
        }
    }
    if (!fuseDensity) {
        q_qdag.resize(mybox->M);
        temp.resize(mybox->M);
    }
}

thrust::device_ptr<const thrust::complex<double>> LinearMolec::reverseHistory() const {
    return shareReverse ? d_q.data() : d_qdag.data();
}

struct NegExponential {
    __host__ __device__
        thrust::complex<double> operator()(const thrust::complex<double> &z) {
            return exp(-z);
    }
};

void LinearMolec::propagateBondStep(thrust::device_ptr<const thrust::complex<double>> previous,
                                   thrust::device_ptr<thrust::complex<double>> next) {
    const int extent = mybox->M;
    auto forwardInput = previous;
    auto inverseOutput = next;
    if (!directBond) {
        thrust::copy(previous, previous + extent, qf.begin());
        forwardInput = qf.data();
        inverseOutput = qf.data();
    }

    mybox->cufftTransformDouble(thrust::raw_pointer_cast(forwardInput),
                               thrust::raw_pointer_cast(a.data()), 1, !foldBondScale);
    thrust::transform(d_bond_fft.begin(), d_bond_fft.end(), a.begin(), hf.begin(),
                      thrust::multiplies<thrust::complex<double>>());

    // In direct mode the inverse writes a disjoint history slice before its in-place site weight.
    mybox->cufftTransformDouble(thrust::raw_pointer_cast(hf.data()),
                               thrust::raw_pointer_cast(inverseOutput), -1, false);
    thrust::transform(inverseOutput, inverseOutput + extent, expW.begin(), next,
                      thrust::multiplies<thrust::complex<double>>());
}

void LinearMolec::enqueueStreamSweeps() {
    const auto& plan = *streamBond->plan;
    const auto stream = plan.stream;
    const int count = mybox->M;
    constexpr int threads = 256;
    const int grid = (count + threads - 1) / threads;
    auto* weights = thrust::raw_pointer_cast(expW.data());
    auto* frequency = thrust::raw_pointer_cast(a.data());
    auto* product = thrust::raw_pointer_cast(hf.data());
    const auto* symbol = thrust::raw_pointer_cast(d_bond_fft.data());
    auto* forward = thrust::raw_pointer_cast(d_q.data());
    auto* reverse = thrust::raw_pointer_cast(d_qdag.data());
    const auto sweep = [&](Complex* history, bool backwards) {
        size_t site = 0;
        for (int i = 0; i < numBlocks; ++i) {
            const int block = backwards ? numBlocks - 1 - i : i;
            if (!N[block]) continue;
            const auto* field = thrust::raw_pointer_cast(mybox->Species[intSpecies[block]].d_w.data());
            bondBlockWeight<<<grid, threads, 0, stream>>>(field, weights, site == 0 ? history : nullptr, count);
            for (int j = 0; j < N[block]; ++j, ++site) {
                if (site == 0) continue;
                auto* next = history + site * count;
                plan.enqueueZ2Z(next - count, frequency, CUFFT_FORWARD);
                bondMultiply<<<grid, threads, 0, stream>>>(symbol, frequency, product, count);
                plan.enqueueZ2Z(product, next, CUFFT_INVERSE);
                bondMultiply<<<grid, threads, 0, stream>>>(next, weights, next, count);
            }
        }
    };
    sweep(forward, false);
    if (!isSymmetric) sweep(reverse, true);
    else if (!shareReverse) {
        requireCuda(cudaMemcpyAsync(reverse, forward, d_q.size() * sizeof(Complex),
                                     cudaMemcpyDeviceToDevice, stream));
    }
    requireCuda(cudaPeekAtLastError());
}

void LinearMolec::submitStreamPropagators(bool join, bool exclusive) {
    if (a.empty()) {
        a.resize(mybox->M);
        hf.resize(mybox->M);
    }
    auto& execution = *streamBond;
    if (!execution.plan || (exclusive && !execution.exclusive)) {
        if (execution.graph) requireCuda(cudaGraphExecDestroy(execution.graph));
        execution.graph = nullptr;
        execution.warmed = false;
        execution.exclusive = exclusive;
        execution.plan = exclusive ? nullptr : mybox->streamBondPlan.lock();
        if (!execution.plan) {
            execution.plan = std::make_shared<StreamBondPlan>(*mybox);
            if (!exclusive) mybox->streamBondPlan = execution.plan;
        }
    }
    const auto& plan = *execution.plan;
    requireCuda(cudaEventRecord(plan.ready, nullptr));
    requireCuda(cudaStreamWaitEvent(plan.stream, plan.ready));

    if (execution.capture && !execution.captureFailed && execution.warmed) {
        bool same = execution.graph
            && execution.capturedForward == thrust::raw_pointer_cast(d_q.data())
            && execution.capturedReverse == thrust::raw_pointer_cast(reverseHistory());
        for (int b = 0; same && b < numBlocks; ++b) {
            const auto* field = thrust::raw_pointer_cast(mybox->Species[intSpecies[b]].d_w.data());
            same = execution.capturedFields[b] == field;
        }
        if (!same) {
            if (execution.graph) requireCuda(cudaGraphExecDestroy(execution.graph));
            execution.graph = nullptr;
            const auto start = std::chrono::steady_clock::now();
            auto status = cudaStreamBeginCapture(plan.stream, cudaStreamCaptureModeThreadLocal);
            cudaGraph_t captured = nullptr;
            if (status == cudaSuccess) {
                try {
                    enqueueStreamSweeps();
                } catch (const std::runtime_error&) {
                    status = cudaErrorStreamCaptureInvalidated;
                }
                const auto ended = cudaStreamEndCapture(plan.stream, &captured);
                if (status == cudaSuccess) status = ended;
            }
            if (status == cudaSuccess) {
                status = cudaGraphInstantiate(&execution.graph, captured, 0);
            }
            if (captured) requireCuda(cudaGraphDestroy(captured));
            if (status == cudaSuccess) {
                execution.capturedForward = thrust::raw_pointer_cast(d_q.data());
                execution.capturedReverse = thrust::raw_pointer_cast(reverseHistory());
                execution.capturedFields.clear();
                for (int species : intSpecies) {
                    execution.capturedFields.push_back(thrust::raw_pointer_cast(mybox->Species[species].d_w.data()));
                }
                const auto buildUs = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count();
                std::cout << "Bond graph instantiated for molecule " << blockSpecies[0]
                          << ", build us " << buildUs << std::endl;
            } else {
                execution.captureFailed = true;
                std::cout << "Bond graph capture unavailable, using eager stream: "
                          << cudaGetErrorString(status) << std::endl;
                (void)cudaGetLastError();
            }
        }
    }
    if (execution.graph) requireCuda(cudaGraphLaunch(execution.graph, plan.stream));
    else enqueueStreamSweeps();
    execution.warmed = true;
    requireCuda(cudaEventRecord(plan.complete, plan.stream));
    if (join) joinStreamPropagators();
}

void LinearMolec::joinStreamPropagators() {
    requireCuda(cudaStreamWaitEvent(nullptr, streamBond->plan->complete));
}

void LinearMolec::enqueueBatchedSweeps(LinearMolec& other) {
    auto& bond = *batchedBond;
    const auto& plan = bond.plan;
    const auto stream = plan.stream;
    const int count = mybox->M;
    constexpr int threads = 256;
    const int grid = (count + threads - 1) / threads;
    auto* frontier = thrust::raw_pointer_cast(batchedBond->frontier.data());
    auto* spectrum = thrust::raw_pointer_cast(batchedBond->spectrum.data());
    auto* floatFrontier = thrust::raw_pointer_cast(bond.floatFrontier.data());
    auto* floatSpectrum = thrust::raw_pointer_cast(bond.floatSpectrum.data());
    auto* firstWeight = thrust::raw_pointer_cast(expW.data());
    auto* secondWeight = thrust::raw_pointer_cast(other.expW.data());
    const auto* symbol = thrust::raw_pointer_cast(d_bond_fft.data());

    struct BlockCursor {
        LinearMolec& chain;
        int block, step, stop = 0;

        void loadIfNeeded(int site, Complex* weight, int grid, int threads, int count, cudaStream_t stream) {
            if (site != stop) return;
            while (chain.N[block] == 0) block += step;
            const auto* field = thrust::raw_pointer_cast(chain.mybox->Species[chain.intSpecies[block]].d_w.data());
            bondBlockWeight<<<grid, threads, 0, stream>>>(field, weight, nullptr, count);
            stop += chain.N[block];
            block += step;
        }
    };
    const auto sweep = [&](Complex* firstTape, Complex* secondTape, bool reverse) {
        const int step = reverse ? -1 : 1;
        BlockCursor first{*this, reverse ? numBlocks - 1 : 0, step};
        BlockCursor second{other, reverse ? other.numBlocks - 1 : 0, step};
        for (size_t site = 0; site < size_t(Ntot); ++site) {
            first.loadIfNeeded(site, firstWeight, grid, threads, count, stream);
            second.loadIfNeeded(site, secondWeight, grid, threads, count, stream);
            if (bond.mixed) {
                if (site == 0) {
                    seedFloatBatch<<<grid, threads, 0, stream>>>(firstWeight, secondWeight,
                        floatFrontier, firstTape, secondTape, count);
                    continue;
                }
                plan.enqueueC2C(floatFrontier, floatSpectrum, CUFFT_FORWARD);
                scaleFloatBatch<<<grid, threads, 0, stream>>>(symbol, floatSpectrum, count);
                plan.enqueueC2C(floatSpectrum, floatFrontier, CUFFT_INVERSE);
                finishFloatBatch<<<grid, threads, 0, stream>>>(firstWeight, secondWeight, floatFrontier,
                    firstTape + site * count, secondTape + site * count, count);
                continue;
            }
            if (site == 0) {
                seedBatch<<<grid, threads, 0, stream>>>(firstWeight, secondWeight,
                    frontier, firstTape, secondTape, count);
                continue;
            }
            plan.enqueueZ2Z(frontier, spectrum, CUFFT_FORWARD);
            scaleBatch<<<grid, threads, 0, stream>>>(symbol, spectrum, count);
            plan.enqueueZ2Z(spectrum, frontier, CUFFT_INVERSE);
            finishBatch<<<grid, threads, 0, stream>>>(firstWeight, secondWeight, frontier,
                firstTape + site * count, secondTape + site * count, count);
        }
    };
    sweep(thrust::raw_pointer_cast(d_q.data()), thrust::raw_pointer_cast(other.d_q.data()), false);
    if (!isSymmetric) {
        sweep(thrust::raw_pointer_cast(d_qdag.data()), thrust::raw_pointer_cast(other.d_qdag.data()), true);
    } else {
        for (auto* molecule : {this, &other}) {
            if (!molecule->shareReverse) {
                requireCuda(cudaMemcpyAsync(thrust::raw_pointer_cast(molecule->d_qdag.data()),
                    thrust::raw_pointer_cast(molecule->d_q.data()), molecule->d_q.size() * sizeof(Complex),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }
    }
    requireCuda(cudaPeekAtLastError());
}

bool LinearMolec::calcPairedDensity(LinearMolec& other, bool batch, bool streams) {
    if (mybox != other.mybox || doSmear || other.doSmear || !directBond || !other.directBond
        || !foldBondScale || !other.foldBondScale || Ntot <= 0 || other.Ntot <= 0) return false;
    const int dim = mybox->returnDimension();
    if (dim != 2 && dim != 3) return false;

    if (batch && Ntot == other.Ntot && isSymmetric == other.isSymmetric) {
        const char* mode = std::getenv("SCFT_MATILDA_FP32_BOND");
        const bool mixed = !mybox->fullPrecisionBonds && mode
            && (std::string_view(mode) == "1" || std::string_view(mode) == "polish");
        if (batchedBond && batchedBond->mixed != mixed) batchedBond.reset();
        if (!batchedBond) {
            batchedBond = std::make_unique<BatchedBondExecution>(*mybox, mixed);
            std::cout << (mixed ? "C2C FP32" : "Z2Z FP64") << " batch-two bond plan, "
                      << 2 * size_t(mybox->M) * (mixed ? sizeof(cufftComplex) : sizeof(Complex)) * 2
                      << " frontier and spectrum bytes" << std::endl;
        }
        const auto& plan = batchedBond->plan;
        requireCuda(cudaEventRecord(plan.ready, nullptr));
        requireCuda(cudaStreamWaitEvent(plan.stream, plan.ready));
        auto& bond = *batchedBond;
        if (streamBond && streamBond->capture && bond.warmed && !bond.captureFailed) {
            bond.inputs.clear();
            for (auto* molecule : {this, &other}) {
                bond.inputs.push_back(thrust::raw_pointer_cast(molecule->d_q.data()));
                bond.inputs.push_back(thrust::raw_pointer_cast(molecule->reverseHistory()));
                for (int species : molecule->intSpecies) {
                    bond.inputs.push_back(thrust::raw_pointer_cast(mybox->Species[species].d_w.data()));
                }
            }
            if (!bond.graph || bond.inputs != bond.capturedInputs) {
                if (bond.graph) requireCuda(cudaGraphExecDestroy(bond.graph));
                bond.graph = nullptr;
                const auto start = std::chrono::steady_clock::now();
                cudaGraph_t captured = nullptr;
                auto status = cudaStreamBeginCapture(plan.stream, cudaStreamCaptureModeThreadLocal);
                if (status == cudaSuccess) {
                    try {
                        enqueueBatchedSweeps(other);
                    } catch (const std::runtime_error&) {
                        status = cudaErrorStreamCaptureInvalidated;
                    }
                    const auto ended = cudaStreamEndCapture(plan.stream, &captured);
                    if (status == cudaSuccess) status = ended;
                }
                if (status == cudaSuccess) status = cudaGraphInstantiate(&bond.graph, captured, 0);
                if (captured) requireCuda(cudaGraphDestroy(captured));
                if (status == cudaSuccess) {
                    bond.capturedInputs = bond.inputs;
                    const auto us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start).count();
                    std::cout << "Batch-two graph instantiated, build us " << us << std::endl;
                } else {
                    bond.captureFailed = true;
                    std::cout << "Batch graph capture unavailable, using eager stream: "
                              << cudaGetErrorString(status) << std::endl;
                    (void)cudaGetLastError();
                }
            }
        }
        if (bond.graph) requireCuda(cudaGraphLaunch(bond.graph, plan.stream));
        else enqueueBatchedSweeps(other);
        bond.warmed = true;
        requireCuda(cudaEventRecord(plan.complete, plan.stream));
        requireCuda(cudaStreamWaitEvent(nullptr, plan.complete));
    } else if (streams) {
        if (!streamBond) streamBond = std::make_unique<StreamBondExecution>(false);
        if (!other.streamBond) other.streamBond = std::make_unique<StreamBondExecution>(false);
        submitStreamPropagators(false, true);
        other.submitStreamPropagators(false, true);
        joinStreamPropagators();
        other.joinStreamPropagators();
    } else {
        return false;
    }
    reducePartition();
    contractDensity();
    other.reducePartition();
    other.contractDensity();
    return true;
}

void LinearMolec::calcPropagators() {
    if (streamBond) {
        submitStreamPropagators(true, false);
    } else {
    if (directBond && a.empty()) {
        a.resize(mybox->M);
        hf.resize(mybox->M);
    }
    

    //////////////////////////////////
    // Calculate forward propagator //
    //////////////////////////////////
    int ind=0;

    for ( int b=0 ; b<numBlocks ; b++ ) {
        if ( doSmear ) {
            mybox->convolveTComplexDouble(mybox->Species[intSpecies[b]].d_w, 
                W, d_smearFunc);
        }
        else {
            W = mybox->Species[intSpecies[b]].d_w;
        }
    
        // expW = exp(-W)
        thrust::transform(W.begin(), W.end(), expW.begin(), NegExponential()); 



        for ( int s=0; s<N[b] ; s++ ) {
            
            // Initial condition
            if ( ind==0 ) {
                // qdag = exp(W)
                thrust::copy(expW.begin(), expW.end(), d_q.begin());
                ind += 1;              
            }

            // Rest of the chain
            else {

                const auto next = d_q.data() + ind * mybox->M;
                propagateBondStep(next - mybox->M, next);

                ind += 1;
            } // ind != 0


        } // s=0:N[b]
    }// b=0:numBlocks


    if ( isSymmetric ) {
        if (!shareReverse) d_qdag = d_q;
    }

    //////////////////////////////
    // Complimentary propagator //
    //////////////////////////////
    else {
        ind=0;

        for ( int b=0 ; b<numBlocks ; b++ ) {
            if ( doSmear ) {
                mybox->convolveTComplexDouble(mybox->Species[intSpecies[numBlocks-b-1]].d_w, 
                    W, d_smearFunc);
            }
            else {
                W = mybox->Species[intSpecies[numBlocks-b-1]].d_w;
            }            
            
            thrust::transform(W.begin(), W.end(), expW.begin(), NegExponential()); 


            for ( int s=0; s<N[numBlocks-b-1] ; s++ ) {
                
                // Initial condition
                if ( ind==0 ) {
                    // qdag = exp(W)
                    thrust::copy(expW.begin(), expW.end(), d_qdag.begin());
                    ind += 1;
                }

                // Rest of the chain
                else {
                    
                    const auto next = d_qdag.data() + ind * mybox->M;
                    propagateBondStep(next - mybox->M, next);

                    ind += 1;
                }

            } // s=0:N[numBlocks-b-1]
        }// b=0:numBlocks
    }// Calculating complimentary propagator


    }

    reducePartition();
}

void LinearMolec::reducePartition() {
    this->Q = thrust::reduce(d_q.begin()+(Ntot-1)*mybox->M, d_q.begin()+Ntot*mybox->M, thrust::complex<double>(0.0), 
                        thrust::plus<thrust::complex<double>>()) * mybox->gvol / mybox->V;


}

struct Exponential {
    __host__ __device__
        thrust::complex<double> operator()(const thrust::complex<double> &z) {
            return exp(z);
    }
};

namespace {
struct ContourDensity {
    const thrust::complex<double>* forward;
    const thrust::complex<double>* reverse;
    const thrust::complex<double>* field;
    thrust::complex<double> factor;
    int sites;
    int gridSize;

    __host__ __device__ thrust::complex<double> operator()(int voxel) const {
        thrust::complex<double> density(0.0);
        for (int site = 0; site < sites; ++site) {
            density += forward[site * gridSize + voxel] * reverse[voxel - site * gridSize];
        }
        return density * (factor * exp(field[voxel]));
    }
};
}

void LinearMolec::calcDensity() {

    // First, calculate the propagators
    calcPropagators();
    contractDensity();
}

void LinearMolec::contractDensity() {
    int M = mybox->M;
    thrust::complex<double> factor;
    if ( nmolecs > 0.0 ) {
        factor = nmolecs / Q / mybox->V;
    }
    else if (activity > 0.0 ) {
        factor = activity;
    }


    const auto reverse = reverseHistory();
    if (!fuseDensity) thrust::fill(d_cDensity.begin(), d_cDensity.end(), 0.0);
    int ind = 0;
    for ( int b=0 ; b<numBlocks; b++ ) {
        if (fuseDensity) {
            const auto voxel = thrust::make_counting_iterator(0);
            const auto& field = mybox->Species[intSpecies[b]].d_w;
            thrust::transform(thrust::device, voxel, voxel + M, d_cDensity.begin() + b * M,
                ContourDensity{thrust::raw_pointer_cast(d_q.data() + ind * M),
                               thrust::raw_pointer_cast(reverse + (Ntot - 1 - ind) * M),
                               thrust::raw_pointer_cast(field.data()), factor, N[b], M});
            ind += N[b];
            continue;
        }

        // Accumulate density of all monomers in this block
        for ( int s=0 ; s<N[b]; s++ ) {

            // q_qdag[:] = q[ind][:] * qdag[Ntot-1-ind][:]
            thrust::transform(d_q.begin()+ind*M, d_q.begin()+(ind+1)*M, 
                                reverse+(Ntot-1-ind)*M, q_qdag.begin(),
                                thrust::multiplies<thrust::complex<double>>());                                 

            // cDensity[:] += q_qdag[:]
            thrust::transform(q_qdag.begin(), q_qdag.end(), d_cDensity.begin()+b*M, d_cDensity.begin()+b*M,
                                thrust::plus<thrust::complex<double>>());

            ind += 1;
        }

        W = mybox->Species[intSpecies[b]].d_w;
        // expW = exp(W)
        thrust::transform(W.begin(), W.end(), expW.begin(), Exponential()); 
        
        // factor = n / Q / V
        thrust::fill(temp.begin(), temp.end(), factor);

        // temp <- factor * expW
        thrust::transform(temp.begin(), temp.end(), expW.begin(), temp.begin(), 
                            thrust::multiplies<thrust::complex<double>>()); 

        // Multiply this blocks density by expW*factor
        thrust::transform(d_cDensity.begin()+b*M, d_cDensity.begin()+(b+1)*M, temp.begin(), 
                            d_cDensity.begin()+b*M, thrust::multiplies<thrust::complex<double>>()); 

    }

    // Define total density as juts center density for now. needs to be convolved
    // with shape functions once those are implemented.
    if ( doSmear ) {
        // smear the density field
        mybox->convolveTComplexDouble(d_cDensity, d_density, d_smearFunc);
    }

    // Not using smearing
    else {
        d_density = d_cDensity;
    }
    
    if ( activity > 0.0 ) {
        nSites = activity * mybox->V * Q.real() * Ntot;

        // thrust::complex<double> ntemp;
        // ntemp = thrust::reduce(d_cDensity.begin(), d_cDensity.end(), thrust::complex<double>(0.0),
        //                         thrust::plus<thrust::complex<double>>()) * mybox->gvol;
        
        // std::cout << "nmolecs: " << nmolecs << " ntmp: " << ntemp << std::endl;
            // this->Q = thrust::reduce(d_q.begin()+(Ntot-1)*mybox->M, d_q.begin()+Ntot*mybox->M, thrust::complex<double>(0.0), 
            //             thrust::plus<thrust::complex<double>>()) * mybox->gvol / mybox->V;
    }

    // Finally, accumulate density onto the relevant species field
    for ( int b=0 ; b<numBlocks ; b++ ) {
        int is = intSpecies[b];
        thrust::transform(d_density.begin()+b*M, d_density.begin()+(b+1)*M, mybox->Species[is].d_density.begin(), 
                            mybox->Species[is].d_density.begin(), thrust::plus<thrust::complex<double>>());
    }

}


// Computes the contribution to the Hamiltonian 
// for this molecule style. 
std::complex<double> LinearMolec::calcHTerm() {
    std::complex<double> ht = 0.0;

    if ( nmolecs > 0.0 ) {
        ht = (std::complex<double>)( -nmolecs * log( Q ) );
    }
    else if ( activity > 0.0 ) {
        ht = (std::complex<double>)( -activity*mybox->V*Q );
    }
    else {
        die("Invalid conditions to calcHTerm()");
    }

    return ht;
}

void LinearMolec::modifyMolecule(std::istringstream& iss) {

    std::string property, how2Mod;
    iss >> property;
    
    if ( property == "activity" ) {
        iss >> how2Mod;

        double change;
        iss >> change;

        std::cout << "activity changed from " << activity ;

        modifyMolecParam(activity, how2Mod, change);
        
        // if (how2Mod == "scale" ) {
        //     activity *= change;
        // }

        // else if ( how2Mod == "delta" ) {
        //     activity += change;
        // }

        // else if ( how2Mod == "value" ) {
        //     activity = change;
        // }

        std::cout << " to " << activity << std::endl;
    }

    // Changes here require re-calculating nmolecs
    else if ( property == "phi" ) {
        die("changes in FTS:molec:phi not yet supported");
    }

    // Changes here require re-calculating nmolecs
    else if ( property == "nmolecs" ) {
        die("changes in FTS:molec:nmolecs not yet supported");
    }

    else {
        std::string lastWords = property + " is not a valid property to change in FTS:molecule";
        die(lastWords.c_str());
    }
}


// Once smearing is implemented, smear functions need to 
// be included in the linear coefficients
void LinearMolec::computeLinearTerms() {
    nmolecs = mybox->C * phi * mybox->Vfree * mybox->Nr / (double(Ntot));
    std::cout << " nSites = " << nmolecs * double(Ntot) << " nmolecs = " << nmolecs << std::endl;

    double alpha = double(Ntot) / double(mybox->Nr);

    // n*Ntot**2 / (Nr * V)
    double prefactor = nmolecs * alpha * double(Ntot) / mybox->V;
    thrust::host_vector<thrust::complex<double>> pref(mybox->M, prefactor);
    
    // Declare htmp storage variable
    thrust::host_vector<thrust::complex<double>> htmp(mybox->M);
    thrust::host_vector<thrust::complex<double>> gd(mybox->M);

    // Get the total Debye function for Helfand potential
    mybox->computeHomopolyDebye(gd, alpha);


    // htmp = prefactor * gd
    thrust::transform(pref.begin(), pref.end(), gd.begin(), htmp.begin(),
        thrust::multiplies<thrust::complex<double>>());
    

    // Find the Helfand potential and add this molecules contribution
    for ( int i=0 ; i<mybox->Potentials.size(); i++ ) {
        std::string pstyle = mybox->Potentials[i]->printStyle();
        if ( pstyle == "Helfand" || pstyle == "Edwards" ) {
            thrust::host_vector<thrust::complex<double>> Atmp(mybox->M);
            Atmp = mybox->Potentials[i]->d_Akpl;

            thrust::transform(htmp.begin(), htmp.end(), Atmp.begin(),
                Atmp.begin(), thrust::plus<thrust::complex<double>>());

            mybox->Potentials[i]->d_Akpl = Atmp;
        }
    }


    // Add to Flory potentials

    // If homopolymer, just add htmp to d_Akpl and be done
    if ( numBlocks == 1 ) {
        for ( int i=0 ; i<mybox->Potentials.size(); i++ ) {
            if ( mybox->Potentials[i]->printStyle() == "Flory" ) {
                thrust::host_vector<thrust::complex<double>> Atmp(mybox->M);
                Atmp = mybox->Potentials[i]->d_Akpl;

                thrust::transform(htmp.begin(), htmp.end(), Atmp.begin(), Atmp.begin(),
                                    thrust::plus<thrust::complex<double>>());

                mybox->Potentials[i]->d_Akpl = Atmp;
            }
        }
    }// numBlocks == 1 (homopolymer)


    // Block polymers
    else {
        thrust::host_vector<thrust::complex<double>> gaa(mybox->M);
        thrust::host_vector<thrust::complex<double>> gac(mybox->M);
        thrust::host_vector<thrust::complex<double>> gcc(mybox->M);
        thrust::host_vector<thrust::complex<double>> gMT(mybox->M);

        for ( int i=0; i<mybox->Potentials.size() ; i++ ) {
            if ( mybox->Potentials[i]->printStyle() == "Flory" ) {
                for ( int j=0 ; j<numBlocks-1 ; j++ ) {
                    // does this potential act on block j?
                    if ( blockSpecies[j] != mybox->Potentials[i]->actsOn[0] &&
                         blockSpecies[j] != mybox->Potentials[i]->actsOn[1] ) {
                            continue;
                    }

                    double fA = double(N[j]) / double(Ntot);
                    
                    mybox->computeIntRABlockDebye(gaa, fA, alpha);
                    gMT = gaa;

                    int Nbetween = 0;
                    for ( int k=j+1 ; k<numBlocks ; k++ ) {
                        // does this potential act on block k?
                        if ( blockSpecies[k] != mybox->Potentials[i]->actsOn[0] &&
                             blockSpecies[k] != mybox->Potentials[i]->actsOn[1] ) {
                            continue;
                        }

                        std::cout << "potential " << i << "accumulating species " << blockSpecies[j] << " and " << blockSpecies[k] << std::endl;

                        double fC = double(N[k]) / double(Ntot);
                        double fB = double(Nbetween) / double(Ntot);

                        mybox->computeIntRABlockDebye(gcc, fC, alpha);
                        thrust::transform(gMT.begin(), gMT.end(), gcc.begin(), gMT.begin(), 
                            thrust::plus<thrust::complex<double>>());

                        // interblock Debye includes 2.0 prefactor
                        mybox->computeIntERBlockDebye(gac, fA, fB, fC, alpha );

                        // gMT = gaa + gcc + 2.0 * gac
                        thrust::transform(gMT.begin(), gMT.end(), gac.begin(), gMT.begin(), 
                            thrust::plus<thrust::complex<double>>());

                        // htmp = prefactor * gMT
                        thrust::transform(pref.begin(), pref.end(), gMT.begin(), htmp.begin(),
                            thrust::multiplies<thrust::complex<double>>());

                        // copy prefactor to host   
                        thrust::host_vector<thrust::complex<double>> Atmp(mybox->M);
                        Atmp = mybox->Potentials[i]->d_Akpl;

                        // accumulate htmp
                        thrust::transform(htmp.begin(), htmp.end(), Atmp.begin(), Atmp.begin(),
                                            thrust::plus<thrust::complex<double>>());

                        // prefactor back to device
                        mybox->Potentials[i]->d_Akpl = Atmp;                            

                        // accumulate Nbetween
                        Nbetween += N[k];
                    }// k=j+1:numBlocks
                }// j=0:numBlocks
            }// if potentialStyle == "Flory"
        }

    }// numBlocks > 1

}

void LinearMolec::recomputeNmolecs() {
    if ( activity > 0.0 ) {
        nmolecs = mybox->C * phi * mybox->Vfree * mybox->Nr / (double(Ntot));
        std::cout << " recomputed nSites = " << nmolecs * double(Ntot) << " nmolecs = " << nmolecs << std::endl;
    }
}
