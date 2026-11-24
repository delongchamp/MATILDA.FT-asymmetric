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

    explicit StreamBondPlan(FTS_Box& box) {
        try {
            requireCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            requireCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            requireCuda(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming));
            requireFft(cufftCreate(&fft));
            if (box.returnDimension() == 2) {
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
};

struct StreamBondExecution {
    std::shared_ptr<StreamBondPlan> plan;
    const bool capture;
    bool warmed = false;
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
    W.resize(mybox->M);
    if (!directBond) qf.resize(mybox->M);
    hf.resize(mybox->M);
    a.resize(mybox->M);
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

void LinearMolec::submitStreamPropagators() {
    auto& execution = *streamBond;
    if (!execution.plan) {
        execution.plan = mybox->streamBondPlan.lock();
        if (!execution.plan) {
            execution.plan = std::make_shared<StreamBondPlan>(*mybox);
            mybox->streamBondPlan = execution.plan;
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
    requireCuda(cudaStreamWaitEvent(nullptr, plan.complete));
}

void LinearMolec::calcPropagators() {
    if (streamBond) {
        submitStreamPropagators();
    } else {
    

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

    // Calculate the partition function
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
