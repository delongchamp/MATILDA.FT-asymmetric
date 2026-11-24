// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "include_libs.h"
#include "fts_potential_incompress.h"
#include "FTS_Box.h"
#include "fts_species.h"
#include <cmath>
#include <cstdlib>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
//#include "globals.h"

void die(const char*);
double ran2();


PotentialIncompress::PotentialIncompress(std::istringstream& iss, FTS_Box* p_box) : FTS_Potential(iss, p_box) {


    // Set stringstream to be ready to read kappa
    iss.seekg(0);
    std::string s1;
    iss >> s1;
    iss >> s1;

    potentialStyle = "Incompress";

    iss >> delt;

    double ivalue = 0.0;
    wpl.resize(mybox->M,ivalue);
    d_wpl.resize(mybox->M, ivalue);
    d_Akpl.resize(mybox->M, ivalue);
    wplAlloc_flag = 1;

    if (mybox->fieldAssembly != 3) d_rho_total.resize(mybox->M, ivalue);
    d_dHdw.resize(mybox->M, ivalue);

    // Set default update scheme
    updateScheme = "EM";

    while (iss.tellg() != -1 ) {
        iss >> s1;
        if ( s1 == "initialize" ) {
            initializeField(iss, wpl);
        }

        else if ( s1 == "updateScheme" ) {
            iss >> updateScheme;
        }

        else if ( s1 == "modify" ) {
            iss >> s1;
            if ( s1 != "zeromean" && s1 != "zeroMean" ) { die("Invalid modify option on Helfand potential"); }
            zeroMean = true;
        }
    }// optional arguments

    d_wpl = wpl;

    if (const char* setting = std::getenv("SCFT_MATILDA_PRESSURE_STEP")) {
        char* end = nullptr;
        const double multiplier = std::strtod(setting, &end);
        if (end == setting || *end || !std::isfinite(multiplier) || multiplier <= 0.0) {
            die("SCFT_MATILDA_PRESSURE_STEP must be a positive finite number");
        }
        if (multiplier != 1.0 && updateScheme == "EMPC" && mybox->returnFTSstyle() == "scft") {
            delt *= multiplier;
            std::cout << "EMPC pressure multiplier " << multiplier << ", step " << delt << std::endl;
        }
    }

    if ( updateScheme == "EMPC" ) {
        d_dHdwplo.resize(mybox->M, ivalue);        
        d_wplo.resize(mybox->M, ivalue);

        // ensure PC flag set to TRUE.
        mybox->PCflag = 1;
    }

}// PotentialHelfand constructor


// Updates potential fields using the chosen scheme
// If two-part predictor/corrector scheme is to be used, then 
// this step is the predictor step
namespace {
using Complex = thrust::complex<double>;

struct PressureForce {
    const Complex* species[3];
    int count;
    double Nr;
    double C;

    __host__ __device__ Complex operator()(int i) const {
        Complex total(0.0, 0.0);
        for (int j = 0; j < count; ++j) total += species[j][i];
        return Complex(-total.imag() / Nr, total.real() / Nr - C);
    }
};
}

void PotentialIncompress::computeForce() {
    const auto size = mybox->Species.size();
    if (mybox->fieldAssembly == 3 && size <= 3) {
        PressureForce force{{}, static_cast<int>(size), mybox->Nr, mybox->C};
        for (size_t j = 0; j < size; ++j) {
            force.species[j] = thrust::raw_pointer_cast(mybox->Species[j].d_density.data());
        }
        const auto first = thrust::make_counting_iterator(0);
        // The predictor snapshot and field update are ordered on this same stream.
        thrust::transform(thrust::cuda::par_nosync, first, first + mybox->M, d_dHdw.begin(), force);
        return;
    }
    if (d_rho_total.empty()) d_rho_total.resize(mybox->M);
    thrust::fill(d_rho_total.begin(), d_rho_total.end(), 0.0);
    for (const auto& species : mybox->Species) {
        thrust::transform(species.d_density.begin(), species.d_density.end(),
            d_rho_total.begin(), d_rho_total.begin(), thrust::plus<Complex>());
    }
    auto* force = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(d_dHdw.data()));
    const auto* field = reinterpret_cast<const cuDoubleComplex*>(thrust::raw_pointer_cast(d_wpl.data()));
    const auto* density = reinterpret_cast<const cuDoubleComplex*>(thrust::raw_pointer_cast(d_rho_total.data()));
    d_makeIncompressForce<<<mybox->M_Grid, mybox->M_Block>>>(force, field, density, mybox->C,
        mybox->Nr, mybox->M);
}

void PotentialIncompress::updateFields() {

    computeForce();
    auto* _d_wpl = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(d_wpl.data()));
    auto* _d_dHdw = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(d_dHdw.data()));



    if ( updateScheme == "EMPC" ) {
        storePredictorData();
    }


    // Update the fields
    if ( updateScheme == "EM" || updateScheme == "EMPC") {
        d_fts_updateEM<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdw, delt, mybox->M);
    }


    else if ( updateScheme == "1S" ) {
        // Put the force and potential into k-space
        mybox->cufftWrapperDouble(d_dHdw, d_dHdw, 1);
        mybox->cufftWrapperDouble(d_wpl, d_wpl, 1);

        // Pointer to linear coefficient
        cuDoubleComplex* _d_Ak = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Akpl.data());

        // Call updater
        d_fts_update1S<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdw, _d_Ak, delt, mybox->M);

        // Bring potential back to r-space
        mybox->cufftWrapperDouble(d_wpl, d_wpl, -1);
    }

    // Check for modifiers
    if ( zeroMean == true ) {
        thrust::complex<double> mean = thrust::reduce(d_wpl.begin(), d_wpl.end()) / double(mybox->M);
        
        // dtmp = mean
        thrust::device_vector<thrust::complex<double>> dtmp(mybox->M, mean);

        // wpl(r) = wpl(r) - mean
        thrust::transform(d_wpl.begin(), d_wpl.end(), dtmp.begin(), d_wpl.begin(), 
            thrust::minus<thrust::complex<double>>());
    }

}// updateFields




void PotentialIncompress::correctFields() {
    if ( updateScheme != "EMPC" ) {
        return;
    }

    computeForce();
    auto* _d_wpl = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(d_wpl.data()));
    auto* _d_dHdw = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(d_dHdw.data()));
    const auto* _d_dHdwplo = reinterpret_cast<const cuDoubleComplex*>(thrust::raw_pointer_cast(d_dHdwplo.data()));
    const auto* _d_wplo = reinterpret_cast<const cuDoubleComplex*>(thrust::raw_pointer_cast(d_wplo.data()));


    // Corrector step for field updates
    d_fts_updateEMPC<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_wplo, _d_dHdw, _d_dHdwplo, delt, mybox->M);
    

    
    // Check for modifiers
    if ( zeroMean == true ) {
        thrust::complex<double> mean = thrust::reduce(d_wpl.begin(), d_wpl.end()) / double(mybox->M);
        
        // dtmp = mean
        thrust::device_vector<thrust::complex<double>> dtmp(mybox->M, mean);

        // wpl(r) = wpl(r) - mean
        thrust::transform(d_wpl.begin(), d_wpl.end(), dtmp.begin(), d_wpl.begin(), 
            thrust::minus<thrust::complex<double>>());
    }    
}



// Stores initial potential and force forms for predictor/corrector updates
void PotentialIncompress::storePredictorData() {
    d_wplo = d_wpl;
    d_dHdwplo = d_dHdw;
}




// This routine is currently written to deal with dHdw in real space
// the I*rho0 term should change if it changed to k-space updating
__global__ void d_makeIncompressForce(
    cuDoubleComplex* dHdw,              // Field holding dHdw
    const cuDoubleComplex* w,           // current d_wpl
    const cuDoubleComplex* rho_total,   // current total density
    const double C,                     // Polymer concentration, based on Nr
    const double Nr,                    // Reference chain length
    const int M                         // number of grid points
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    // dHdw = C * w / kN - I * C + I * rho_total / Nr ;
    dHdw[ind].x = -rho_total[ind].y / Nr;
    dHdw[ind].y = rho_total[ind].x / Nr - C;
}


void PotentialIncompress::writeFields(int potInd ) { 
    char nm[30];
    sprintf(nm, "wpl_Incompress%d.dat", potInd);

    // Transfer field to the host;
    wpl = d_wpl;
    mybox->writeTComplexGridData(nm, wpl);
}


// Computes this potential's contribution to the effective Hamiltonian
std::complex<double> PotentialIncompress::calcHamiltonian() {
    thrust::device_vector<thrust::complex<double>> dtmp(mybox->M);
    thrust::complex<double> I(0.0,1.0);



    // -i C * int(wpl)
    thrust::complex<double> integral = thrust::reduce(d_wpl.begin(), d_wpl.end()) * mybox->gvol;

    Hterm = -I * mybox->C * integral;

    //std::cout << Hterm << std::endl;

    return Hterm;
    
}

void PotentialIncompress::initLinearCoeffs() {
    
    thrust::fill(d_Akpl.begin(), d_Akpl.end(), 0.0);
}

PotentialIncompress::~PotentialIncompress() {}