// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#ifndef _FTS_MOLEC_LINEAR
#define _FTS_MOLEC_LINEAR
#include "fts_molecule.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

class FTS_Box;

class LinearMolec : public FTS_Molec {
    private:
        const bool directBond;
        const bool foldBondScale;

        void propagateBondStep(thrust::device_ptr<const thrust::complex<double>> previous,
                               thrust::device_ptr<thrust::complex<double>> next);

    public:
        int numBlocks;
        bool isSymmetric = true;

        thrust::device_vector<int> d_N;                     // [numBlocks] length of each block
        thrust::host_vector<int> N;                         // [numBlocks] length of each block
        int Ntot;                                           // total chain length, sum of N
        
        thrust::host_vector<int> intSpecies;                 // [numBlocks] type for each block   
        thrust::device_vector<int> d_intSpecies;               // [numBlocks] type for each block   
        thrust::host_vector<std::string> blockSpecies;      // [numBlocks] type for each block as a string
        

        thrust::device_vector<thrust::complex<double>> d_q;    // [Ntot*M] Chain propagator
        thrust::device_vector<thrust::complex<double>> d_qdag; // [Ntot*M] inverse propagator

        // [M] Fourier bond symbol, including 1/M in folded mode
        thrust::device_vector<thrust::complex<double>> d_bond_fft;    

        // storage variables
        thrust::device_vector<thrust::complex<double>> W;
        thrust::device_vector<thrust::complex<double>> qf;
        thrust::device_vector<thrust::complex<double>> hf;
        thrust::device_vector<thrust::complex<double>> a;
        thrust::device_vector<thrust::complex<double>> expW;
        thrust::device_vector<thrust::complex<double>> q_qdag;
        thrust::device_vector<thrust::complex<double>> temp;

        
        ~LinearMolec();
        LinearMolec(std::istringstream& iss, FTS_Box*);

        void calcPropagators();
        void calcDensity() override;
        std::complex<double> calcHTerm() override;
        void computeLinearTerms() override;
        void modifyMolecule(std::istringstream&) override;
        void recomputeNmolecs(void) override;
};

#endif
