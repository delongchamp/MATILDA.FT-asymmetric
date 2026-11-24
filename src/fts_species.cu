// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_species.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

void die(const char*);


FTS_Species::FTS_Species(std::istringstream &iss, FTS_Box* p_box) : box(p_box) {
    this->input_command = iss.str();
    
    iss >> this->fts_species; 

    int M = box->M;

    // Resize the arrays to be of the grid dimension size
    this->density.resize(M);
    this->d_density.resize(M);
    this->d_Ak.resize(M);
    this->d_w.resize(M);

    return;
}

FTS_Species::~FTS_Species() {
    return;
}
FTS_Species::FTS_Species() {
    return;
}


// Loops over all of the molecules and zeroes the species densities
void FTS_Species::zeroDensity() {
    thrust::fill(d_density.begin(), d_density.end(), 0.0);
}


// Allows one step to directly add I * wpl field to species field ws
struct plusITimes {
    __host__ __device__ 
        thrust::complex<double> operator()(const thrust::complex<double> &ws, const thrust::complex<double> &wpl) {
            thrust::complex<double> I = thrust::complex<double>(0.0,1.0);
            return ws + I * wpl;
        }
};



namespace {
using Complex = thrust::complex<double>;

struct BuildSpeciesField {
    enum Operation { Add, Subtract, AddImaginary };
    struct Term {
        const Complex* field;
        Operation operation;
    };
    Term terms[8];
    int count = 0;
    Complex scale;

    bool append(const thrust::device_vector<Complex>& field, Operation operation) {
        if (count == 8) return false;
        terms[count++] = {thrust::raw_pointer_cast(field.data()), operation};
        return true;
    }

    __host__ __device__ Complex operator()(int i) const {
        Complex value(0.0, 0.0);
        for (int term = 0; term < count; ++term) {
            const auto entry = terms[term];
            const Complex operand = entry.field[i];
            switch (entry.operation) {
                case Add: value += operand; break;
                case Subtract: value -= operand; break;
                case AddImaginary: value += Complex(0.0, 1.0) * operand; break;
            }
        }
        return value * scale;
    }
};

struct ScaleSpeciesField {
    Complex scale;
    __host__ __device__ Complex operator()(const Complex& value) const { return value * scale; }
};
}

// Loops over potentials and constructs the fields for each species
void FTS_Species::buildPotentialField() {
    if (box->fieldAssembly == 3) {
        BuildSpeciesField build{};
        build.scale = 1.0 / box->Nr;
        bool supported = true;
        for (auto* potential : box->Potentials) {
            const auto style = potential->printStyle();
            if (style == "Incompress") {
                supported = build.append(potential->d_wpl, BuildSpeciesField::AddImaginary);
            } else if (style == "Flory" && fts_species == potential->actsOn[0]) {
                supported = build.append(potential->d_wmi, BuildSpeciesField::Subtract)
                    && build.append(potential->d_wpl, BuildSpeciesField::AddImaginary);
            } else if (style == "Flory" && fts_species == potential->actsOn[1]) {
                supported = build.append(potential->d_wmi, BuildSpeciesField::Add)
                    && build.append(potential->d_wpl, BuildSpeciesField::AddImaginary);
            } else if (style == "Particle" && fts_species == potential->actsOn[0]) {
                supported = build.append(potential->d_wpl, BuildSpeciesField::Add);
            } else if (style != "Flory" && style != "Particle") {
                supported = false;
            }
            if (!supported) break;
        }
        if (supported) {
            const auto first = thrust::make_counting_iterator(0);
            thrust::transform(thrust::device, first, first + box->M, d_w.begin(), build);
            return;
        }
    }

    // Zero the fields
    thrust::fill(d_w.begin(), d_w.end(), thrust::complex<double>(0.0,0.0));

    for ( int i=0 ; i<box->Potentials.size() ; i++ ) {

        if ( box->Potentials[i]->printStyle() == "Helfand" ) {

            // d_w += I * wpl from Helfand
            thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                d_w.begin(), plusITimes());

        }

        if ( box->Potentials[i]->printStyle() == "Incompress" ) {

            // d_w += I * wpl from Helfand
            thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                d_w.begin(), plusITimes());

        }

        if ( box->Potentials[i]->printStyle() == "Edwards" ) {

            // d_w += I * wpl from Edwards
            thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                d_w.begin(), plusITimes());

        }

        else if ( box->Potentials[i]->printStyle() == "Flory" ) {
            if ( fts_species == box->Potentials[i]->actsOn[0] ) {
                
                // wIJ- field is *subtracted* from the first listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wmi.begin(),
                    d_w.begin(), thrust::minus<thrust::complex<double>>());

                // d_w += I * wpl for both listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(),
                     d_w.begin(), plusITimes());
            }

            if ( fts_species == box->Potentials[i]->actsOn[1] ) {
                
                // wIJ- field is *added* to the second listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wmi.begin(), 
                    d_w.begin(), thrust::plus<thrust::complex<double>>());

                // d_w += I * wpl for both listed species
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(), 
                    d_w.begin(), plusITimes());                    
            }
        }

        else if (box->Potentials[i]->printStyle() == "Particle" ) {
            if (fts_species == box->Potentials[i]->actsOn[0] ) {
                // AN: add C* chi* rho_exp_Nr to field...this is wpl from particle pot.
                thrust::transform(d_w.begin(), d_w.end(), box->Potentials[i]->d_wpl.begin(), d_w.begin(), thrust::plus<thrust::complex<double>>()); 
            }
        }
        
    }// i=0:potentials.size()

    // Normalize the field by Nr
    if (box->fieldAssembly >= 1) {
        thrust::transform(d_w.begin(), d_w.end(), d_w.begin(), ScaleSpeciesField{1.0 / box->Nr});
    } else {
        thrust::device_vector<thrust::complex<double>> invNr(box->M);
        thrust::fill(invNr.begin(), invNr.end(), 1.0/double(box->Nr));
        thrust::transform(d_w.begin(), d_w.end(), invNr.begin(), d_w.begin(),
                          thrust::multiplies<thrust::complex<double>>());
    }


    // thrust::host_vector<thrust::complex<double>> htmp(box->M);
    // htmp = d_w;
    // if ( fts_species == "A" )
    //     box->writeTComplexGridData("from_speciesA.dat", htmp);
    // else {
    //     box->writeTComplexGridData("from_speciesB.dat", htmp);
    // //    die("done");
    // }
}


// Writes the density field to a text file
void FTS_Species::writeDensity(const int id) {
    char nm[40];

    sprintf(nm, "rhoSpecies%d.dat", id);
    
    // Transfer data to host
    density = d_density;
    box->writeTComplexGridData(std::string(nm), density);
}

// Writes the total field acting on a particular species
void FTS_Species::writeSpeciesFields(const int id) {
    char nm[40];
    sprintf(nm, "fieldSpecies%d.dat", id);

    thrust::host_vector<thrust::complex<double>> htmp = d_w;
    box->writeTComplexGridData(std::string(nm), htmp);

}