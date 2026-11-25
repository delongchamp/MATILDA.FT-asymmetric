enum class ResearchMethod { Direct, OtherPressure, MaskedPressure, BB1, Regional, GMRES };

ResearchMethod requestedResearchMethod() {
    const char* option = std::getenv("SCFT_MATILDA_RESEARCH_METHOD");
    if (!option || std::string_view(option) == "direct") return ResearchMethod::Direct;
    const std::string_view name(option);
    if (name == "pressure_other") return ResearchMethod::OtherPressure;
    if (name == "pressure_mask") return ResearchMethod::MaskedPressure;
    if (name == "bb1") return ResearchMethod::BB1;
    if (name == "regional") return ResearchMethod::Regional;
    if (name == "gmres3") return ResearchMethod::GMRES;
    throw std::runtime_error("Unsupported SCFT research method " + std::string(name));
}

__global__ void pressureAccessibility(const Complex* mask, double scale, double* weights, int sites) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) weights[i] = sqrt(fmin(1.0, fmax(0.0, 1.0 - scale * mask[i].real())));
}

__global__ void pressureWindow(Complex* dest, const Complex* source, const double* weight,
                               Complex mean, int sites) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) dest[i] = weight[i] * (source[i] - mean);
}

__global__ void applyPressureCorrection(Complex* pressure, const Complex* delta, Complex mean, int sites) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) pressure[i] += delta[i] - mean;
}

__global__ void applyScaledPlain(MixFields fields, int sites, const Complex* x, const Complex* r,
                                 double gain) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    for (int component = 0; component < 3; ++component) {
        const int offset = component * sites + i;
        fields.field[component][i] = x[offset] + gain * r[offset];
    }
}

struct SecantProducts {
    double ss = 0, yy = 0, sy = 0;
};

struct AddSecant {
    __host__ __device__ SecantProducts operator()(SecantProducts a, SecantProducts b) const {
        return {a.ss + b.ss, a.yy + b.yy, a.sy + b.sy};
    }
};

struct SecantAt {
    const Complex *current, *previous, *currentFit, *previousFit;
    int sites;

    __host__ __device__ SecantProducts operator()(int i) const {
        const auto s = current[i] - previous[i];
        const auto y = previousFit[i] - currentFit[i];
        const double weight = i >= 2 * sites ? 4.0 : 1.0;
        return {weight * (s.real() * s.real() + s.imag() * s.imag()),
                weight * (y.real() * y.real() + y.imag() * y.imag()),
                weight * (s.real() * y.real() + s.imag() * y.imag())};
    }
};

__global__ void offsetPressureSymbol(double* weights, double value, int sites) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) weights[i] -= value;
}
