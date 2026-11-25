__global__ void regionalGram(const Complex* residual, int sites, MixWindow window,
                             const double* wall, bool near, double normalizer, double* parts) {
    __shared__ double shared[mixThreads];
    const int width = 3 * sites;
    const auto [row, column] = mixGramEntry<GramLayout::Ordered>(blockIdx.x, window.history);
    double total = 0;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < width; i += blockDim.x * gridDim.y) {
        const auto x = residualColumn(residual, width, i, window, row);
        const auto y = residualColumn(residual, width, i, window, column);
        const double local = wall[i % sites];
        const double weight = (i >= 2 * sites ? 4.0 : 1.0) * (near ? local : 1 - local);
        total += weight * (x.real() * y.real() + x.imag() * y.imag());
    }
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int half = mixThreads / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) shared[threadIdx.x] += shared[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0) parts[blockIdx.x * gridDim.y + blockIdx.y] = shared[0] / normalizer;
}

__global__ void applyRegionalMix(MixFields fields, int sites, const Complex* x, const Complex* r,
                                 MixWindow global, MixWindow near, MixWindow far, const double* wall,
                                 Complex* pressureChange) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    for (int component = 0; component < 3; ++component) {
        const auto index = [&](int j) { return (3 * global.slot[j] + component) * sites + i; };
        const int current = index(global.history);
        Complex next = x[current] + r[current], correction{};
        for (int h = 0; h < global.history; ++h) {
            const int a = index(h), b = index(h + 1);
            const auto difference = (x[b] - x[a]) + (r[b] - r[a]);
            next -= global.gamma[h] * difference;
            const double local = wall[i] * (near.gamma[h] - global.gamma[h]) +
                (1 - wall[i]) * (far.gamma[h] - global.gamma[h]);
            correction -= 0.5 * local * difference;
        }
        fields.field[component][i] = next + correction;
        if (component == 0) pressureChange[i] = correction;
    }
}

__global__ void centerRegionalPressure(Complex* pressure, int sites, Complex mean) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sites) pressure[i] -= mean;
}
