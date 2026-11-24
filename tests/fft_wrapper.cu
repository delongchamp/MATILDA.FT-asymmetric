// MATILDA.FT regression test, released under the GNU Public License version 2.
#include "Box.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using Complex = thrust::complex<double>;

class TestBox final : public Box {
public:
    TestBox(int nx, int ny, int nz) {
        M = nx * ny * nz;
        ftTimer = 0;
        const auto status = nz == 1
            ? cufftPlan2d(&fftplan, ny, nx, CUFFT_Z2Z)
            : cufftPlan3d(&fftplan, nz, ny, nx, CUFFT_Z2Z);
        if (status != CUFFT_SUCCESS) throw std::runtime_error("Cannot plan test FFT");
    }
    ~TestBox() override { cufftDestroy(fftplan); }
    void readInput(std::ifstream&) override {}
    void writeFields() override {}
    void writeTime() override {}
    int converged(int) override { return 0; }
    void NVT(int) override {}
    void writeData(int) override {}
    void findSpinodal(std::istringstream&) override {}
    void doTimeStep(int) override {}
    void modifyBox(std::istringstream&) override {}
};

void assertNear(Complex actual, Complex expected, const char* what) {
    if (!(thrust::abs(actual - expected) <= 1e-12)) throw std::runtime_error(what);
}

void testRoundTrip(int nx, int ny, int nz) {
    TestBox box(nx, ny, nz);
    const Complex sentinel(123, 456);
    thrust::host_vector<Complex> expected(box.M);
    Complex mean;
    for (int i = 0; i < box.M; ++i) {
        expected[i] = Complex(1.5 + std::sin(i * 0.16), -0.75 + std::cos(i * 0.31));
        mean += expected[i] / double(box.M);
    }
    const thrust::device_vector<Complex> input = expected;
    thrust::device_vector<Complex> output(box.M + 1, sentinel);
    for (int iteration = 0; iteration < 3; ++iteration) {
        box.cufftWrapperDouble(input, output, 1);
        thrust::host_vector<Complex> spectral = output;
        assertNear(spectral[0], mean, "wrong normalized zero mode");
        box.cufftWrapperDouble(output, output, -1);
        thrust::host_vector<Complex> actual = output;
        assertNear(actual.back(), sentinel, "wrote past the FFT grid");
        for (int i = 0; i < box.M; ++i) {
            assertNear(actual[i], expected[i], "out-of-place forward / in-place inverse");
        }
        box.cufftWrapperDouble(output, output, 1);
        thrust::host_vector<Complex> aliased = output;
        for (int i = 0; i < box.M; ++i) {
            assertNear(aliased[i], spectral[i], "aliased and out-of-place forward disagree");
        }
    }
    thrust::host_vector<Complex> original = input;
    for (int i = 0; i < box.M; ++i) {
        assertNear(original[i], expected[i], "out-of-place transform changed its input");
    }
}
}

// Needed by the archive's die.cu when linking the test without main.cu.
std::string giveQuote() { return {}; }
long idum = 0;

int main() {
    try {
        testRoundTrip(35, 35, 1);
        testRoundTrip(4, 5, 6);
        std::cout << "FFT repeat, normalization and aliasing checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
