constexpr int smallLimit = 8;
using SmallVector = std::array<double, smallLimit>;
using SmallMatrix = std::array<SmallVector, smallLimit>;

struct SmallSpectrum {
    SmallVector values{};
    SmallMatrix vectors{};
};

SmallSpectrum symmetricSpectrum(SmallMatrix matrix, int n) {
    SmallSpectrum result;
    for (int i = 0; i < n; ++i) result.vectors[i][i] = 1.0;
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool changed = false;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double off = matrix[p][q];
                const double scale = std::max(std::abs(matrix[p][p]), std::abs(matrix[q][q]));
                if (std::abs(off) <= 1e-15 * scale) continue;
                changed = true;
                const double theta = 0.5 * std::atan2(2 * off, matrix[q][q] - matrix[p][p]);
                const double c = std::cos(theta), s = std::sin(theta);
                const double pp = matrix[p][p], qq = matrix[q][q];
                matrix[p][p] = c * c * pp + s * s * qq - 2 * c * s * off;
                matrix[q][q] = s * s * pp + c * c * qq + 2 * c * s * off;
                matrix[p][q] = matrix[q][p] = 0;
                for (int i = 0; i < n; ++i) {
                    if (i != p && i != q) {
                        const double ip = matrix[i][p], iq = matrix[i][q];
                        matrix[i][p] = matrix[p][i] = c * ip - s * iq;
                        matrix[i][q] = matrix[q][i] = s * ip + c * iq;
                    }
                    const double ip = result.vectors[i][p], iq = result.vectors[i][q];
                    result.vectors[i][p] = c * ip - s * iq;
                    result.vectors[i][q] = s * ip + c * iq;
                }
            }
        }
        if (!changed) break;
    }
    for (int i = 0; i < n; ++i) result.values[i] = matrix[i][i];
    for (int i = 0; i < n; ++i) {
        const int greatest = std::max_element(result.values.begin() + i, result.values.begin() + n) -
                             result.values.begin();
        std::swap(result.values[i], result.values[greatest]);
        for (int row = 0; row < n; ++row) std::swap(result.vectors[row][i], result.vectors[row][greatest]);
    }
    return result;
}

struct SmallResponse {
    SmallVector amplitudes{};
    int rank = 0;
    double condition = 0.0;
};

SmallResponse regularizedResponse(const SmallMatrix& gram, const SmallVector& rhs, int rank) {
    const auto spectrum = symmetricSpectrum(gram, rank);
    SmallResponse fit;
    const double largest = spectrum.values[0];
    if (!(largest > 0) || !std::isfinite(largest)) return fit;
    const double ridge = 1e-6 * largest;
    for (int m = 0; m < rank; ++m) {
        const double value = spectrum.values[m];
        if (!(value > 1e-12 * largest)) break;
        double component = 0.0;
        for (int i = 0; i < rank; ++i) component += spectrum.vectors[i][m] * rhs[i];
        for (int i = 0; i < rank; ++i) fit.amplitudes[i] += spectrum.vectors[i][m] * component / (value + ridge);
        ++fit.rank;
        fit.condition = std::sqrt(largest / value);
    }
    return fit;
}
