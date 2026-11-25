// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int tileWidth = 8;
constexpr double erfcCutoff = 8.0;

struct ParticleGrid {
    int nodes[3], tiles[3];
    double length[3], dx[3];
};

struct RenderParticle {
    double center[3], radius, width;
};

__global__ void fillParticleTiles(ParticleGrid grid, int sites, double scale,
                                  const RenderParticle* particles, const int* offsets,
                                  const int* indices, thrust::complex<double>* density) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) return;
    const int x = i % grid.nodes[0];
    const int y = (i / grid.nodes[0]) % grid.nodes[1];
    const int z = i / grid.nodes[0] / grid.nodes[1];
    const int tile = x / tileWidth + grid.tiles[0] *
        (y / tileWidth + grid.tiles[1] * (z / tileWidth));
    const double point[] = {x * grid.dx[0], y * grid.dx[1], z * grid.dx[2]};
    double sum = 0.0;
    for (int entry = offsets[tile]; entry < offsets[tile + 1]; ++entry) {
        const auto& particle = particles[indices[entry]];
        double distance2 = 0.0;
        for (int j = 0; j < 3; ++j) {
            double dr = particle.center[j] - point[j];
            if (dr >= 0.5 * grid.length[j]) dr -= grid.length[j];
            else if (dr < -0.5 * grid.length[j]) dr += grid.length[j];
            distance2 += dr * dr;
        }
        const double t = (sqrt(distance2) - particle.radius) / particle.width;
        if (t <= erfcCutoff) sum += scale * erfc(t);
    }
    density[i] = sum;
}

std::vector<int> coveredTiles(const ParticleGrid& grid, int axis, double center, double reach) {
    if (!(center >= 0 && center <= grid.length[axis])) {
        throw std::runtime_error("Particle tiles require centers inside the periodic box");
    }
    std::vector<int> result;
    for (int tile = 0; tile < grid.tiles[axis]; ++tile) {
        const double low = tile * tileWidth * grid.dx[axis];
        const double high = (std::min((tile + 1) * tileWidth, grid.nodes[axis]) - 1) *
            grid.dx[axis];
        for (int image = -1; image <= 1; ++image) {
            const double shifted = center + image * grid.length[axis];
            if (low - reach <= shifted && shifted <= high + reach) {
                result.push_back(tile);
                break;
            }
        }
    }
    return result;
}

void renderParticleTiles(FTS_Box& box, ParticleMolec& molecule) {
    if (box.returnDimension() != 3) {
        throw std::runtime_error("Particle tiles require a 3D grid");
    }
    ParticleGrid grid{};
    int tileCount = 1;
    for (int j = 0; j < 3; ++j) {
        grid.nodes[j] = box.Nx[j];
        grid.tiles[j] = (box.Nx[j] + tileWidth - 1) / tileWidth;
        grid.length[j] = box.L[j];
        grid.dx[j] = box.dx[j];
        tileCount *= grid.tiles[j];
    }
    std::vector<RenderParticle> particles;
    std::vector<std::vector<int>> lists(tileCount);
    for (int p = 0; p < molecule.particleNum; ++p) {
        const RenderParticle particle{{molecule.center[3 * p], molecule.center[3 * p + 1],
            molecule.center[3 * p + 2]}, molecule.Rp[p], molecule.xi[p]};
        if (!(std::isfinite(particle.radius) && particle.radius >= 0 &&
              std::isfinite(particle.width) && particle.width > 0)) {
            throw std::runtime_error("Particle tiles require finite nonnegative radii and positive widths");
        }
        particles.push_back(particle);
        std::array<std::vector<int>, 3> covered;
        for (int j = 0; j < 3; ++j) {
            const double reach = particle.radius + erfcCutoff * particle.width +
                16 * std::numeric_limits<double>::epsilon() * grid.length[j];
            covered[j] = coveredTiles(grid, j, particle.center[j], reach);
        }
        for (const int z : covered[2]) {
            for (const int y : covered[1]) {
                for (const int x : covered[0]) {
                    lists[x + grid.tiles[0] * (y + grid.tiles[1] * z)].push_back(p);
                }
            }
        }
    }
    std::vector<int> offsets{0}, indices;
    for (const auto& list : lists) {
        if (list.size() > size_t(std::numeric_limits<int>::max()) - indices.size()) {
            throw std::runtime_error("Particle tile list is too large");
        }
        indices.insert(indices.end(), list.begin(), list.end());
        offsets.push_back(static_cast<int>(indices.size()));
    }
    thrust::device_vector<RenderParticle> deviceParticles(particles.begin(), particles.end());
    thrust::device_vector<int> deviceOffsets(offsets.begin(), offsets.end());
    thrust::device_vector<int> deviceIndices(indices.begin(), indices.end());
    molecule.d_NPdensity.resize(box.M);
    constexpr int threads = 256;
    fillParticleTiles<<<(box.M + threads - 1) / threads, threads>>>(
        grid, box.M, box.Nr * box.C * 0.5,
        thrust::raw_pointer_cast(deviceParticles.data()), thrust::raw_pointer_cast(deviceOffsets.data()),
        thrust::raw_pointer_cast(deviceIndices.data()), thrust::raw_pointer_cast(molecule.d_NPdensity.data()));
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    molecule.density = molecule.d_NPdensity;
    std::cout << "Particle tiles: " << tileCount << " tiles, " << indices.size() << " candidates\n";
}
}
