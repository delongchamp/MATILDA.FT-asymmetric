// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
#ifndef MATILDA_SCFT_RACE_H
#define MATILDA_SCFT_RACE_H

#include <cstddef>

class FTS_Box;
void scftRaceStage(const char* phase, int updates = 0, int evaluations = 0,
                   std::size_t requestedBytes = 0);
void runScftRace(FTS_Box& box, int nsteps, const char* mode);

#endif
