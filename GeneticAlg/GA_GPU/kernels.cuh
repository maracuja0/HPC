#ifndef HPC_KERNELS_CUH
#define HPC_KERNELS_CUH

#include <curand_kernel.h>
#include "IndividualGPU.h"

__global__
void setup_rng(curandState* states, int seed, int n);


__global__
void init_population_kernel(IndividualGPU* population,
                            curandState* states,
                            int population_size);

__global__
void fitness_kernel(IndividualGPU* population,
                    const double* points_x,
                    const double* points_y,
                    double* fitness,
                    int num_points,
                    int population_size);

__global__
void crossover_kernel(const IndividualGPU* parents,
                        IndividualGPU* children,
                        const int* sorted_idx,

                        curandState* states,
                        int population_size);

__global__
void mutation_kernel(IndividualGPU* population,
                     curandState* states,
                     int population_size,
                     float mutation_rate,
                     float Em,
                     float sigma);

#endif //HPC_KERNELS_CUH