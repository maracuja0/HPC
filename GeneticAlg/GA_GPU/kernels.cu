#include "kernels.cuh"

__global__
void setup_rng(curandState* states, int seed, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        curand_init(seed, idx, 0, &states[idx]);
}


__global__
void init_population_kernel(IndividualGPU* population,
                            curandState* states,
                            int population_size) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    curandState localState = states[idx];

    for (float & coeff : population[idx].coeffs) {
        float r = curand_uniform(&localState); // [0,1]
        coeff = r * 2.0f - 1.0f; // [-1,1]
    }

    states[idx] = localState;
}

__global__
void fitness_kernel(
    IndividualGPU* population,
    const double* points_x,
    const double* points_y,
    double* fitness,
    int num_points,
    int population_size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    double error = 0.0;

    for (int i = 0; i < num_points; i++) {

        double x = points_x[i];
        double y = points_y[i];

        double y_pred = 0.0;
        double x_pow = 1.0;

        for (float coeff : population[idx].coeffs) {
            y_pred += coeff * x_pow;
            x_pow *= x;
        }

        double diff = y_pred - y;
        error += diff * diff;
    }

    fitness[idx] = error / static_cast<double>(num_points);
}

__global__
void crossover_kernel(
    const IndividualGPU* parents,
    IndividualGPU* children,
    const int* sorted_idx,
    curandState* states,
    int population_size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    curandState localState = states[idx];

    const int elite_count = max(1, population_size / 20);
    const int parent_pool = max(2, population_size / 2);

    if (idx < elite_count) {
        children[idx] = parents[sorted_idx[idx]];
        states[idx] = localState;
        return;
    }

    int r1 = min(static_cast<int>(curand_uniform(&localState) * static_cast<float>(parent_pool)),
              parent_pool - 1);
    int r2 = min(static_cast<int>(curand_uniform(&localState) * static_cast<float>(parent_pool)),
                 parent_pool - 1);

    if (r2 == r1) {
        r2 = (r2 + 1) % parent_pool;
    }

    int p1 = sorted_idx[r1];
    int p2 = sorted_idx[r2];

    int cut = 1 + min((int)(curand_uniform(&localState) * (POLY_SIZE - 1)), POLY_SIZE - 2);

    IndividualGPU child{};

    for (int i = 0; i < POLY_SIZE; ++i) {
        child.coeffs[i] = (i < cut)
            ? parents[p1].coeffs[i]
            : parents[p2].coeffs[i];
    }

    children[idx] = child;
    states[idx] = localState;
}


__global__
void mutation_kernel(IndividualGPU* population,
                     curandState* states,
                     int population_size,
                     float mutation_rate,
                     float Em,
                     float sigma)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    curandState localState = states[idx];

    for (float &coeff : population[idx].coeffs)
    {
        float r = curand_uniform(&localState);

        if (r < mutation_rate)
        {
            float z = curand_normal(&localState);
            float delta = Em + sigma * z;
            coeff += delta;
        }
    }

    states[idx] = localState;
}
