#include "GeneticAlgorithmGPU.h"


GeneticAlgorithmGPU::GeneticAlgorithmGPU(int population_size, int max_iter, int max_const_iter)
: population_size(population_size), max_generations(max_iter), max_const_generations(max_const_iter),
d_population(population_size), d_children(population_size),
d_fitness(population_size), d_idx(population_size) {}

GeneticAlgorithmGPU::~GeneticAlgorithmGPU()= default;

GAResultGPU GeneticAlgorithmGPU::fit(const std::vector<double>& x, const std::vector<double>& y, const float Em, const float Dm)
{
    init_gpu(x, y);

    int num_points = x.size();

    float total_gpu_ms = 0.0f;
    float gpu_ms = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n_const_iter = 0;
    double global_best_fitness = DBL_MAX;
    const double eps = 1e-8;
    int gen = 0;
    for (; gen < max_generations; ++gen) {
        cudaEventRecord(start);

        evaluate(num_points);
        selection();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_ms, start, stop);
        total_gpu_ms += gpu_ms;

        double current_best;
        cudaMemcpy(&current_best,
                   thrust::raw_pointer_cast(d_fitness.data()),
                   sizeof(double),
                   cudaMemcpyDeviceToHost);

        if (current_best < global_best_fitness - eps)
        {
            global_best_fitness = current_best;
            n_const_iter = 0;
        }
        else
        {
            ++n_const_iter;
        }

        if (n_const_iter >= max_const_generations)
        {
            break;
        }

        cudaEventRecord(start);

        crossover();
        mutation(Em, Dm);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_ms, start, stop);

        total_gpu_ms += gpu_ms;

        if (gen % 20 == 0) {
            double best;
            cudaMemcpy(&best,
                  thrust::raw_pointer_cast(d_fitness.data()),
                  sizeof(double),
                  cudaMemcpyDeviceToHost);
            std::cout << "Gen " << gen
            << " best fitness: " << best
            << " stagnation: " << n_const_iter
            << std::endl;
        }
    }

    cudaEventRecord(start);

    evaluate(num_points);
    selection();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);
    total_gpu_ms += gpu_ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int best_idx;
    cudaMemcpy(&best_idx,
               thrust::raw_pointer_cast(d_idx.data()),
               sizeof(int),
               cudaMemcpyDeviceToHost);

    IndividualGPU best_individual;
    cudaMemcpy(&best_individual,
               thrust::raw_pointer_cast(d_population.data()) + best_idx,
               sizeof(IndividualGPU),
               cudaMemcpyDeviceToHost);

    GAResultGPU result;
    result.coeffs.resize(POLY_SIZE);

    for (int i = 0; i < POLY_SIZE; i++) {
        result.coeffs[i] = best_individual.coeffs[i];
    }

    double best;
    cudaMemcpy(&best,
        thrust::raw_pointer_cast(d_fitness.data()),
        sizeof(double),
        cudaMemcpyDeviceToHost);

    result.best_fitness = best;
    result.last_generation = gen;
    result.gpu_time = total_gpu_ms;

    uninit_gpu();

    return result;
}

void GeneticAlgorithmGPU::init_gpu(const std::vector<double>& x, const std::vector<double>& y)
{
    cudaMalloc(&d_x, x.size() * sizeof(double));
    cudaMalloc(&d_y, y.size() * sizeof(double));
    cudaMalloc(&d_states, population_size * sizeof(curandState));

    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (population_size + threads - 1) / threads;

    setup_rng<<<blocks, threads>>>(d_states, time(NULL), population_size);

    // Инициализация популяции
    init_population_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_population.data()),
        d_states,
        population_size
    );
}

void GeneticAlgorithmGPU::uninit_gpu()
{
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_states);

}

void GeneticAlgorithmGPU::evaluate(int num_points)
{
    int threads = 256;
    int blocks = (population_size + threads - 1) / threads;

    fitness_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_population.data()),
        d_x,
        d_y,
        thrust::raw_pointer_cast(d_fitness.data()),
        num_points,
        population_size
    );
}

void GeneticAlgorithmGPU::selection()
{
    thrust::sequence(d_idx.begin(), d_idx.end());
    thrust::sort_by_key(
        d_fitness.begin(),
        d_fitness.end(),
        d_idx.begin()
    );
}

void GeneticAlgorithmGPU::crossover()
{
    int threads = 256;
    int blocks = (population_size + threads - 1) / threads;

    crossover_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_population.data()),
        thrust::raw_pointer_cast(d_children.data()),
        thrust::raw_pointer_cast(d_idx.data()),
        d_states,
        population_size
    );

    std::swap(d_population, d_children);
}

void GeneticAlgorithmGPU::mutation(const float Em, const float Dm)
{
    int threads = 256;
    int blocks = (population_size + threads - 1) / threads;
    float sigma = sqrtf(Dm);

    mutation_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_population.data()),
        d_states,
        population_size,
        0.01f,
        Em, sigma
    );
}
