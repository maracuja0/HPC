#ifndef HPC_GENETICALGORITHMGPU_H
#define HPC_GENETICALGORITHMGPU_H

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <curand_kernel.h>

#include "IndividualGPU.h"
#include "kernels.cuh"


struct GAResultGPU {
    double best_fitness;
    std::vector<float> coeffs;
    float gpu_time;
    int last_generation;
};

class GeneticAlgorithmGPU
{
public:
    GeneticAlgorithmGPU(int population_size, int max_iter, int max_const_iter);
    ~GeneticAlgorithmGPU();
    GAResultGPU fit(const std::vector<double>& x,
             const std::vector<double>& y, float Em=0.0f, float Dm=0.01f);

private:
    int population_size;
    int max_generations;
    int max_const_generations;

    thrust::device_vector<IndividualGPU> d_population;
    thrust::device_vector<IndividualGPU> d_children;
    thrust::device_vector<double> d_fitness;
    thrust::device_vector<int> d_idx;

    double* d_x = nullptr;
    double* d_y = nullptr;
    curandState* d_states = nullptr;

    void init_gpu(const std::vector<double>& x,
                  const std::vector<double>& y);

    void uninit_gpu();

    void evaluate(int num_points);
    void selection();
    void crossover();
    void mutation(float Em, float Dm);
};


#endif //HPC_GENETICALGORITHMGPU_H