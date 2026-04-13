#ifndef HPC_GENETICALGORITHMCPU_H
#define HPC_GENETICALGORITHMCPU_H

#include <random>
#include <vector>

#include "IndividualCPU.h"

struct GAResultCPU {
    double best_fitness = DBL_MAX;
    std::vector<double> coeffs;
    double cpu_time_ms = 0.0;
    int last_generation = 0;
};

class GeneticAlgorithmCPU
{
public:
    GeneticAlgorithmCPU(int population_size, int max_iter, int max_const_iter);

    GAResultCPU fit(const std::vector<double>& x,
                    const std::vector<double>& y,
                    float Em = 0.0f,
                    float Dm = 0.01f);

private:
    int population_size;
    int max_generations;
    int max_const_generations;

    std::vector<IndividualCPU> population;
    std::vector<IndividualCPU> children;
    std::vector<double> fitness;
    std::vector<int> idx;

    std::vector<double> x_host;
    std::vector<double> y_host;

    std::mt19937 rng;

    void init_cpu(const std::vector<double>& x, const std::vector<double>& y);
    void evaluate(int num_points);
    void selection();
    void crossover();
    void mutation(float Em, float Dm);
};


#endif //HPC_GENETICALGORITHMCPU_H