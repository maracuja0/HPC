#include "GeneticAlgorithmCPU.h"

#include <numeric>
#include <chrono>
#include <iostream>


GeneticAlgorithmCPU:: GeneticAlgorithmCPU(int population_size, int max_iter, int max_const_iter)
        : population_size(population_size),
          max_generations(max_iter),
          max_const_generations(max_const_iter),
          population(population_size),
          children(population_size),
          fitness(population_size, DBL_MAX),
          idx(population_size)
{
    std::random_device rd;
    rng.seed(rd());
}


GAResultCPU GeneticAlgorithmCPU::fit(const std::vector<double>& x, const std::vector<double>& y, float Em, float Dm)
{
    init_cpu(x, y);

    const int num_points = static_cast<int>(x.size());
    int n_const_iter = 0;
    double global_best_fitness = DBL_MAX;
    const double eps = 1e-8;

    auto start_time = std::chrono::steady_clock::now();

    int gen = 0;
    for (; gen < max_generations; ++gen)
    {
        evaluate(num_points);
        selection();

        double current_best = fitness[0];
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
            break;

        if (gen % 20 == 0) {
            std::cout << "Gen " << gen
            << " best fitness: " << fitness[0]
            << " stagnation: " << n_const_iter
            << std::endl;
        }

        crossover();
        mutation(Em, Dm);
    }

    evaluate(num_points);
    selection();

    auto end_time = std::chrono::steady_clock::now();

    GAResultCPU result;
    result.best_fitness = fitness[0];
    result.coeffs = population[idx[0]].coeffs;
    result.last_generation = gen;
    result.cpu_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result;
}


void GeneticAlgorithmCPU::init_cpu(const std::vector<double>& x, const std::vector<double>& y)
{
    x_host = x;
    y_host = y;

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& ind : population)
    {
        for (auto& c : ind.coeffs)
            c = dist(rng);
    }
}

void GeneticAlgorithmCPU::evaluate(int num_points)
{
    for (int i = 0; i < population_size; ++i)
    {
        double sum = 0.0;

        for (int p = 0; p < num_points; ++p)
        {
            double pred = 0.0;
            double xp = 1.0;

            for (int k = 0; k < POLY_SIZE_CPU; ++k)
            {
                pred += population[i].coeffs[k] * xp;
                xp *= x_host[p];
            }

            double diff = pred - y_host[p];
            sum += diff * diff;
        }

        fitness[i] = sum / num_points;
    }
}

void GeneticAlgorithmCPU::selection()
{
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b)
              {
                  return fitness[a] < fitness[b];
              });
}

void GeneticAlgorithmCPU::crossover()
{
    const int elite_count = std::max(1, population_size / 20);
    const int parent_pool = std::max(2, population_size / 2);

    for (int i = 0; i < elite_count; ++i)
        children[i] = population[idx[i]];

    std::uniform_int_distribution<int> parent_dist(0, parent_pool - 1);
    std::uniform_int_distribution<int> cut_dist(1, POLY_SIZE_CPU - 1);

    for (int i = elite_count; i < population_size; ++i)
    {
        int r1 = parent_dist(rng);
        int r2 = parent_dist(rng);
        if (r2 == r1) r2 = (r2 + 1) % parent_pool;

        const IndividualCPU& p1 = population[idx[r1]];
        const IndividualCPU& p2 = population[idx[r2]];
        int cut = cut_dist(rng);

        for (int k = 0; k < POLY_SIZE_CPU; ++k)
            children[i].coeffs[k] = (k < cut) ? p1.coeffs[k] : p2.coeffs[k];
    }

    population.swap(children);
}

void GeneticAlgorithmCPU::mutation(float Em, float Dm)
{
    const int elite_count = std::max(1, population_size / 20);
    const double sigma = std::sqrt(Dm);

    std::bernoulli_distribution mut_dist(0.01f);
    std::normal_distribution<double> noise(Em, sigma);

    for (int i = elite_count; i < population_size; ++i)
    {
        for (int k = 0; k < POLY_SIZE_CPU; ++k)
        {
            if (mut_dist(rng))
                population[i].coeffs[k] += noise(rng);
        }
    }
}
