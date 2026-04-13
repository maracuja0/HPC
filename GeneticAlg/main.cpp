#include "GeneticAlgorithmGPU.h"
#include "GeneticAlgorithmCPU.h"
#include <chrono>
#include <iomanip>

#if defined(_WIN32)
#include <Windows.h>
#endif

void print_CUDA_info()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA устройств: " << deviceCount << std::endl;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
}

double function(double x)
{
    return 1 - 5 * x + 20 * std::pow(x,2) - 4 * std::pow(x,3) + 5 * std::pow(x,4);
}

int main()
{

#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8); //UTF-8 кодировка для корректного вывода в консоль
#endif

    print_CUDA_info();

    const int num_points = 1000;
    std::vector<double> x(num_points);
    std::vector<double> y(num_points);

    for (int i = 0; i < num_points; i++) {
        double v = -1.0 + 2.0 * i / static_cast<double>(num_points);
        x[i] = v;
        y[i] = function(v);
    }

    const int population_size = 2000;
    const int max_iter = 2000;
    const int max_const_iter = 200;

    // cpu
    std::cout << "===== CPU =====\n";
    GeneticAlgorithmCPU ga_cpu(population_size, max_iter, max_const_iter);
    auto start_time_cpu = std::chrono::high_resolution_clock::now();
    GAResultCPU result_cpu = ga_cpu.fit(x, y);
    auto end_time_cpu = std::chrono::high_resolution_clock::now();
    auto total_ms_cpu = std::chrono::duration<double, std::milli>(end_time_cpu - start_time_cpu).count();

    // gpu
    std::cout << "\n===== GPU =====\n";
    GeneticAlgorithmGPU ga_gpu(population_size, max_iter, max_const_iter);
    auto start_time_gpu = std::chrono::high_resolution_clock::now();
    GAResultGPU result_gpu = ga_gpu.fit(x, y);
    auto end_time_gpu = std::chrono::high_resolution_clock::now();
    auto total_ms_gpu = std::chrono::duration<double, std::milli>(end_time_gpu - start_time_gpu).count();

    std::cout << "\n\n===== Parameters =====\n\n";
    std::cout << std::left << std::setw(20) << "Num Points |"
                          << std::setw(20) << "Population Size |"
                          << std::setw(20) << "Max Iterations |"
                          << std::setw(20) << "Max const iterations |"

    << std::endl;

    std::cout << std::left << std::setw(20) << num_points
                          << std::setw(20) << population_size
                          << std::setw(20) << max_iter
                          << std::setw(20) << max_const_iter

    << std::endl;

    std::cout << "\n\n===== GA RESULT CPU =====\n";
    std::cout << "Best fitness: " << result_cpu.best_fitness << "\n";

    std::cout << "Total algorithm time: " << total_ms_cpu << " ms\n";
    std::cout << "CPU time: " << result_cpu.cpu_time_ms << " ms\n";
    std::cout << "Last generation: " << result_cpu.last_generation << "\n";

    std::cout << "Coefficients:\n";
    for (size_t i = 0; i < result_cpu.coeffs.size(); i++) {
        std::cout << "  c" << i << " = " << std::round(result_cpu.coeffs[i]) << "\n";
    }


    std::cout << "\n\n===== GA RESULT GPU =====\n";
    std::cout << "Best fitness: " << result_gpu.best_fitness << "\n";

    std::cout << "Total algorithm time: " << total_ms_gpu << " ms\n";
    std::cout << "GPU time (without mem copy): " << result_gpu.gpu_time << " ms\n";
    std::cout << "Last generation: " << result_gpu.last_generation << "\n";

    std::cout << "Coefficients:\n";
    for (size_t i = 0; i < result_gpu.coeffs.size(); i++) {
        std::cout << "  c" << i << " = " << std::round(result_gpu.coeffs[i]) << "\n";
    }

    double speedup_total = total_ms_cpu / total_ms_gpu;
    double speedup_gpu_only = result_cpu.cpu_time_ms / result_gpu.gpu_time;

    std::cout << "\n\n===== SPEEDUP =====\n";
    std::cout << std::setprecision(3);
    std::cout << "Total speedup (CPU total / GPU total): " << speedup_total << "x\n";
    std::cout << "Compute speedup (CPU time / GPU time): " << speedup_gpu_only << "x\n";

    return 0;
}