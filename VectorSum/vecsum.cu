#include <chrono>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <random>
#include <string>

#if defined(_WIN32)
#include <Windows.h>
#endif

#define DEBUG false

const size_t SIZES[] = { 1000, 5000, 10000, 50000, 100000, 500000, 1000000 };
const int N_REPEATS = 15;
const size_t BLOCK_SIZE = 512;

inline void print_CUDA_info()
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

template <typename T>
__global__ void vecsum_kernel(const T* a, T* b, size_t n)
{
    __shared__ T sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    double sum = 0.0;

    if (idx < n) sum += (double)a[idx];
    if (idx + 1 < n) sum += (double)a[idx + 1];

    sdata[tid] = (T)sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        b[blockIdx.x] = sdata[0];
}


template <typename T>
float vecsum_gpu(const T* h_a, T* result, size_t n)
{
    size_t size_a = n * sizeof(T);

    T* d_a;
    T* d_b;

    cudaMalloc(&d_a, size_a);

    size_t max_blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    cudaMalloc(&d_b, max_blocks * sizeof(T));

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    size_t cur_size = n;
    T* cur_in = d_a;
    T* cur_out = d_b;

    size_t blocks = 0;

    while (true)
    {
        blocks = (cur_size + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        vecsum_kernel<T><<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(T)>>>(
            cur_in, cur_out, cur_size
        );

        if (blocks == 1)
            break;

        cur_size = blocks;

        std::swap(cur_in, cur_out);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(result, cur_out, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

template <typename T>
void vecsum_cpu(const T* a, T* b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += (double)a[i];
    }
    *b = (T)sum;
}

template <typename T>
void generate_vector(T* a, size_t n)
{
    thread_local static std::mt19937 gen(std::random_device{}());

    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dis(0, 100);

        for (size_t i = 0; i < n; ++i)
        {
            a[i] = dis(gen);
        }
    }
    else
    {
        std::uniform_real_distribution<T> dis(0.0, 1.0);

        for (size_t i = 0; i < n; ++i)
        {
            a[i] = dis(gen);
        }
    }
}

template <typename T>
void run_one_test(size_t n)
{

    T* h_a = new T[n];

    T sum_cpu = 0;
    T sum_gpu = 0;

    generate_vector(h_a, n);

    double cpu_time_ms = 0.0f;
    double gpu_time_ms = 0.0f;
    double kernel_time_ms = 0.0f;
    // CPU
    try
    {
        for (size_t rep = 0; rep < N_REPEATS; rep++)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            vecsum_cpu(h_a, &sum_cpu, n);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cpu_time_ms /= N_REPEATS;
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    //GPU
    try
    {
        for (size_t rep = 0; rep < N_REPEATS; rep++)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            kernel_time_ms += vecsum_gpu<T>(h_a, &sum_gpu, n);
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        gpu_time_ms /= N_REPEATS;
        kernel_time_ms =  kernel_time_ms / N_REPEATS;

    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    double S = cpu_time_ms / gpu_time_ms;
    double Sk = cpu_time_ms / kernel_time_ms;


    std::cout << std::fixed << std::setprecision(3) << std::left
        << std::setw(15) << std::to_string(n)
        << std::setw(15) << cpu_time_ms
        << std::setw(10) << gpu_time_ms
        << std::setw(10) << kernel_time_ms
        << std::setw(10) << S
        << std::setw(10) << Sk
        << std::setw(15) << sum_cpu
        << std::setw(15) << sum_gpu
    << std::endl;

    delete[] h_a;
}

template <typename T>
void run_tests()
{
    std::cout << "Type: " << typeid(T).name() << std::endl;
    std::cout << std::left << std::setprecision(3) << std::fixed;
    std::cout << std::setw(15) << "Размер"
              << std::setw(15) << "CPU(ms)"
              << std::setw(10) << "GPU(ms)"
              << std::setw(15) << "Kernel(ms)"
              << std::setw(10) << "S_total"
              << std::setw(10) << "S_kernel"
              << std::setw(15) << "Sum_cpu"
              << std::setw(15) << "S_gpu"
    << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    for (auto size: SIZES)
    {
        run_one_test<T>(size);
    }

    std::cout << std::endl;
}

template <typename T>
void warmup_gpu(size_t n, size_t warmup_iters = 5)
{
    T* h_a = new T[n];
    T sum_gpu = 0;
    generate_vector(h_a, n);

    for (size_t it = 0; it < warmup_iters; it++)
    {
        vecsum_gpu<T>(h_a, &sum_gpu, n);
    }

    delete[] h_a;
}

int main() {
#if defined(_WIN32)

    SetConsoleOutputCP(CP_UTF8); //UTF-8 кодировка для корректного вывода в консоль

#endif

    print_CUDA_info();

    warmup_gpu<double>(100);

    run_tests<int>();
    run_tests<float>();
    run_tests<double>();

    return 0;
}