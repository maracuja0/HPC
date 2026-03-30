//
// Created by marac on 14.03.2026.
//

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <random>
#include <string>

#if defined(_WIN32)
#include <Windows.h>
#endif


const size_t DIMS[] = { 100, 200, 400, 800, 1500, 1700, 2000 };
const size_t N_REPEATS = 3;
const size_t BLOCK_SIZE = 16;

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
__global__ void matmul_kernel(const T* A, const T* B, T* C, size_t M, size_t N, size_t K) {
    unsigned int m = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        T sum = 0;
        for (unsigned int k = 0; k < K; k++) {
            sum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = sum;
    }
}

template <typename T>
double matmul_gpu(const T* A, const T* B, T* C, size_t M, size_t N, size_t K)
{
    T* d_A, * d_B, * d_C;
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmul_kernel<T><<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time = 0.0f;
    cudaEventElapsedTime(&kernel_time, start, stop);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return kernel_time;
}


template <typename T>
void matmul_cpu(const T* A, const T* B, T* C, size_t M, size_t N, size_t K)
{

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            T sum = 0;
            for (size_t k = 0; k < K; k++)
            {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

template <typename T>
void generate_matrix(T* A, size_t M, size_t N)
{
    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i * N + j] = static_cast<T>(dis(gen));
        }
    }
}


template <typename T>
double get_max_error(const T* C1, const T* C2, size_t M, size_t N)
{
    double max_error { 0.0 };
    size_t size {M * N};

    for (size_t i = 0; i < size; i++) {
        double diff = std::abs(static_cast<double>(C1[i]) -
                               static_cast<double>(C2[i]));
        max_error = std::max(diff, max_error);
    }

    return max_error;
}


template <typename T>
void run_one_test(size_t M, size_t N, size_t K)
{
    double max_error = 0.0;

    T* h_A = new T[M * K];
    T* h_B = new T[K * N];
    T* h_C_cpu = new T[M * N];
    T* h_C_gpu = new T[M * N];

    generate_matrix(h_A, M, K);
    generate_matrix(h_B, K, N);

    double cpu_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double kernel_time_ms = 0.0;
    // CPU
    try
    {
        for (size_t rep = 0; rep < N_REPEATS; rep++)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
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
            kernel_time_ms += matmul_gpu(h_A, h_B, h_C_gpu, M, N, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        gpu_time_ms /= N_REPEATS;
        kernel_time_ms /= N_REPEATS;

    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    max_error = get_max_error(h_C_cpu, h_C_gpu, M, N);

    double S = cpu_time_ms / gpu_time_ms;
    double Sk = cpu_time_ms / kernel_time_ms;

    std::cout << std::fixed << std::setprecision(3) << std::left
        << std::setw(15) << (std::to_string(M) + "x" + std::to_string(N))
        << std::setw(15) << cpu_time_ms
        << std::setw(10) << gpu_time_ms
        << std::setw(10) << kernel_time_ms
        << std::setw(10) << S
        << std::setw(10) << Sk
        << std::setw(10) << max_error << std::endl;


    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
}

template <typename T>
void run_tests()
{
    std::cout << "Type: " << typeid(T).name() << std::endl;
    std::cout << std::left << std::fixed;
    std::cout << std::setw(15) << "Размер"
              << std::setw(15) << "CPU(ms)"
              << std::setw(10) << "GPU(ms)"
              << std::setw(15) << "Kernel(ms)"
              << std::setw(10) << "S_total"
              << std::setw(10) << "S_kernel"
              << std::setw(15) << "Max_error" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (auto dim: DIMS)
    {
        size_t M = dim;
        size_t N = dim;
        size_t K = dim;

        run_one_test<T>(M, N, K);
    }
    std::cout << std::endl;
}


template <typename T>
void warmup_gpu(size_t M, size_t N, size_t K, size_t warmup_iters = 5)
{
    T* h_A = new T[M * K];
    T* h_B = new T[K * N];
    T* h_C_gpu = new T[M * N];

    generate_matrix(h_A, M, K);
    generate_matrix(h_B, K, N);

    for (size_t it = 0; it < warmup_iters; it++)
    {
        matmul_gpu(h_A, h_B, h_C_gpu, M, N, K);
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
}


int main() {
#if defined(_WIN32)

    SetConsoleOutputCP(CP_UTF8); //UTF-8 кодировка для корректного вывода в консоль

#endif

    print_CUDA_info();

    warmup_gpu<double>(100, 100, 100);

    run_tests<int>();
    run_tests<float>();
    run_tests<double>();

    return 0;
}
