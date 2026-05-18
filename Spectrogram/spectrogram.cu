#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fftw3.h>
#include "EasyBMP.h"

#if defined(_WIN32)
#include <Windows.h>
#endif

constexpr float PI = 3.14159265358979323846f;

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

#pragma pack(push, 1)
struct WAVHeader {
    char     chunkId[4];
    uint32_t chunkSize;
    char     format[4];
    char     subchunk1Id[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char     subchunk2Id[4];
    uint32_t subchunk2Size;
};
#pragma pack(pop)

struct AudioData {
    std::vector<float> samples;
    int sampleRate;
    int numChannels;
    int bitsPerSample;
};

AudioData readWAV(const std::string& filename) {
    std::cout << "\n=== Reading WAV file ===" << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    if (std::string(header.chunkId, 4) != "RIFF" ||
        std::string(header.format, 4) != "WAVE" ||
        header.audioFormat != 1) {
        throw std::runtime_error("Invalid or unsupported WAV file");
    }

    size_t bytesPerSample = header.bitsPerSample / 8;
    size_t numSamples = header.subchunk2Size / bytesPerSample;

    AudioData result;
    result.sampleRate = header.sampleRate;
    result.numChannels = header.numChannels;
    result.bitsPerSample = header.bitsPerSample;
    result.samples.resize(numSamples);

    if (header.bitsPerSample == 16) {
        std::vector<int16_t> raw(numSamples);
        file.read(reinterpret_cast<char*>(raw.data()), header.subchunk2Size);
        for (size_t i = 0; i < numSamples; ++i) {
            result.samples[i] = raw[i] / 32768.0f;
        }
    } else {
        throw std::runtime_error("Only 16-bit PCM is supported");
    }

    if (header.numChannels == 2) {
        std::vector<float> mono(numSamples / 2);
        for (size_t i = 0; i < mono.size(); ++i) {
            mono[i] = (result.samples[2*i] + result.samples[2*i+1]) / 2.0f;
        }
        result.samples = std::move(mono);
        result.numChannels = 1;
    }

    return result;
}

void print_wav_info(AudioData audio)
{
    std::cout << "Sample rate: " << audio.sampleRate << " Hz" << std::endl;
    std::cout << "Channels: " << audio.numChannels << std::endl;
    std::cout << "Bits per sample: " << audio.bitsPerSample << std::endl;
    std::cout << "Duration: " << audio.samples.size() / audio.sampleRate << " sec" << std::endl;
    std::cout << "Total samples: " << audio.samples.size() << std::endl;
}

RGBApixel amplitudeToColor(float amp) {
    amp = std::max(0.0f, std::min(1.0f, amp));

    RGBApixel pixel;
    pixel.Alpha = 0;

    if (amp < 0.125f) {
        // Черный -> Фиолетовый
        float t = amp / 0.125f;
        pixel.Red = static_cast<unsigned char>(0 + t * 128);
        pixel.Green = 0;
        pixel.Blue = static_cast<unsigned char>(0 + t * 255);
    }
    else if (amp < 0.25f) {
        // Фиолетовый -> Синий
        float t = (amp - 0.125f) / 0.125f;
        pixel.Red = static_cast<unsigned char>(128 * (1 - t));
        pixel.Green = 0;
        pixel.Blue = 255;
    }
    else if (amp < 0.375f) {
        // Синий -> Голубой
        float t = (amp - 0.25f) / 0.125f;
        pixel.Red = 0;
        pixel.Green = static_cast<unsigned char>(0 + t * 255);
        pixel.Blue = 255;
    }
    else if (amp < 0.5f) {
        // Голубой -> Зеленый
        float t = (amp - 0.375f) / 0.125f;
        pixel.Red = 0;
        pixel.Green = 255;
        pixel.Blue = static_cast<unsigned char>(255 * (1 - t));
    }
    else if (amp < 0.625f) {
        // Зеленый -> Желтый
        float t = (amp - 0.5f) / 0.125f;
        pixel.Red = static_cast<unsigned char>(0 + t * 255);
        pixel.Green = 255;
        pixel.Blue = 0;
    }
    else if (amp < 0.75f) {
        // Желтый -> Оранжевый
        float t = (amp - 0.625f) / 0.125f;
        pixel.Red = 255;
        pixel.Green = static_cast<unsigned char>(255 * (1 - t * 0.5f));
        pixel.Blue = 0;
    }
    else if (amp < 0.875f) {
        // Оранжевый -> Красный
        float t = (amp - 0.75f) / 0.125f;
        pixel.Red = 255;
        pixel.Green = static_cast<unsigned char>(127 * (1 - t));
        pixel.Blue = 0;
    }
    else {
        // Красный -> Белый
        float t = (amp - 0.875f) / 0.125f;
        pixel.Red = 255;
        pixel.Green = static_cast<unsigned char>(0 + t * 255);
        pixel.Blue = static_cast<unsigned char>(0 + t * 255);
    }

    return pixel;
}

void saveSpectrogramBMP(const std::vector<std::vector<float>>& spectrogram,
                        const std::string& filename) {

    if (spectrogram.empty() || spectrogram[0].empty()) {
        throw std::runtime_error("Empty spectrogram");
    }

    int timeFrames = spectrogram.size();
    int freqBins = spectrogram[0].size();

    float maxVal = spectrogram[0][0];
    for (int t = 0; t < timeFrames; t++) {
        for (int f = 0; f < freqBins; f++) {
            if (spectrogram[t][f] > maxVal) {
                maxVal = spectrogram[t][f];
            }
        }
    }

    BMP image;
    image.SetSize(timeFrames, freqBins);

    for (int t = 0; t < timeFrames; t++) {
        for (int f = 0; f < freqBins; f++) {
            float normVal = spectrogram[t][f] / maxVal;
            RGBApixel pixel = amplitudeToColor(normVal);
            image.SetPixel(t, freqBins - 1 - f, pixel);
        }
    }

    image.WriteToFile(filename.c_str());

    std::cout << "Saved: " << filename << std::endl;
}

std::vector<std::vector<float>> computeSpectrogramFFTW(
    const std::vector<float>& signal,
    int windowSize,
    int hopSize,
    int sampleRate) {

    int numWindows = (signal.size() - windowSize) / hopSize + 1;
    int freqBins = windowSize / 2 + 1;

    std::vector<std::vector<float>> spectrogram(numWindows, std::vector<float>(freqBins));

    std::vector<float> window(windowSize);
    float windowSum = 0;
    for (int i = 0; i < windowSize; ++i) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (windowSize - 1)));
        windowSum += window[i];
    }

    float energyNorm = sqrtf(windowSum * windowSum / windowSize);

    float* in = (float*)fftwf_malloc(sizeof(float) * windowSize);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * freqBins);

    fftwf_plan plan = fftwf_plan_dft_r2c_1d(windowSize, in, out, FFTW_ESTIMATE);

    float maxMag = 0.0f;
    std::vector<std::vector<float>> mags(numWindows, std::vector<float>(freqBins));

    for (int w = 0; w < numWindows; ++w) {
        int start = w * hopSize;
        for (int i = 0; i < windowSize; ++i) {
            in[i] = signal[start + i] * window[i];
        }

        fftwf_execute(plan);

        for (int f = 0; f < freqBins; ++f) {
            float mag = sqrtf(out[f][0] * out[f][0] + out[f][1] * out[f][1]);
            mag = mag * energyNorm / windowSize;
            mags[w][f] = mag;
            if (mag > maxMag) maxMag = mag;
        }
    }

    float minDb = -80.0f;
    float maxDb = 20.0f * log10f(maxMag + 1e-10f);

    for (int w = 0; w < numWindows; ++w) {
        for (int f = 0; f < freqBins; ++f) {
            float db = 20.0f * log10f(mags[w][f] + 1e-10f);
            spectrogram[w][f] = (db - minDb) / (maxDb - minDb);
            spectrogram[w][f] = std::max(0.0f, std::min(1.0f, spectrogram[w][f]));
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    return spectrogram;
}

__global__ void magnitudeKernel(
    const cufftComplex* __restrict__ input,
    float* __restrict__ output,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    float real = input[idx].x;
    float imag = input[idx].y;

    output[idx] = sqrtf(real * real + imag * imag);
}

std::vector<std::vector<float>> computeSpectrogramCUFFT(
    const std::vector<float>& signal,
    int windowSize,
    int hopSize,
    int sampleRate,
    float& gpuTimeMs)
{
    int numWindows = (signal.size() - windowSize) / hopSize + 1;
    int freqBins = windowSize / 2 + 1;
    int totalElems = numWindows * freqBins;

    std::vector<float> window(windowSize);
    float windowSum = 0;
    for (int i = 0; i < windowSize; ++i) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (windowSize - 1)));
        windowSum += window[i];
    }
    float energyNorm = sqrtf(windowSum * windowSum / windowSize);
    float amplitudeNorm = energyNorm / windowSize;

    std::vector<float> allWindows(numWindows * windowSize);
    for (int w = 0; w < numWindows; ++w) {
        int start = w * hopSize;
        for (int i = 0; i < windowSize; ++i) {
            allWindows[w * windowSize + i] = signal[start + i] * window[i];
        }
    }

    cufftReal* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    float* d_mag = nullptr;

    cudaMalloc(&d_in, sizeof(cufftReal) * numWindows * windowSize);
    cudaMalloc(&d_out, sizeof(cufftComplex) * totalElems);
    cudaMalloc(&d_mag, sizeof(float) * totalElems);

    cudaMemcpy(d_in, allWindows.data(),
               sizeof(float) * numWindows * windowSize,
               cudaMemcpyHostToDevice);

    cufftHandle plan;
    int n[1] = { windowSize };

    int inembed[1]  = { windowSize };
    int onembed[1]  = { windowSize };

    cufftPlanMany(&plan,
                  1, n,
                  inembed, 1, windowSize,
                  onembed, 1, freqBins,
                  CUFFT_R2C,
                  numWindows);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cufftExecR2C(plan, d_in, d_out);

    int blockSize = 256;
    int gridSize = (totalElems + blockSize - 1) / blockSize;

    magnitudeKernel<<<gridSize, blockSize>>>(d_out, d_mag, totalElems);

    cudaDeviceSynchronize();

    std::vector<float> spectrogramFlat(totalElems);
    cudaMemcpy(spectrogramFlat.data(), d_mag,
               sizeof(float) * totalElems,
               cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);

    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mag);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<std::vector<float>> spectrogram(numWindows, std::vector<float>(freqBins));

    for (int w = 0; w < numWindows; ++w) {
        for (int f = 0; f < freqBins; ++f) {
            spectrogramFlat[w * freqBins + f] *= amplitudeNorm;
        }
    }

    float maxMag = 0.0f;
    for (int w = 0; w < numWindows; ++w) {
        for (int f = 0; f < freqBins; ++f) {
            float mag = spectrogramFlat[w * freqBins + f];
            if (mag > maxMag) maxMag = mag;
        }
    }

    float minDb = -80.0f;
    float maxDb = 20.0f * log10f(maxMag + 1e-10f);

    for (int w = 0; w < numWindows; ++w) {
        for (int f = 0; f < freqBins; ++f) {
            float db = 20.0f * log10f(spectrogramFlat[w * freqBins + f] + 1e-10f);
            float normVal = (db - minDb) / (maxDb - minDb);
            spectrogram[w][f] = std::max(0.0f, std::min(1.0f, normVal));
        }
    }

    return spectrogram;
}


float compareSpectrograms(const std::vector<std::vector<float>>& a,
                          const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        return -1.0f;
    }

    float mse = 0.0f;
    int count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            float diff = a[i][j] - b[i][j];
            mse += diff * diff;
            ++count;
        }
    }
    mse /= count;

    return std::sqrt(mse);
}


int main() {
#if defined(_WIN32)

    SetConsoleOutputCP(CP_UTF8); //UTF-8 кодировка для корректного вывода в консоль

#endif
    std::cout << "Spectrogram Generator (CPU/GPU)" << std::endl;

    print_CUDA_info();

    std::string inputFile = R"(C:\Users\marac\Downloads\sample-15s.wav)";
    std::string outputFileCPU = "spectrogram_cpu.bmp";
    std::string outputFileGPU = "spectrogram_gpu.bmp";

    int windowSize = 2048;
    int hopSize = windowSize / 8;
    int numRuns = 3;
    try {
        AudioData audio = readWAV(inputFile);
        print_wav_info(audio);

        if (audio.samples.size() < windowSize) {
            throw std::runtime_error("Signal is shorter than window size");
        }

        std::cout << "\n=== FFTW (CPU) Processing ===" << std::endl;
        double totalCpuTime = 0;
        std::vector<std::vector<std::vector<float>>> cpuResults;

        for (int run = 0; run < numRuns; ++run) {
            std::cout << (run + 1) << "/" << numRuns << " " << std::flush;

            auto startCPU = std::chrono::high_resolution_clock::now();
            auto spectrogramCPU = computeSpectrogramFFTW(audio.samples, windowSize, hopSize, audio.sampleRate);
            auto endCPU = std::chrono::high_resolution_clock::now();

            double cpuTimeMs = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();
            totalCpuTime += cpuTimeMs;
            cpuResults.push_back(spectrogramCPU);

            std::cout << cpuTimeMs << " ms" << std::endl;
        }

        double avgCpuTime = totalCpuTime / numRuns;
        std::cout << "Average CPU time: " << avgCpuTime << " ms" << std::endl;
        saveSpectrogramBMP(cpuResults[0], outputFileCPU);

        std::cout << "\n=== CUFFT (GPU) Processing ===" << std::endl;
        double totalGpuTime = 0;
        std::vector<std::vector<std::vector<float>>> gpuResults;

        for (int run = 0; run < numRuns; ++run) {
            std::cout << (run + 1) << "/" << numRuns << " " << std::flush;

            float gpuTimeMs = 0;
            auto spectrogramGPU = computeSpectrogramCUFFT(audio.samples, windowSize, hopSize, audio.sampleRate, gpuTimeMs);
            totalGpuTime += gpuTimeMs;
            gpuResults.push_back(spectrogramGPU);

            std::cout << gpuTimeMs << " ms" << std::endl;
        }

        double avgGpuTime = totalGpuTime / numRuns;
        std::cout << "Average GPU time: " << avgGpuTime << " ms" << std::endl;
        saveSpectrogramBMP(gpuResults[0], outputFileGPU);

        std::cout << "\n=== Comparison ===" << std::endl;
        float rmse = compareSpectrograms(cpuResults[0], gpuResults[0]);
        std::cout << "RMSE between CPU and GPU spectrograms: " << rmse << std::endl;

        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << "FFTW (CPU):  " << avgCpuTime << " ms" << std::endl;
        std::cout << "CUFFT (GPU): " << avgGpuTime << " ms" << std::endl;
        std::cout << "Speedup:     " << (avgCpuTime / avgGpuTime) << "x" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}