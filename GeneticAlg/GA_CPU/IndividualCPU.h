#ifndef HPC_INDIVIDUALCPU_H
#define HPC_INDIVIDUALCPU_H
#include <vector>

constexpr int POLY_SIZE_CPU = 5;

struct IndividualCPU {
    std::vector<double> coeffs;

    IndividualCPU() : coeffs(POLY_SIZE_CPU, 0.0) {}
};

#endif //HPC_INDIVIDUALCPU_H