add_library(GA STATIC
        GeneticAlgorithmGPU.cu
        kernels.cu
)

target_include_directories(GA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(GA PUBLIC CUDA::cudart)

set_target_properties(GA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)