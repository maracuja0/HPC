add_executable(matmul matmul.cu)
target_link_libraries(matmul PRIVATE CUDA::cudart)

set_target_properties(matmul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)