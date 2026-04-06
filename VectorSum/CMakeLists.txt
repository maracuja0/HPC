add_executable(vecsum vecsum.cu)
target_link_libraries(vecsum PRIVATE CUDA::cudart)

set_target_properties(vecsum PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)