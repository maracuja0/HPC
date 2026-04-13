add_subdirectory(GA_GPU)
add_subdirectory(GA_cpu)

add_executable(geneticalg main.cpp)
target_link_libraries(geneticalg PRIVATE GA GA_CPU CUDA::cudart)

set_target_properties(geneticalg PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

if(CMAKE_CUDA_COMPILER)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:
            $<$<CONFIG:Release>:-O3 --use_fast_math>
            >)
endif()