add_library(GA_CPU STATIC
        GeneticAlgorithmCPU.cpp
)

target_include_directories(GA_CPU PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})