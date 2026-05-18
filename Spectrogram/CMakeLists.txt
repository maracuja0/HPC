set(EASYBMP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/EasyBMP)
set(FFTW_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/fftw-3.3.5-dll64)

add_executable(spectrogram spectrogram.cu)
target_sources(spectrogram PRIVATE ${EASYBMP_INCLUDE_DIR}/EasyBMP.cpp)

target_include_directories(spectrogram PRIVATE
        ${EASYBMP_INCLUDE_DIR}
        ${FFTW_LIB_DIR}
)

target_link_libraries(spectrogram PRIVATE
        CUDA::cudart
        cufft
        ${FFTW_LIB_DIR}/libfftw3f-3.lib
)

add_custom_command(TARGET spectrogram POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FFTW_LIB_DIR}/libfftw3f-3.dll
        $<TARGET_FILE_DIR:spectrogram>
        COMMENT "Copying libfftw3f-3.dll to output directory"
)