cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)



cuda_add_executable(
    main
    main.cu utils.h generator.h core.h core.cu
)


target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --O3
                       >)