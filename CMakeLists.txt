cmake_minimum_required(VERSION 3.14)
project(taper-hashmap-bench CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# Don't override CXX_FLAGS if user provided them via command line
if(NOT CMAKE_CXX_FLAGS_RELEASE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=armv8-a+crc")
endif()
if(NOT CMAKE_CXX_FLAGS_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=armv8-a+crc")
endif()

include(FetchContent)
FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.8.3)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(xxhash GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git GIT_TAG v0.8.2)
FetchContent_MakeAvailable(xxhash)

add_executable(taper_bench src/taper_bench.cpp)
target_include_directories(taper_bench PRIVATE include ${xxhash_SOURCE_DIR})
target_link_libraries(taper_bench PRIVATE benchmark::benchmark)

add_executable(profile_taper src/profile_taper.cpp)
target_include_directories(profile_taper PRIVATE include ${xxhash_SOURCE_DIR})

add_executable(taper_test src/taper_test.cpp)
target_include_directories(taper_test PRIVATE include ${xxhash_SOURCE_DIR})

add_executable(profile_taper_4str src/profile_taper_4str.cpp)
target_include_directories(profile_taper_4str PRIVATE include ${xxhash_SOURCE_DIR})
