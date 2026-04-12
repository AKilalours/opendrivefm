cmake_minimum_required(VERSION 3.18)
project(bench_latency CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Optimisation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# ── Find LibTorch ─────────────────────────────────────────────────────────────
# Set TORCH_PATH to your LibTorch installation:
#   cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(bench_latency bench_latency.cpp)
target_link_libraries(bench_latency "${TORCH_LIBRARIES}")
target_compile_features(bench_latency PRIVATE cxx_std_17)

# ── Apple Silicon / MPS note ──────────────────────────────────────────────────
if(APPLE)
    message(STATUS "Building on Apple — MPS backend available via Python")
    message(STATUS "C++ harness uses CPU backend for pure compute measurement")
endif()

set_property(TARGET bench_latency PROPERTY CXX_STANDARD 17)
