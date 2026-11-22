cmake_minimum_required(VERSION 3.16)
project(hypno_piecetable CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC in Tauri builds (the Rust toolchain pulls vcvars), MinGW g++ otherwise.
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # A MinGW DLL that links libstdc++/libgcc/libwinpthread dynamically fails
    # to load on a machine without MSYS2 on PATH (ctypes reports "or one of its
    # dependencies"). Static-link the runtimes so the DLL is self-contained.
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

set(HYPNO_SOURCES
    piece_table.cpp
    c_api.cpp
)

# Shared library: Tauri needs piecetable.dll for its extern "C" FFI, and
# koffi can load either .dll directly or this same shared target.
add_library(piecetable SHARED ${HYPNO_SOURCES})

# Single translation unit for the test harness keeps the build simple; a
# static lib plus executable would be marginally faster to link but this is
# a test binary, not a hot path.
add_executable(piecetable_test test_piecetable.cpp ${HYPNO_SOURCES})

add_executable(c_api_test test_c_api.cpp ${HYPNO_SOURCES})

if(NOT MSVC)
    target_link_libraries(piecetable_test pthread)
    target_link_libraries(c_api_test pthread)
endif()