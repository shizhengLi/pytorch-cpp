cmake_minimum_required(VERSION 3.14)
project(pytorchcpp VERSION 0.1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 添加PIC支持

# 输出路径配置
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加include目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 第三方依赖
include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# pybind11 (Python绑定)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

# 添加子目录
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bindings)
add_subdirectory(examples)

# 安装配置
install(DIRECTORY include/ DESTINATION include) 