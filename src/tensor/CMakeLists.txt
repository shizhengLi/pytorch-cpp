set(TENSOR_SOURCES
    tensor.cpp
)

add_library(tensor STATIC ${TENSOR_SOURCES})
target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(tensor PUBLIC cxx_std_17) 