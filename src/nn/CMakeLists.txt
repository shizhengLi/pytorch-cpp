set(NN_SOURCES
    linear.cpp
    activation.cpp
    sequential.cpp
    loss.cpp
    module.cpp
)

add_library(nn STATIC ${NN_SOURCES})
target_include_directories(nn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(nn PUBLIC cxx_std_17)
target_link_libraries(nn PUBLIC tensor autograd) 