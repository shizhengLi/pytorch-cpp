set(OPTIM_SOURCES
    sgd.cpp
    adam.cpp
    optimizer.cpp
)

add_library(optim STATIC ${OPTIM_SOURCES})
target_include_directories(optim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(optim PUBLIC cxx_std_17)
target_link_libraries(optim PUBLIC tensor autograd) 