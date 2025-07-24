set(AUTOGRAD_SOURCES
    function.cpp
    variable.cpp
)

add_library(autograd STATIC ${AUTOGRAD_SOURCES})
target_include_directories(autograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(autograd PUBLIC cxx_std_17)
target_link_libraries(autograd PUBLIC tensor) 