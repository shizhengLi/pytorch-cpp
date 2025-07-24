# 简单的矩阵运算示例
add_executable(matrix_ops matrix_ops.cpp)
target_link_libraries(matrix_ops PRIVATE tensor)

# 自动求导示例
add_executable(autograd_example autograd_example.cpp)
target_link_libraries(autograd_example PRIVATE tensor autograd)

# 线性回归示例
add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression PRIVATE tensor autograd nn optim)

# MNIST分类示例
add_executable(mnist_classification mnist_classification.cpp)
target_link_libraries(mnist_classification PRIVATE tensor autograd nn optim) 