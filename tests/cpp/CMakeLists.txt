enable_testing()

# 添加测试执行文件
add_executable(tensor_test tensor_test.cpp)
add_executable(autograd_test autograd_test.cpp)
add_executable(nn_test nn_test.cpp)
add_executable(optim_test optim_test.cpp)

# 链接测试库
target_link_libraries(tensor_test PRIVATE tensor GTest::gtest_main)
target_link_libraries(autograd_test PRIVATE tensor autograd GTest::gtest_main)
target_link_libraries(nn_test PRIVATE tensor autograd nn GTest::gtest_main)
target_link_libraries(optim_test PRIVATE tensor autograd nn optim GTest::gtest_main)

# 添加测试
include(GoogleTest)
gtest_discover_tests(tensor_test)
gtest_discover_tests(autograd_test)
gtest_discover_tests(nn_test)
gtest_discover_tests(optim_test) 