//
// Created by 芦yafei  on 17/7/27.
//

// This sample shows how to write a simple unit test for a function,
// using Google C++ testing framework.
//这个示例用来展示如何应用Google C++测试框架来写一个简单的函数测试单元
// Writing a unit test using Google C++ testing framework is easy as 1-2-3:
//应用Google C++写一个单元测试很简单，只需要1-2-3共三个步骤

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//第一步:引入一些必须的头文件
// Don't forget gtest.h, which declares the testing framework.
//不要忘了引入声明测试框架的gtest.h头文件

#include "stdafx.h"
#include <limits.h>
#include "activation_function.h"
#include <gtest/gtest.h>

// Step 2. Use the TEST macro to define your tests.
//第二步：应用TEST宏来定义你的测试

//TEST has two parameters: the test case name and the test name.
//TEST宏包含两个参数：一个案例【TestCaseName】名，一个测试【TestName】名

// After using the macro, you should define your test logic between a  pair of braces.
//在应用了宏之后，你应该定义这两者之间的测试逻辑

//You can use a bunch of macros to indicate the success or failure of a test.
//你应该使用一些宏命令去指出测试是否成功

//EXPECT_TRUE and EXPECT_EQ are  examples of such macros.
//其中EXPECT_TRUE和EXPECT_EQ就是这样的宏的例子

// For a complete list, see gtest.h.
//要查看完整的列表，可以去看看gtest.h头文件

// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.
//在gtest中，测试都被分组到测试案例中

// This is how we keep test code organized.
//这就是我们有效组织测试代码的方式

//  You should put logically related tests into the same test case.
//你应该将这些逻辑上相关的测试应用到相似的测试案例中去

// The test case name and the test name should both be valid C++ identifiers.
//测试案例名和测试名应该都是有效的C++标识符

// And you should not use underscore (_) in the names.
//并且你不应该应用强调符号(_)到命名中

// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.
//gtest能够保证你定义的每个测试都能准确的运行一次，但是它并不能保证这些测试执行的顺序

//Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//所以你应该依输出结果不依赖自身顺序的方式来写这些测试案例

// </TechnicalDetails>

// Tests Factorial().
//测试阶乘函数Factorial().