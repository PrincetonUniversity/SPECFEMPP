#include <gtest/gtest.h>
#include <stdexcept>
#include "../../include/gll_library.h"
#include <iostream>

TEST(GLL_tests, PNLEG){
    const auto& pnleg = gll_library::pnleg;
    float tol = 1e-6;

    try {
        gll_library::pnleg(-1.0, 0);
        FAIL();
    } 
    catch(const std::invalid_argument& err){
        EXPECT_STREQ(err.what(), "value of n > 0");
    }

    EXPECT_NEAR(pnleg(-1.0, 1), -1.0, tol);
    EXPECT_NEAR(pnleg(0.0, 1), 0.0, tol);
    EXPECT_NEAR(pnleg(1.0, 1), 1.0, tol);
    EXPECT_NEAR(pnleg(-1.0, 2), 1.0, tol);
    EXPECT_NEAR(pnleg(0.0, 2), -0.5, tol);
    EXPECT_NEAR(pnleg(1.0, 2), 1.0, tol);
    EXPECT_NEAR(pnleg(-1.0, 5), -1.0, tol);
    EXPECT_NEAR(pnleg(0.0, 5), 0.0, tol);
    EXPECT_NEAR(pnleg(1.0, 5), 1.0, tol);
}

TEST(GLL_tests, PNDLEG){
    const auto& pndleg = gll_library::pndleg;

    float tol = 1e-6;
    try {
        gll_library::pndleg(-1.0, 0);
        FAIL();
    } 
    catch(const std::invalid_argument& err){
        EXPECT_STREQ(err.what(), "value of n > 0");
    }

    EXPECT_NEAR(pndleg(-1.0, 1), 1.0, tol);
    EXPECT_NEAR(pndleg(0.0, 1), 1.0, tol);
    EXPECT_NEAR(pndleg(1.0, 1), 1.0, tol);
    EXPECT_NEAR(pndleg(0.0, 2), 0.0, tol);
    EXPECT_NEAR(pndleg(-0.2852315, 5), 0.0, tol);
    EXPECT_NEAR(pndleg(0.2852315, 5), 0.0, tol);
    EXPECT_NEAR(pndleg(-0.7650553, 5), 0.0, tol);
    EXPECT_NEAR(pndleg(0.7650553, 5), 0.0, tol);
}