#include "fk_injection/fk_injection.hpp"

#include <gtest/gtest.h>

#include <type_traits>

TEST(FkPluginSkeleton, TypeRealFollowsPrecisionOption) {
#ifdef FK_INJECTION_EXPECT_DOUBLE
  EXPECT_TRUE((std::is_same_v<fk_injection::type_real, double>));
#else
  EXPECT_TRUE((std::is_same_v<fk_injection::type_real, float>));
#endif
}
