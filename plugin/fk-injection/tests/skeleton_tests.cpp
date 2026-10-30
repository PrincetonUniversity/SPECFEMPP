#include "specfem/fk_plugin.hpp"

#include <gtest/gtest.h>

#include <type_traits>

TEST(FkPluginSkeleton, TypeRealFollowsPrecisionOption) {
#ifdef FK_PLUGIN_EXPECT_DOUBLE
  EXPECT_TRUE((std::is_same_v<specfem::fk_plugin::type_real, double>));
#else
  EXPECT_TRUE((std::is_same_v<specfem::fk_plugin::type_real, float>));
#endif
  EXPECT_TRUE((std::is_same_v<specfem::fk_plugin_impl::type_real,
                              specfem::fk_plugin::type_real>));
}
