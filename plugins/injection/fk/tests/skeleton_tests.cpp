#include "specfem/fk_plugin.hpp"
#include "specfem/fk_plugin/impl/kokkos_lifetime.hpp"

#include <Kokkos_Core.hpp>
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

// The runner initialized Kokkos, so the guards must leave it running.
TEST(FkPluginSkeleton, KokkosLifetimeKeepsForeignKokkos) {
  ASSERT_TRUE(Kokkos::is_initialized());
  {
    const specfem::fk_plugin_impl::KokkosLifetime first;
    const specfem::fk_plugin_impl::KokkosLifetime second;
  }
  EXPECT_TRUE(Kokkos::is_initialized());
  EXPECT_FALSE(Kokkos::is_finalized());
}
