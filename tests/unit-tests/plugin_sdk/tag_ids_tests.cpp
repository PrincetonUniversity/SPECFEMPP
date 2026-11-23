// The injection ABI freezes the SPECFEM++ medium and property tag identifiers.
// These checks fail to compile if the two ever drift apart.

#include "specfem/macros/material_iterators.hpp"
#include "specfem/plugin/injection_abi.h"
#include <boost/preprocessor.hpp>
#include <gtest/gtest.h>

namespace specfem::plugin_sdk_test {

#define SPECFEM_TEST_TAG_ID(TAG) BOOST_PP_TUPLE_ELEM(0, TAG)
#define SPECFEM_TEST_TAG_ENUM(TAG) BOOST_PP_TUPLE_ELEM(1, TAG)

#define SPECFEM_TEST_CHECK_TAG(ABI_VALUE, TAG)                                 \
  static_assert((ABI_VALUE) == SPECFEM_TEST_TAG_ID(TAG),                       \
                #ABI_VALUE " differs from the id of " #TAG);                   \
  static_assert((ABI_VALUE) == static_cast<int>(SPECFEM_TEST_TAG_ENUM(TAG)),   \
                #ABI_VALUE " differs from the enum value of " #TAG)

SPECFEM_TEST_CHECK_TAG(specfem_medium_acoustic, MEDIUM_TAG_ACOUSTIC);
SPECFEM_TEST_CHECK_TAG(specfem_medium_elastic, MEDIUM_TAG_ELASTIC);
SPECFEM_TEST_CHECK_TAG(specfem_medium_elastic_spin, MEDIUM_TAG_ELASTIC_SPIN);
SPECFEM_TEST_CHECK_TAG(specfem_property_isotropic, PROPERTY_TAG_ISOTROPIC);
SPECFEM_TEST_CHECK_TAG(specfem_property_anisotropic, PROPERTY_TAG_ANISOTROPIC);
SPECFEM_TEST_CHECK_TAG(specfem_property_isotropic_cosserat,
                       PROPERTY_TAG_ISOTROPIC_COSSERAT);

#undef SPECFEM_TEST_CHECK_TAG
#undef SPECFEM_TEST_TAG_ENUM
#undef SPECFEM_TEST_TAG_ID

// The checks above are compile-time; this test records that they ran.
TEST(PluginSdkTagIds, MatchSpecfemTags) { SUCCEED(); }

} // namespace specfem::plugin_sdk_test
