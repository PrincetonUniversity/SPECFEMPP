// The injection ABI freezes the SPECFEM++ medium and property tag identifiers.
// These checks fail to compile if the two ever drift apart.

#include "specfem/macros/material_iterators.hpp"
#include "specfem/plugin/injection_abi.h"
#include <boost/preprocessor.hpp>
#include <gtest/gtest.h>

namespace specfem::plugin_sdk_test {

#define SPECFEMPP_TEST_TAG_ID(TAG) BOOST_PP_TUPLE_ELEM(0, TAG)
#define SPECFEMPP_TEST_TAG_ENUM(TAG) BOOST_PP_TUPLE_ELEM(1, TAG)

#define SPECFEMPP_TEST_CHECK_TAG(ABI_VALUE, TAG)                               \
  static_assert((ABI_VALUE) == SPECFEMPP_TEST_TAG_ID(TAG),                     \
                #ABI_VALUE " differs from the id of " #TAG);                   \
  static_assert((ABI_VALUE) == static_cast<int>(SPECFEMPP_TEST_TAG_ENUM(TAG)), \
                #ABI_VALUE " differs from the enum value of " #TAG)

SPECFEMPP_TEST_CHECK_TAG(specfempp_medium_acoustic, MEDIUM_TAG_ACOUSTIC);
SPECFEMPP_TEST_CHECK_TAG(specfempp_medium_elastic, MEDIUM_TAG_ELASTIC);
SPECFEMPP_TEST_CHECK_TAG(specfempp_medium_elastic_spin,
                         MEDIUM_TAG_ELASTIC_SPIN);
SPECFEMPP_TEST_CHECK_TAG(specfempp_property_isotropic, PROPERTY_TAG_ISOTROPIC);
SPECFEMPP_TEST_CHECK_TAG(specfempp_property_anisotropic,
                         PROPERTY_TAG_ANISOTROPIC);
SPECFEMPP_TEST_CHECK_TAG(specfempp_property_isotropic_cosserat,
                         PROPERTY_TAG_ISOTROPIC_COSSERAT);

#undef SPECFEMPP_TEST_CHECK_TAG
#undef SPECFEMPP_TEST_TAG_ENUM
#undef SPECFEMPP_TEST_TAG_ID

// The checks above are compile-time; this test records that they ran.
TEST(PluginSdkTagIds, MatchSpecfemTags) { SUCCEED(); }

} // namespace specfem::plugin_sdk_test
