#include "test_fixture.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace compute_derivatives_test_impl {

using Dimension = specfem::element::dimension_tag;
using Medium = specfem::element::medium_tag;
using Property = specfem::element::property_tag;
using Attenuation = specfem::element::attenuation_tag;

template <Dimension DimensionTag, Medium MediumTag,
          Property PropertyTag = Property::isotropic,
          Attenuation AttenuationTag = Attenuation::none>
using KernelCase = specfem::tags::Tags<DimensionTag, MediumTag, PropertyTag,
                                       AttenuationTag, false>;

// SH anisotropic derivatives are an aborting placeholder, not a supported case.
using KernelCases = ::testing::Types<
    KernelCase<Dimension::dim2, Medium::acoustic>,
    KernelCase<Dimension::dim2, Medium::poroelastic>,
    KernelCase<Dimension::dim2, Medium::elastic_psv>,
    KernelCase<Dimension::dim2, Medium::elastic_psv, Property::anisotropic>,
    KernelCase<Dimension::dim2, Medium::elastic_psv, Property::isotropic,
               Attenuation::constant_isotropic>,
    KernelCase<Dimension::dim2, Medium::elastic_psv, Property::anisotropic,
               Attenuation::constant_isotropic>,
    KernelCase<Dimension::dim2, Medium::elastic_sh>,
    KernelCase<Dimension::dim3, Medium::acoustic>,
    KernelCase<Dimension::dim3, Medium::elastic>,
    KernelCase<Dimension::dim3, Medium::elastic, Property::anisotropic>,
    KernelCase<Dimension::dim3, Medium::elastic, Property::isotropic,
               Attenuation::constant_isotropic>>;

struct KernelCaseNames {
  template <typename Tags> static std::string GetName(int) {
    return std::string(Tags::dimension_tag == Dimension::dim2 ? "Dim2_"
                                                              : "Dim3_") +
           specfem::element::to_string(Tags::medium_tag, Tags::property_tag) +
           (Tags::attenuation_tag == Attenuation::none ? "_None"
                                                       : "_Attenuating");
  }
};

TYPED_TEST_SUITE(ComputeDerivativesTest, KernelCases, KernelCaseNames);

TYPED_TEST(ComputeDerivativesTest, DispatchesAndAccumulates) {
  using Index = typename TestFixture::Index;
  using Kernels = typename TestFixture::Kernels;

  this->set_fields(3.0, 2.0);
  const type_real expected = this->expected_kernel();
  ASSERT_TRUE(std::isfinite(expected));
  ASSERT_GT(std::abs(expected), 0.0);

  std::vector<Kernels> first_step;
  for (int step = 1; step <= 2; ++step) {
    this->compute();
    int point = 0;
    this->for_each_point([&](const Index &index) {
      SCOPED_TRACE(::testing::Message() << "step=" << step << ", point="
                                        << point << ", ispec=" << index.ispec);
      Kernels kernels;
      specfem::assembly::load_on_host(index, this->assembly.kernels, kernels);
      EXPECT_NEAR(this->selected_kernel(kernels), step * expected,
                  std::abs(expected) * 1.e-5);
      for (int k = 0; k < Kernels::nprops; ++k) {
        EXPECT_TRUE(std::isfinite(kernels[k])) << "kernel=" << k;
        if (step == 2) {
          const type_real twice = 2.0 * first_step[point][k];
          EXPECT_NEAR(kernels[k], twice, std::abs(twice) * 1.e-5)
              << "kernel=" << k;
        }
      }
      if (step == 1) {
        first_step.push_back(kernels);
      }
      ++point;
    });
  }
}

TYPED_TEST(ComputeDerivativesTest, ZeroFieldsProduceZeroKernels) {
  using Index = typename TestFixture::Index;
  using Kernels = typename TestFixture::Kernels;

  this->set_fields(0.0, 0.0);
  this->compute();
  this->for_each_point([&](const Index &index) {
    Kernels kernels;
    specfem::assembly::load_on_host(index, this->assembly.kernels, kernels);
    for (int k = 0; k < Kernels::nprops; ++k) {
      EXPECT_NEAR(kernels[k], 0.0, 1.e-12)
          << "ispec=" << index.ispec << ", kernel=" << k;
    }
  });
}

} // namespace compute_derivatives_test_impl
