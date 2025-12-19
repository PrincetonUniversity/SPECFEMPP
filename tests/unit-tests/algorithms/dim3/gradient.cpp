/**
 * @file gradient.cpp
 * @brief Unit tests for 3D spectral element gradient computation algorithms.
 *
 * This file implements comprehensive tests for validating the 3D gradient
 * computation in spectral element methods. The Jacobian matrix represents
 * the transformation between reference coordinates \f$(\xi, \eta, \gamma)\f$
 * and physical coordinates \f$(x, y, z)\f$ for hexahedral spectral elements.
 *
 * The test framework uses a tag-dispatch pattern to enable extensible test
 * scenarios through initializer overloading. The tests verify:
 * - Coordinate transformation between reference \f$[-1,1]^3\f$ and physical
 * space
 * - Gradient computation using equation 29 from Komatitsch and Tromp (1999)
 * - Consistency across multi-element meshes
 * - Validation against analytical solutions for manufactured test cases
 *
 * @note To add new test cases, follow the pattern:
 * 1. Define new initializer tags in appropriate namespaces
 * 2. Overload corresponding init_* functions
 * 3. Add tuple combinations to GradientTestTypes
 * 4. Optionally provide specialized gradient reference implementations
 *
 * @see specfem::algorithms::gradient
 * @see specfem::algorithms::impl::element_gradient for 3D implementation
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>
#include <vector>

#include "SPECFEM_Environment.hpp"
#include "algorithms/gradient.hpp"
#include "datatypes/point_view.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_each_level.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"

namespace specfem::algorithms_test {

constexpr static int ngll =
    5; ///< Number of GLL points per dimension (5th order spectral elements)

/**
 * @brief Initializer tags for scalar field test data generation.
 */
namespace FunctionInitializer3D {
struct ZERO {};        ///< Zero-valued field: f(x,y,z) = 0 everywhere
struct UNIFORM {};     ///< Uniform field: f(x,y,z) = 1 everywhere
struct RANDOM {};      ///< Pseudo-random field values for stochastic testing
struct TWO_ELEMENT {}; ///< Multi-element field with element-dependent values
} // namespace FunctionInitializer3D

/**
 * @brief Initializer tags for quadrature derivative matrix configuration.
 *
 * These tags control the initialization of the derivative matrix used in
 * gradient computation. The matrix represents \f$\frac{\partial L_j}{\partial
 * \xi_i}\f$ where \f$L_j\f$ are Lagrange basis functions.
 */
namespace QuadratureInitializer3D {
struct IDENTITY {}; ///< Identity matrix for simplified analytical verification
struct LAGRANGE {}; ///< Proper GLL Lagrange derivative matrix from quadrature
} // namespace QuadratureInitializer3D

/**
 * @brief Initializer tags for Jacobian matrix test configurations.
 *
 * These tags define different geometric transformation scenarios for testing
 * coordinate mapping between reference element \f$[-1,1]^3\f$ and physical
 * space.
 */
namespace JacobianInitializer3D {
struct IDENTITY {};    ///< Identity Jacobian (no geometric transformation)
struct TWO_ELEMENT {}; ///< Multi-element mesh with scaled transformations
} // namespace JacobianInitializer3D

struct lagrange_derivative3D {
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using view_type = Kokkos::View<type_real[5][5], memory_space>;
  view_type xi;
  view_type eta;
  view_type gamma;

  KOKKOS_FUNCTION lagrange_derivative3D(const view_type &view)
      : xi(view), eta(view), gamma(view) {}
};

/**
 * @brief Initialize identity Jacobian matrix for single-element test cases.
 *
 * Creates a single spectral element with identity Jacobian transformation at
 * all GLL points. This represents a reference element with no geometric
 * deformation, where the partial derivatives form identity relationships:
 * \f$\frac{\partial \xi}{\partial x} = \frac{\partial \eta}{\partial y} =
 * \frac{\partial \gamma}{\partial z} = 1\f$ and all cross terms are zero.
 *
 * @return Vector containing single element's Jacobian data with identity values
 *
 * @note This configuration allows direct comparison between reference and
 * physical space gradients since the coordinate transformation is trivial.
 */
std::vector<std::array<
    std::array<std::array<std::array<std::array<type_real, 3>, 3>, 5>, 5>, 5> >
init_jacobian(const JacobianInitializer3D::IDENTITY &) {
  std::vector<std::array<
      std::array<std::array<std::array<std::array<type_real, 3>, 3>, 5>, 5>,
      5> >
      jacobian(1);
  for (int iz = 0; iz < 5; ++iz) {
    for (int iy = 0; iy < 5; ++iy) {
      for (int ix = 0; ix < 5; ++ix) {
        // Initialize identity Jacobian matrix
        jacobian[0][iz][iy][ix][0][0] = 1.0; // ξₓ = ∂ξ/∂x
        jacobian[0][iz][iy][ix][0][1] = 0.0; // ξᵧ = ∂ξ/∂y
        jacobian[0][iz][iy][ix][0][2] = 0.0; // ξᵤ = ∂ξ/∂z
        jacobian[0][iz][iy][ix][1][0] = 0.0; // ηₓ = ∂η/∂x
        jacobian[0][iz][iy][ix][1][1] = 1.0; // ηᵧ = ∂η/∂y
        jacobian[0][iz][iy][ix][1][2] = 0.0; // ηᵤ = ∂η/∂z
        jacobian[0][iz][iy][ix][2][0] = 0.0; // γₓ = ∂γ/∂x
        jacobian[0][iz][iy][ix][2][1] = 0.0; // γᵧ = ∂γ/∂y
        jacobian[0][iz][iy][ix][2][2] = 1.0; // γᵤ = ∂γ/∂z
      }
    }
  }
  return jacobian;
}

/**
 * @brief Initialize two-element Jacobian dataset with scaled transformations.
 *
 * Creates a two-element mesh where each element has scaled identity Jacobian
 * matrices. Element 0 has identity scaling (1.0), while element 1 has 2.0
 * scaling, effectively creating elements of different sizes.
 *
 * @return Vector containing Jacobian data for both elements with scaling
 * factors
 *
 * @note This tests multi-element gradient computation and coordinate
 *       transformation consistency across element boundaries.
 */
std::vector<std::array<
    std::array<std::array<std::array<std::array<type_real, 3>, 3>, 5>, 5>, 5> >
init_jacobian(const JacobianInitializer3D::TWO_ELEMENT &) {
  std::vector<std::array<
      std::array<std::array<std::array<std::array<type_real, 3>, 3>, 5>, 5>,
      5> >
      jacobian(2);
  for (int ielement = 0; ielement < 2; ++ielement) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int iy = 0; iy < 5; ++iy) {
        for (int ix = 0; ix < 5; ++ix) {
          const type_real scale = 1.0 + ielement; // Element scaling factor
          jacobian[ielement][iz][iy][ix][0][0] = scale; // Scaled ξₓ
          jacobian[ielement][iz][iy][ix][0][1] = 0.0;   // ξᵧ = 0
          jacobian[ielement][iz][iy][ix][0][2] = 0.0;   // ξᵤ = 0
          jacobian[ielement][iz][iy][ix][1][0] = 0.0;   // ηₓ = 0
          jacobian[ielement][iz][iy][ix][1][1] = scale; // Scaled ηᵧ
          jacobian[ielement][iz][iy][ix][1][2] = 0.0;   // ηᵤ = 0
          jacobian[ielement][iz][iy][ix][2][0] = 0.0;   // γₓ = 0
          jacobian[ielement][iz][iy][ix][2][1] = 0.0;   // γᵧ = 0
          jacobian[ielement][iz][iy][ix][2][2] = scale; // Scaled γᵤ
        }
      }
    }
  }
  return jacobian;
}

/**
 * @brief Helper wrapper that materializes jacobian_matrix containers for tests.
 *
 * This class provides a type-safe interface for creating and accessing Jacobian
 * matrix data using compile-time dispatch based on initializer tags. It manages
 * the storage and conversion between test data formats and SPECFEM++ assembly
 * data structures.
 *
 * @tparam Initializer Tag describing which init_jacobian overload to invoke
 */
template <typename Initializer> struct JacobianMatrix3D {
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim3;

private:
  std::vector<std::array<
      std::array<std::array<std::array<std::array<type_real, 3>, 3>, 5>, 5>,
      5> >
      _jacobian;

public:
  /**
   * @brief Construct Jacobian data using the supplied initializer tag.
   *
   * @param initializer Tag-dispatch parameter selecting initialization strategy
   */
  JacobianMatrix3D(const Initializer &initializer)
      : _jacobian(init_jacobian(initializer)) {}

  /**
   * @brief Accessor for individual Jacobian matrix components.
   *
   * @param ielement Element index
   * @param iz GLL point index in z direction
   * @param iy GLL point index in y direction
   * @param ix GLL point index in x direction
   * @param idim First tensor index (0=ξ, 1=η, 2=γ)
   * @param jdim Second tensor index (0=x, 1=y, 2=z)
   * @return Reference to Jacobian component J[idim][jdim]
   */
  const type_real &operator()(const int ielement, const int iz, const int iy,
                              const int ix, const int idim,
                              const int jdim) const {
    return _jacobian[ielement][iz][iy][ix][idim][jdim];
  }

  /**
   * @brief Get the number of spectral elements in this dataset.
   *
   * @return Number of elements
   */
  const int n_elements() const { return static_cast<int>(_jacobian.size()); }

  /**
   * @brief Convert to specfem::assembly::jacobian_matrix format.
   *
   * This method creates a properly formatted jacobian_matrix object that can
   * be used with the production gradient algorithms. The conversion maps the
   * internal storage to the xix, xiy, xiz, etax, etay, etaz, gammax, gammay,
   * gammaz components expected by the assembly framework.
   *
   * @return Assembled jacobian_matrix ready for algorithm testing
   */
  specfem::assembly::jacobian_matrix<dimension_tag> jacobian() const {
    specfem::assembly::jacobian_matrix<dimension_tag> jacobian_matrix(
        n_elements(), 5, 5, 5);

    for (int ielement = 0; ielement < n_elements(); ++ielement) {
      for (int iz = 0; iz < 5; ++iz) {
        for (int iy = 0; iy < 5; ++iy) {
          for (int ix = 0; ix < 5; ++ix) {
            jacobian_matrix.h_xix(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][0][0]; // ∂ξ/∂x
            jacobian_matrix.h_xiy(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][0][1]; // ∂ξ/∂y
            jacobian_matrix.h_xiz(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][0][2]; // ∂ξ/∂z
            jacobian_matrix.h_etax(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][1][0]; // ∂η/∂x
            jacobian_matrix.h_etay(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][1][1]; // ∂η/∂y
            jacobian_matrix.h_etaz(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][1][2]; // ∂η/∂z
            jacobian_matrix.h_gammax(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][2][0]; // ∂γ/∂x
            jacobian_matrix.h_gammay(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][2][1]; // ∂γ/∂y
            jacobian_matrix.h_gammaz(ielement, iz, iy, ix) =
                _jacobian[ielement][iz][iy][ix][2][2]; // ∂γ/∂z
          }
        }
      }
    }

    jacobian_matrix.sync_views();
    return jacobian_matrix;
  }
};

/**
 * @brief Build identity derivative matrix for simplified testing.
 *
 * Creates an identity matrix that simplifies gradient computation validation
 * by eliminating the complexity of proper Lagrange derivative evaluation.
 * Useful for analytical verification of gradient transformation logic.
 *
 * @return 5×5 identity matrix for quadrature derivative operations
 */
std::array<std::array<type_real, 5>, 5>
init_quadrature(const QuadratureInitializer3D::IDENTITY &) {
  std::array<std::array<type_real, 5>, 5> quadrature{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      quadrature[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
  return quadrature;
}

/**
 * @brief Build proper GLL Lagrange derivative matrix.
 *
 * Constructs the actual derivative matrix used in spectral element methods,
 * \f$h'_{ij} = \frac{dL_j}{d\xi}\Big|_{\xi_i}\f$, where \f$L_j(\xi)\f$ are
 * the Lagrange interpolating polynomials and \f$\xi_i\f$ are GLL quadrature
 * points.
 *
 * @return 5×5 GLL derivative matrix for realistic gradient computation
 */
std::array<std::array<type_real, 5>, 5>
init_quadrature(const QuadratureInitializer3D::LAGRANGE &) {
  std::array<std::array<type_real, 5>, 5> quadrature{};

  const specfem::quadrature::gll::gll gll{};
  const auto hprime = gll.get_hhprime();
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      quadrature[j][i] = hprime(i, j); // Transpose for proper indexing
    }
  }
  return quadrature;
}

/**
 * @brief Helper owning quadrature derivative matrix for gradient evaluation.
 *
 * This class manages derivative matrices used in gradient computation,
 * providing type-safe access to different quadrature configurations through
 * compile-time dispatch.
 *
 * @tparam Initializer Tag selecting which init_quadrature overload to use
 */
template <typename Initializer> struct Quadrature3D {
  std::array<std::array<type_real, 5>, 5> _quadrature;

  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  using view_type = Kokkos::View<type_real[5][5], memory_space>;

  /**
   * @brief Construct derivative matrix using the selected initializer tag.
   *
   * @param initializer Tag-dispatch parameter for quadrature initialization
   */
  Quadrature3D(const Initializer &initializer)
      : _quadrature(init_quadrature(initializer)) {}

  lagrange_derivative3D quadrature() const {
    view_type::HostMirror quadrature_view("quadrature_view");
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        quadrature_view(i, j) = _quadrature[i][j];
      }
    }

    const auto d_quadrature =
        Kokkos::create_mirror_view_and_copy(memory_space(), quadrature_view);

    const lagrange_derivative3D lagrange_deriv(d_quadrature);
    return lagrange_deriv;
  }

  /**
   * @brief Access derivative matrix entry.
   *
   * @param i Row index (output GLL point)
   * @param j Column index (input GLL point)
   * @return Derivative value \f$\frac{dL_j}{d\xi}\Big|_{\xi_i}\f$
   */
  type_real operator()(const int &i, const int &j) const {
    return _quadrature[i][j];
  }
};

/**
 * @brief Generate zero-valued scalar field for manufactured solution tests.
 *
 * Creates a field where f(ξ,η,γ) = 0 at all GLL points. This provides a trivial
 * case where the exact gradient is known analytically: ∇f = (0,0,0) everywhere.
 *
 * @return Single-element vector containing zero field data
 */
auto init_function(const FunctionInitializer3D::ZERO &) {
  std::vector<std::array<std::array<std::array<type_real, ngll>, ngll>, ngll> >
      _f(1);
  for (int icomp = 0; icomp < 1; ++icomp) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int iy = 0; iy < ngll; ++iy) {
        for (int ix = 0; ix < ngll; ++ix) {
          _f[icomp][iz][iy][ix] = 0.0;
        }
      }
    }
  }
  return _f;
}

/**
 * @brief Generate uniform scalar field for manufactured solution tests.
 *
 * Creates a constant field where f(ξ,η,γ) = 1 at all GLL points. For identity
 * Jacobian and identity quadrature, this yields known gradient behavior that
 * can be verified analytically.
 *
 * @return Single-element vector containing uniform field data
 */
auto init_function(const FunctionInitializer3D::UNIFORM &) {
  std::vector<std::array<std::array<std::array<type_real, 5>, 5>, 5> > _f(1);
  for (int icomp = 0; icomp < 1; ++icomp) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int iy = 0; iy < 5; ++iy) {
        for (int ix = 0; ix < 5; ++ix) {
          _f[icomp][iz][iy][ix] = 1.0;
        }
      }
    }
  }
  return _f;
}

/**
 * @brief Generate pseudo-random scalar field for stochastic validation.
 *
 * Creates a field with random values between [0,1] to test gradient computation
 * with non-trivial function values. Helps verify numerical stability and
 * algorithmic correctness for arbitrary field configurations.
 *
 * @return Single-element vector containing random field data
 *
 * @warning Uses rand() for simplicity; not cryptographically secure
 */
auto init_function(const FunctionInitializer3D::RANDOM &) {
  std::vector<std::array<std::array<std::array<type_real, 5>, 5>, 5> > _f(1);
  for (int icomp = 0; icomp < 1; ++icomp) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int iy = 0; iy < 5; ++iy) {
        for (int ix = 0; ix < 5; ++ix) {
          _f[icomp][iz][iy][ix] =
              static_cast<type_real>(rand()) / static_cast<type_real>(RAND_MAX);
        }
      }
    }
  }
  return _f;
}

/**
 * @brief Generate two-element scalar field with element-wise constant values.
 *
 * Creates a multi-element field where element 0 has f = 1.0 and element 1 has
 * f = 2.0. This tests gradient computation consistency across element
 * boundaries and validates multi-element assembly pathways.
 *
 * @return Two-element vector containing element-dependent constant field data
 */
auto init_function(const FunctionInitializer3D::TWO_ELEMENT &) {
  std::vector<std::array<std::array<std::array<type_real, 5>, 5>, 5> > _f(2);
  for (int ielement = 0; ielement < 2; ++ielement) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int iy = 0; iy < 5; ++iy) {
        for (int ix = 0; ix < 5; ++ix) {
          _f[ielement][iz][iy][ix] = 1.0; // Element-dependent constant
        }
      }
    }
  }
  return _f;
}

/**
 * @brief Helper owning scalar test data and exposing Kokkos views for gradient
 * calls.
 *
 * This class manages scalar field data for test cases, providing conversion
 * between test-friendly storage formats and the Kokkos Views required by
 * the production gradient algorithms.
 *
 * @tparam Initializer Tag selecting which init_function overload to invoke
 */
template <typename Initializer> struct Function3D {
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim3;
  constexpr static int components = 1; ///< Scalar field (single component)

  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using memory_traits = Kokkos::MemoryTraits<>;

  using view_type = Kokkos::View<type_real *[ngll][ngll][ngll][components],
                                 memory_space, memory_traits>;

  /**
   * @brief Construct scalar field using the supplied initializer tag.
   *
   * @param initializer Tag-dispatch parameter for field initialization
   */
  Function3D(const Initializer &initializer) : _f(init_function(initializer)) {}

  /**
   * @brief Element-wise access to scalar field values.
   *
   * @param ielement Element index
   * @param iz GLL point index in z direction
   * @param iy GLL point index in y direction
   * @param ix GLL point index in x direction
   * @return Field value at specified location
   */
  const type_real &operator()(const int &ielement, const int &iz, const int &iy,
                              const int &ix) const {
    return _f[ielement][iz][iy][ix];
  }

  /**
   * @brief Get the number of spectral elements in this field.
   *
   * @return Number of elements
   */
  int n_elements() const { return static_cast<int>(_f.size()); }

  /**
   * @brief Convert to Kokkos View format for gradient algorithm interface.
   *
   * Creates a Kokkos View that can be passed to specfem::algorithms::gradient()
   * and related chunk-based execution frameworks.
   *
   * @return Kokkos View containing field data in algorithm-compatible format
   */
  view_type view() const {
    view_type::HostMirror f_view("f_view", _f.size());
    for (int ielement = 0; ielement < static_cast<int>(_f.size()); ++ielement) {
      for (int iz = 0; iz < ngll; ++iz) {
        for (int iy = 0; iy < ngll; ++iy) {
          for (int ix = 0; ix < ngll; ++ix) {
            f_view(ielement, iz, iy, ix, 0) = _f[ielement][iz][iy][ix];
          }
        }
      }
    }

    const auto f_device =
        Kokkos::create_mirror_view_and_copy(memory_space(), f_view);

    return f_device;
  }

private:
  std::vector<std::array<std::array<std::array<type_real, ngll>, ngll>, ngll> >
      _f; ///< Internal
          ///< field
          ///< storage
};

/**
 * @brief Analytical gradient reference for zero field with identity
 * configuration.
 *
 * For f(x,y,z) = 0 everywhere, the exact gradient is ∇f = (0,0,0) at all
 * points. This provides a trivial but important validation case for algorithm
 * correctness.
 *
 * @return Gradient field with zero values at all GLL points
 */
std::vector<std::array<
    std::array<std::array<std::array<std::array<type_real, 3>, 1>, 5>, 5>, 5> >
gradient(const JacobianMatrix3D<JacobianInitializer3D::IDENTITY> &jacobian,
         const Quadrature3D<QuadratureInitializer3D::IDENTITY> &quadrature,
         const Function3D<FunctionInitializer3D::ZERO> &) {
  std::vector<std::array<
      std::array<std::array<std::array<std::array<type_real, 3>, 1>, 5>, 5>,
      5> >
      grad(1);
  for (int iz = 0; iz < 5; ++iz) {
    for (int iy = 0; iy < 5; ++iy) {
      for (int ix = 0; ix < 5; ++ix) {
        grad[0][iz][iy][ix][0][0] = 0.0; // ∂f/∂x = 0
        grad[0][iz][iy][ix][0][1] = 0.0; // ∂f/∂y = 0
        grad[0][iz][iy][ix][0][2] = 0.0; // ∂f/∂z = 0
      }
    }
  }
  return grad;
}

/**
 * @brief Analytical gradient reference for uniform field with identity
 * configuration.
 *
 * For f(x,y,z) = 1 (constant) with identity Jacobian and identity quadrature,
 * the expected behavior depends on the specific derivative implementation.
 * With identity quadrature, uniform functions produce unit gradients.
 *
 * @return Gradient field with unit values reflecting derivative matrix behavior
 */
std::vector<std::array<
    std::array<std::array<std::array<std::array<type_real, 3>, 1>, 5>, 5>, 5> >
gradient(const JacobianMatrix3D<JacobianInitializer3D::IDENTITY> &jacobian,
         const Quadrature3D<QuadratureInitializer3D::IDENTITY> &quadrature,
         const Function3D<FunctionInitializer3D::UNIFORM> &) {
  std::vector<std::array<
      std::array<std::array<std::array<std::array<type_real, 3>, 1>, 5>, 5>,
      5> >
      grad(1);
  for (int iz = 0; iz < 5; ++iz) {
    for (int iy = 0; iy < 5; ++iy) {
      for (int ix = 0; ix < 5; ++ix) {
        grad[0][iz][iy][ix][0][0] = 1.0; // Identity quadrature effect
        grad[0][iz][iy][ix][0][1] = 1.0; // Identity quadrature effect
        grad[0][iz][iy][ix][0][2] = 1.0; // Identity quadrature effect
      }
    }
  }
  return grad;
}

/**
 * @brief Generic reference gradient calculator mirroring
 * specfem::algorithms::impl::element_gradient() for 3D.
 *
 * This function implements the reference calculation following the 3D spectral
 * element gradient formulation. It computes the gradient by:
 * 1. Computing reference coordinate derivatives using the quadrature matrix
 * 2. Transforming to physical coordinates using the Jacobian matrix
 *
 * @tparam Jacobian JacobianMatrix3D type providing coordinate transformation
 * data
 * @tparam Quadrature3D Quadrature3D type providing derivative matrix
 * @tparam Function3D Function3D type providing scalar field data
 * @param jacobian Jacobian transformation data
 * @param quadrature Derivative matrix for reference coordinate differentiation
 * @param function Scalar field to differentiate
 * @return Computed gradient at all GLL points in all elements
 */
template <typename Jacobian, typename Quadrature3D, typename Function3D>
std::vector<std::array<
    std::array<std::array<std::array<std::array<type_real, 3>, 1>, 5>, 5>, 5> >
gradient(const Jacobian &jacobian, const Quadrature3D &quadrature,
         const Function3D &function) {

  std::vector<std::array<
      std::array<std::array<std::array<std::array<type_real, 3>, 1>, 5>, 5>,
      5> >
      grad(function.n_elements());

  for (int ielement = 0; ielement < function.n_elements(); ++ielement) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int iy = 0; iy < ngll; ++iy) {
        for (int ix = 0; ix < ngll; ++ix) {
          // Compute reference coordinate derivatives using quadrature matrix
          type_real df_dxi = 0.0;    // ∂f/∂ξ
          type_real df_deta = 0.0;   // ∂f/∂η
          type_real df_dgamma = 0.0; // ∂f/∂γ

          // Sum over GLL points using derivative matrix
          for (int l = 0; l < ngll; ++l) {
            df_dxi += quadrature(ix, l) * function(ielement, iz, iy, l);
            df_deta += quadrature(iy, l) * function(ielement, iz, l, ix);
            df_dgamma += quadrature(iz, l) * function(ielement, l, iy, ix);
          }

          // Transform to physical coordinates using Jacobian
          grad[ielement][iz][iy][ix][0][0] =                    // ∂f/∂x
              df_dxi * jacobian(ielement, iz, iy, ix, 0, 0) +   // ∂f/∂ξ · ∂ξ/∂x
              df_deta * jacobian(ielement, iz, iy, ix, 1, 0) +  // ∂f/∂η · ∂η/∂x
              df_dgamma * jacobian(ielement, iz, iy, ix, 2, 0); // ∂f/∂γ · ∂γ/∂x

          grad[ielement][iz][iy][ix][0][1] =                    // ∂f/∂y
              df_dxi * jacobian(ielement, iz, iy, ix, 0, 1) +   // ∂f/∂ξ · ∂ξ/∂y
              df_deta * jacobian(ielement, iz, iy, ix, 1, 1) +  // ∂f/∂η · ∂η/∂y
              df_dgamma * jacobian(ielement, iz, iy, ix, 2, 1); // ∂f/∂γ · ∂γ/∂y

          grad[ielement][iz][iy][ix][0][2] =                    // ∂f/∂z
              df_dxi * jacobian(ielement, iz, iy, ix, 0, 2) +   // ∂f/∂ξ · ∂ξ/∂z
              df_deta * jacobian(ielement, iz, iy, ix, 1, 2) +  // ∂f/∂η · ∂η/∂z
              df_dgamma * jacobian(ielement, iz, iy, ix, 2, 2); // ∂f/∂γ · ∂γ/∂z
        }
      }
    }
  }

  return grad;
}

template <typename Jacobian, typename Quadrature3D, typename Function3D>
Kokkos::View<type_real *****, Kokkos::DefaultExecutionSpace>::HostMirror
execute(const Jacobian &jacobian_matrix, const Quadrature3D &quadrature,
        const Function3D &function) {
  // Set up chunked domain iteration for production algorithm testing
  using simd = specfem::datatype::simd<type_real, false>;
  Kokkos::View<int *, Kokkos::HostSpace> h_element_indices(
      "element_indices", function.n_elements());
  for (int i = 0; i < function.n_elements(); ++i) {
    h_element_indices(i) = i;
  }

  const auto element_indices = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace(), h_element_indices);

  using ParallelConfig = specfem::parallel_configuration::chunk_config<
      specfem::dimension::type::dim3, 1, 1, 1, 1, simd,
      Kokkos::DefaultExecutionSpace>;

  const specfem::mesh_entity::element_grid element_grid(ngll, ngll, ngll);

  const specfem::execution::ChunkedDomainIterator chunk(
      ParallelConfig(), element_indices, element_grid);

  const auto jacobian = jacobian_matrix.jacobian();
  const auto function_view = function.view();
  const auto quadrature_view = quadrature.quadrature();

  using Function3DView = specfem::datatype::VectorChunkElementViewType<
      type_real, specfem::dimension::type::dim3, 1, ngll, 1, false,
      Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<> >;

  Kokkos::View<type_real *****, Kokkos::DefaultExecutionSpace> gradient_view(
      "gradient_view", function.n_elements(), ngll, ngll, ngll, 3);

  // Execute production gradient algorithm and validate results
  specfem::execution::for_each_level(
      chunk, KOKKOS_LAMBDA(const typename decltype(chunk)::index_type &index) {
        const Function3DView f(Kokkos::subview(function_view, index.get_range(),
                                               Kokkos::ALL(), Kokkos::ALL(),
                                               Kokkos::ALL(), Kokkos::ALL()));

        // Call the production gradient algorithm
        specfem::algorithms::gradient(
            index, jacobian, quadrature_view, f,
            [&](const auto &iterator_index, const auto &df) {
              const auto local_index = iterator_index.get_index();
              const int ispec = local_index.ispec;
              const int iz = local_index.iz;
              const int iy = local_index.iy;
              const int ix = local_index.ix;

              // Store computed gradient
              gradient_view(ispec, iz, iy, ix, 0) = df(0, 0); // ∂f/∂x
              gradient_view(ispec, iz, iy, ix, 1) = df(0, 1); // ∂f/∂y
              gradient_view(ispec, iz, iy, ix, 2) = df(0, 2); // ∂f/∂z
            });
      });

  const auto h_gradient_view = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultHostExecutionSpace(), gradient_view);

  return h_gradient_view;
}

} // namespace specfem::algorithms_test

using namespace specfem::algorithms_test;

/**
 * @brief GoogleTest fixture bundling manufactured data and Jacobian views.
 *
 * This fixture provides a type-safe framework for testing gradient computation
 * with different combinations of Jacobian matrices, quadrature configurations,
 * and scalar field data. Each test instantiation receives a unique combination
 * of initializer tags that determine the specific test scenario.
 *
 * @tparam TestingType Tuple of (JacobianInitializer3D, QuadratureInitializer3D,
 * FunctionInitializer3D)
 */
template <typename TestingType>
struct GradientTestFixture3D : public ::testing::Test {
  using JacobianInitializer3D =
      std::tuple_element_t<0, TestingType>; ///< Jacobian initialization
                                            ///< strategy
  using QuadratureInitializer3D =
      std::tuple_element_t<1, TestingType>; ///< Quadrature3D initialization
                                            ///< strategy
  using FunctionInitializer3D =
      std::tuple_element_t<2, TestingType>; ///< Function3D initialization
                                            ///< strategy

  /**
   * @brief Construct test helpers using tag dispatch for each component.
   *
   * This constructor uses the compile-time type information to instantiate
   * the appropriate helper objects for the specific test combination.
   */
  GradientTestFixture3D()
      : jacobian_matrix(JacobianInitializer3D{}),
        quadrature(QuadratureInitializer3D{}),
        function(FunctionInitializer3D{}) {}

  JacobianMatrix3D<JacobianInitializer3D> jacobian_matrix; ///< Coordinate
                                                           ///< transformation
                                                           ///< data
  Quadrature3D<QuadratureInitializer3D> quadrature; ///< Derivative matrix for
                                                    ///< gradient computation
  Function3D<FunctionInitializer3D> function;       ///< Scalar field test data
};

/**
 * @brief Test case combinations for parameterized gradient validation.
 *
 * This type list defines all the test scenarios that will be instantiated by
 * GoogleTest's parameterized testing framework. Each tuple represents a unique
 * combination of initialization strategies that exercises different aspects of
 * the gradient computation algorithm.
 *
 * **Current test cases:**
 * 1. **Identity+Zero**: Basic correctness test with trivial inputs
 * 2. **Identity+Uniform**: Simple analytical validation case
 * 3. **Lagrange+Random**: Realistic quadrature with arbitrary field data
 * 4. **TwoElement+TwoElement**: Multi-element consistency validation
 *
 * **Adding new test cases:**
 * To extend the test coverage, follow this process:
 *
 * 1. **Define new initializer tags** in the appropriate namespace:
 *    ```cpp
 *    namespace JacobianInitializer3D {
 *    struct SCALED_ROTATION {}; // New Jacobian type
 *    }
 *    ```
 *
 * 2. **Overload the corresponding init_* function**:
 *    ```cpp
 *    std::vector<...> init_jacobian(const
 * JacobianInitializer3D::SCALED_ROTATION &) {
 *        // Implementation for rotated/scaled Jacobian
 *    }
 *    ```
 *
 * 3. **Add new tuple to GradientTestTypes**:
 *    ```cpp
 *    std::tuple<JacobianInitializer3D::SCALED_ROTATION,
 *               QuadratureInitializer3D::LAGRANGE,
 *               FunctionInitializer3D::POLYNOMIAL>
 *    ```
 *
 * 4. **Optionally add specialized gradient() overload** for analytical
 * reference:
 *    ```cpp
 *    std::vector<...> gradient(
 *        const JacobianMatrix3D<JacobianInitializer3D::SCALED_ROTATION>
 * &jacobian, const Quadrature3D<QuadratureInitializer3D::LAGRANGE> &quadrature,
 *        const Function3D<FunctionInitializer3D::POLYNOMIAL> &function) {
 *        // Analytical reference implementation
 *    }
 *    ```
 *
 * The test framework will automatically instantiate and execute the new
 * scenario.
 */
using GradientTestTypes = ::testing::Types<
    std::tuple<JacobianInitializer3D::IDENTITY,
               QuadratureInitializer3D::IDENTITY, FunctionInitializer3D::ZERO>,
    std::tuple<JacobianInitializer3D::IDENTITY,
               QuadratureInitializer3D::IDENTITY,
               FunctionInitializer3D::UNIFORM>,
    std::tuple<JacobianInitializer3D::IDENTITY,
               QuadratureInitializer3D::LAGRANGE,
               FunctionInitializer3D::RANDOM>,
    std::tuple<JacobianInitializer3D::TWO_ELEMENT,
               QuadratureInitializer3D::IDENTITY,
               FunctionInitializer3D::TWO_ELEMENT> >;

/**
 * @brief Instantiate parameterized test suite for all declared tuple
 * combinations.
 */
TYPED_TEST_SUITE(GradientTestFixture3D, GradientTestTypes);

/**
 * @brief Validate specfem::algorithms::impl::element_gradient() against
 * manufactured solutions.
 *
 * This test performs comprehensive validation of the 3D gradient computation
 * by:
 * 1. Computing expected gradients using the reference implementation
 * 2. Setting up chunked domain iteration matching production usage
 * 3. Calling the production element_gradient algorithm with test data
 * 4. Comparing computed vs. expected results at each GLL point
 *
 * The test validates both the mathematical accuracy and the execution pathway
 * of the gradient algorithm, ensuring consistency between reference and
 * production implementations across different geometric and field
 * configurations.
 *
 * @note This test directly calls the element_gradient implementation since
 * there's no top-level 3D gradient wrapper function currently available.
 */
TYPED_TEST(GradientTestFixture3D, TestGradientComputation) {
  // Verify consistency between function and Jacobian element counts
  ASSERT_TRUE(this->function.n_elements() == this->jacobian_matrix.n_elements())
      << "Mismatch in number of elements between function and jacobian matrix";

  // Compute expected results using reference implementation
  const auto expected_gradient = specfem::algorithms_test::gradient(
      this->jacobian_matrix, this->quadrature, this->function);

  const auto computed_gradient = specfem::algorithms_test::execute(
      this->jacobian_matrix, this->quadrature, this->function);

  // Validate computed results against expected values
  for (int ielement = 0; ielement < this->function.n_elements(); ++ielement) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int iy = 0; iy < ngll; ++iy) {
        for (int ix = 0; ix < ngll; ++ix) {
          const auto e = expected_gradient[ielement][iz][iy][ix];
          const auto df = Kokkos::subview(computed_gradient, ielement, iz, iy,
                                          ix, Kokkos::ALL());

          // Compare each gradient component with tolerance
          if (!specfem::utilities::is_close(df(0), e[0][0]) ||
              !specfem::utilities::is_close(df(1), e[0][1]) ||
              !specfem::utilities::is_close(df(2), e[0][2])) {
            ADD_FAILURE() << "Gradient mismatch at element " << ielement
                          << ", GLL point (" << ix << "," << iy << "," << iz
                          << "): Expected (" << e[0][0] << ", " << e[0][1]
                          << ", " << e[0][2] << "), but got (" << df(0) << ", "
                          << df(1) << ", " << df(2) << ")";
          }
        }
      }
    }
  }

  SUCCEED() << "Gradient computation validation passed for all test points.";
}
