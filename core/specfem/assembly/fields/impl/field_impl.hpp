#pragma once

#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Implementation details for spectral element field storage
 *
 * This namespace contains the internal implementation classes and functions
 * for managing simulation fields in SPECFEM++. The classes provide optimized
 * storage layouts and access patterns for different field types (displacement,
 * velocity, acceleration, mass matrix) with support for multiple physical
 * media and spatial dimensions.
 *
 * @note These are implementation details. Use the public interface in
 * specfem::assembly::fields instead.
 */
namespace fields_impl {

/**
 * @brief Base storage class for individual field components in spectral
 * elements
 *
 * This template class provides the foundational storage and access mechanisms
 * for individual field components (displacement, velocity, acceleration, mass
 * matrix) in spectral element simulations. It manages both device and host
 * memory views with efficient synchronization capabilities.
 *
 * The class is designed to handle different physical media and spatial
 * dimensions while maintaining optimal memory layouts for Kokkos-based
 * computations.
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam MediumTag Physical medium type (elastic, acoustic, poroelastic)
 * @tparam DataClass Field component type (displacement, velocity, acceleration,
 * mass_matrix)
 *
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass>
class base_field {
private:
  int nglob; ///< Number of global points in the field

public:
  /// @brief Number of components for this medium and dimension (e.g., 2 for 2D,
  /// 3 for 3D)
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  /// @brief Data class type identifier for this field component
  constexpr static auto data_class = DataClass;

  /**
   * @brief Default constructor.
   *
   * Creates an uninitialized base field with no allocated storage.
   */
  base_field() = default;

  /**
   * @brief Construct base field with specified size and name.
   *
   * Allocates device and host memory views for the field data with the
   * appropriate number of components based on the medium type and dimension.
   * The field is stored as a 2D array [nglob × components].
   *
   * @param nglob Number of global points in the spectral element mesh
   * @param name Descriptive name for the field (used for debugging and
   * profiling)
   */
  base_field(const int nglob, std::string name)
      : nglob(nglob), data(name, nglob, components),
        h_data(Kokkos::create_mirror_view(data)) {}

  template <bool on_device, specfem::data_access::DataClassType U,
            typename std::enable_if_t<U == data_class, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION constexpr type_real &get_value(
      const std::integral_constant<specfem::data_access::DataClassType, U>,
      const int &iglob, const int &icomp) const {

    if constexpr (on_device) {
      return data(iglob, icomp);
    } else {
      return h_data(iglob, icomp);
    }
  }

  template <specfem::sync::kind SyncType> void sync() const {
    if constexpr (SyncType == specfem::sync::kind::HostToDevice) {
      Kokkos::deep_copy(data, h_data);
    } else if constexpr (SyncType == specfem::sync::kind::DeviceToHost) {
      Kokkos::deep_copy(h_data, data);
    }
  }

private:
  /// @brief Device-accessible view type for field data storage (nglob ×
  /// components)
  using ViewType = Kokkos::View<type_real **, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;
  ViewType data;               ///< Device memory view for field data
  ViewType::HostMirror h_data; ///< Host mirror view for CPU operations

protected:
  /**
   * @brief Get the appropriate field view based on execution space.
   *
   * Returns either the device view or host mirror view depending on the
   * template parameter. This method provides the foundation for derived
   * classes to access field data in the appropriate memory space.
   *
   * @tparam on_device If true, return device view; if false, return host view
   * @return Device view or host mirror view of the field data
   *
   * @note This is a protected method intended for use by derived classes
   */
  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION
      std::conditional_t<on_device, ViewType, ViewType::HostMirror>
      get_base_field_view() const {
    if constexpr (on_device) {
      return data;
    } else {
      return h_data;
    }
  }
};

/**
 * @brief Complete field implementation for spectral element wave simulations
 *
 * This class provides a unified interface to all field components required for
 * spectral element wave propagation simulations.
 *
 * Field components by medium type:
 * - **Elastic media**: displacement (u), velocity (\f$\dot{u}\f$), acceleration
 * (\f$\ddot{u}\f$)
 * - **Acoustic media**: potential (\f$\phi\f$), velocity potential
 * (\f$\dot{\phi}\f$), acceleration potential (\f$\ddot{\phi}\f$)
 * - **Poroelastic media**: solid and fluid displacements, velocities,
 * accelerations
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam MediumTag Physical medium type (elastic, acoustic, poroelastic)
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class field_impl
    : public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::displacement>,
      public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::velocity>,
      public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::acceleration>,
      public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::mass_matrix> {
private:
  /// @brief Type alias for displacement field base class
  using displacement_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::displacement>;
  /// @brief Type alias for velocity field base class
  using velocity_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::velocity>;
  /// @brief Type alias for acceleration field base class
  using acceleration_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::acceleration>;
  /// @brief Type alias for mass matrix (inverse) field base class
  using mass_inverse_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::mass_matrix>;

public:
  /// @brief Compile-time dimension tag for this field implementation
  constexpr static auto dimension_tag = DimensionTag;
  /// @brief Compile-time medium tag for this field implementation
  constexpr static auto medium_tag = MediumTag;
  /// @brief Number of field components for this medium and dimension
  /// combination
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  // Bring get_value methods from base classes into scope for unified field access
  using acceleration_base_type::get_value; ///< Access acceleration field values
  using displacement_base_type::get_value; ///< Access displacement field values
  using mass_inverse_base_type::get_value; ///< Access mass matrix field values
  using velocity_base_type::get_value;     ///< Access velocity field values

  /**
   * @brief Default constructor.
   *
   * Creates an uninitialized field implementation with no allocated storage.
   * Fields must be properly initialized before use.
   */
  field_impl() = default;

  /**
   * @brief Construct field implementation from mesh and element information.
   *
   * Initializes all field components (displacement, velocity, acceleration,
   * mass matrix) based on the spectral element mesh structure and element
   * classifications. This constructor determines the number of global points
   * from the mesh and element types, then allocates appropriate storage for all
   * field components.
   *
   * @param mesh Spectral element mesh containing connectivity and global
   * numbering
   * @param element_type Element type classification for medium-specific
   * allocation
   * @param assembly_index_mapping Host view mapping local to global indices
   */
  field_impl(
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_type,
      Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
          assembly_index_mapping);

  /**
   * @brief Construct field implementation with specified number of global
   * points.
   *
   * Allocates storage for all field components (displacement, velocity,
   * acceleration, mass matrix) with the specified number of global points. This
   * constructor is useful when the global point count is known a priori.
   *
   * @param nglob Number of global points in the spectral element mesh
   */
  field_impl(const int nglob);

  /**
   * @brief Synchronize all field components between host and device memory.
   *
   * Performs synchronization of all field components (displacement, velocity,
   * acceleration, mass matrix) in a single operation. This is more efficient
   * than synchronizing each field individually when all fields need to be
   * transferred.
   *
   * @tparam SyncField Synchronization direction (HostToDevice or DeviceToHost)
   */
  template <specfem::sync::kind SyncField> void sync_fields() const;

  int nglob; ///< Number of global points in this field implementation

  /**
   * @brief Get device view of displacement field \f$\mathbf{u}\f$.
   *
   * Returns a device-accessible Kokkos view of the displacement field for use
   * in GPU kernels. The view has dimensions [nglob × components] where
   * components correspond to spatial directions (ux, uy, uz for 3D elastic
   * media).
   *
   * @return Device view of displacement field data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_field() const {
    return displacement_base_type::template get_base_field_view<true>();
  }

  /**
   * @brief Get device view of velocity field \f$\dot{\mathbf{u}}\f$.
   *
   * Returns a device-accessible Kokkos view of the velocity field (first time
   * derivative of displacement) for use in GPU kernels and time integration
   * schemes.
   *
   * @return Device view of velocity field data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_field_dot() const {
    return velocity_base_type::template get_base_field_view<true>();
  }

  /**
   * @brief Get device view of acceleration field \f$\ddot{\mathbf{u}}\f$.
   *
   * Returns a device-accessible Kokkos view of the acceleration field (second
   * time derivative of displacement) for use in GPU kernels and time
   * integration schemes.
   *
   * @return Device view of acceleration field data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_field_dot_dot() const {
    return acceleration_base_type::template get_base_field_view<true>();
  }

  /**
   * @brief Get device view of inverse mass matrix \f$\mathbf{M}^{-1}\f$.
   *
   * Returns a device-accessible Kokkos view of the inverse mass matrix for
   * efficient solution of the spectral element wave equation. In spectral
   * element methods, the mass matrix is typically diagonal, making inversion
   * trivial.
   *
   * @return Device view of inverse mass matrix data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_mass_inverse() const {
    return mass_inverse_base_type::template get_base_field_view<true>();
  }

  /**
   * @brief Get host view of displacement field for CPU operations.
   *
   * Returns a host-accessible Kokkos view of the displacement field for
   * CPU-based operations, post-processing, I/O, and debugging. Data must be
   * synchronized from device before accessing if computations were performed on
   * GPU.
   *
   * @return Host view of displacement field data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_field() const {
    return displacement_base_type::template get_base_field_view<false>();
  }

  /**
   * @brief Get host view of velocity field for CPU operations.
   *
   * Returns a host-accessible Kokkos view of the velocity field for CPU-based
   * post-processing, analysis, and I/O operations.
   *
   * @return Host view of velocity field data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_field_dot() const {
    return velocity_base_type::template get_base_field_view<false>();
  }

  /**
   * @brief Get host view of acceleration field for CPU operations.
   *
   * Returns a host-accessible Kokkos view of the acceleration field for
   * CPU-based post-processing, analysis, and debugging operations.
   *
   * @return Host view of acceleration field data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_field_dot_dot() const {
    return acceleration_base_type::template get_base_field_view<false>();
  }

  /**
   * @brief Get host view of inverse mass matrix for CPU operations.
   *
   * Returns a host-accessible Kokkos view of the inverse mass matrix for
   * CPU-based analysis, debugging, and verification operations.
   *
   * @return Host view of inverse mass matrix data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_mass_inverse() const {
    return mass_inverse_base_type::template get_base_field_view<false>();
  }
};

/**
 * @brief Perform deep copy between field implementations of the same type.
 *
 * Copies all field data (displacement, velocity, acceleration, mass matrix) and
 * both device and host views from source to destination field implementation.
 * This operation preserves all field state and is commonly used for
 * checkpointing, field swapping in time-reversal simulations, and creating
 * backup copies.
 *
 * The function performs synchronous copies of:
 * - Device displacement, velocity, acceleration fields
 * - Host mirrors of all field components
 * - Mass matrix data (both device and host)
 *
 * @tparam DimensionTag Spatial dimension of the fields
 * @tparam MediumTag Medium type of the fields
 * @param dst Destination field implementation (must be pre-allocated)
 * @param src Source field implementation to copy from
 *
 * @pre Both dst and src must have the same dimensions and be properly
 * initialized
 * @post dst contains identical field data as src in both device and host memory
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
void deep_copy(const fields_impl::field_impl<DimensionTag, MediumTag> &dst,
               const fields_impl::field_impl<DimensionTag, MediumTag> &src) {

  Kokkos::deep_copy(dst.get_field(), src.get_field());
  Kokkos::deep_copy(dst.get_field_dot(), src.get_field_dot());
  Kokkos::deep_copy(dst.get_field_dot_dot(), src.get_field_dot_dot());
  Kokkos::deep_copy(dst.get_host_field(), src.get_host_field());
  Kokkos::deep_copy(dst.get_host_field_dot(), src.get_host_field_dot());
  Kokkos::deep_copy(dst.get_host_field_dot_dot(), src.get_host_field_dot_dot());

  return;
}

} // namespace fields_impl
} // namespace specfem::assembly
