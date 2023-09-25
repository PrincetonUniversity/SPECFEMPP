#ifndef SPECFEM_ENUM_HPP
#define SPECFEM_ENUM_HPP

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
/**
 * @brief enums namespace is used to store enumerations.
 *
 */
namespace enums {

/**
 * @brief Cartesian axes
 *
 */
enum class axes {
  x, ///< X axis
  y, ///< Y axis
  z  ///< Z axis
};

namespace seismogram {
enum class type {
  displacement, ///< Displacement seismogram
  velocity,     ///< Velocity Seismogram
  acceleration  ///< Acceleration seismogram
};

enum format {
  seismic_unix, ///< Seismic unix output format
  ascii         ///< ASCII output format
};

} // namespace seismogram

/**
 * @brief element namespace is used to store element properties used in the
 * element class.
 *
 */
namespace element {

/**
 * @brief type of element
 *
 * This is primarily used to label the element as elastic, acoustic or
 * poroelastic.
 *
 */
enum type {
  elastic,    ///< elastic element
  acoustic,   ///< acoustic element
  poroelastic ///< poroelastic element
};

/**
 * @brief dimensionality property of the element
 *
 */
namespace dimension {
/**
 * @brief 2D element
 *
 */
class dim2 {
public:
  constexpr static int dim = 2;

  /**
   * @brief Array to store temporary values when doing reductions
   *
   * @tparam T array type
   */
  template <typename T> struct array_type {
    T data[2]; ///< Data array

    /**
     * @brief operator [] to access the data array
     *
     * @param i index
     * @return T& reference to the data array
     */
    KOKKOS_INLINE_FUNCTION T &operator[](const int &i) { return data[i]; }

    /**
     * @brief operator [] to access the data array
     *
     * @param i index
     * @return const T& reference to the data array
     */
    KOKKOS_INLINE_FUNCTION const T &operator[](const int &i) const {
      return data[i];
    }

    /**
     * @brief operator += to add two arrays
     *
     * @param rhs right hand side array
     * @return array_type<T>& reference to the array
     */
    KOKKOS_INLINE_FUNCTION array_type<T> &operator+=(const array_type<T> &rhs) {
      for (int i = 0; i < 2; i++) {
        data[i] += rhs[i];
      }
      return *this;
    }

    /**
     * @brief Initialize the array for sum reductions
     *
     */
    KOKKOS_INLINE_FUNCTION void init() {
      for (int i = 0; i < 2; i++) {
        data[i] = 0.0;
      }
    }

    // Default constructor
    /**
     * @brief Construct a new array type object
     *
     */
    KOKKOS_INLINE_FUNCTION array_type() { init(); }

    // Copy constructor
    /**
     * @brief Copy constructor
     *
     * @param other other array
     */
    KOKKOS_INLINE_FUNCTION array_type(const array_type<T> &other) {
      for (int i = 0; i < 2; i++) {
        data[i] = other[i];
      }
    }
  };
};
/**
 * @brief 3D element
 *
 */
class dim3 {
public:
  constexpr static int dim = 3;

  /**
   * @brief Array to store temporary values when doing reductions
   *
   * @tparam T array type
   */
  template <typename T> struct array_type {
    T data[3]; ///< Data array

    /**
     * @brief operator [] to access the data array
     *
     * @param i index
     * @return T& reference to the data array
     */
    KOKKOS_INLINE_FUNCTION T &operator[](const int &i) { return data[i]; }

    /**
     * @brief operator [] to access the data array
     *
     * @param i index
     * @return const T& reference to the data array
     */
    KOKKOS_INLINE_FUNCTION const T &operator[](const int &i) const {
      return data[i];
    }

    /**
     * @brief operator += to add two arrays
     *
     * @param rhs right hand side array
     * @return array_type<T>& reference to the array
     */
    KOKKOS_INLINE_FUNCTION array_type<T> &operator+=(const array_type<T> &rhs) {
      for (int i = 0; i < 3; i++) {
        data[i] += rhs[i];
      }
      return *this;
    }

    /**
     * @brief Initialize the array for sum reductions
     *
     */
    KOKKOS_INLINE_FUNCTION void init() {
      for (int i = 0; i < 3; i++) {
        data[i] = 0.0;
      }
    }

    // Default constructor
    /**
     * @brief Construct a new array type object
     *
     */
    KOKKOS_INLINE_FUNCTION array_type() { init(); }

    // Copy constructor
    /**
     * @brief Copy constructor
     *
     * @param other other array
     */
    KOKKOS_INLINE_FUNCTION array_type(const array_type<T> &other) {
      for (int i = 0; i < 3; i++) {
        data[i] = other[i];
      }
    }
  };
};
} // namespace dimension

/**
 * @brief number of quadrature points defined either at compile time or at
 * runtime
 *
 */
namespace quadrature {

/**
 * @brief define the number of quadrature points at runtime
 *
 */
class dynamic_quadrature_points {
public:
  int ngllx; ///< number of quadrature points in the x direction
  int ngllz; ///< number of quadrature points in the z direction

  /**
   * @brief Scratch memory space. Shared memory for GPUs and thread local memory
   * for CPUs.
   *
   */
  using scratch_memory_space =
      specfem::kokkos::DevExecSpace::scratch_memory_space;

  /**
   * @brief Team member type. See Kokkos TeamHandle documentation for more
   * details.
   *
   */
  using member_type = specfem::kokkos::DeviceTeam::member_type;

  /**
   * @brief Scratch view type
   *
   * For dynamic quadrature points, the scratch view type is a 2D dynamic view
   * stored on scratch space.
   *
   * @tparam T Type of the scratch view
   * @tparam N Number of components
   */
  template <typename T, int N>
  using ScratchViewType =
      Kokkos::View<T **[N], Kokkos::LayoutRight,
                   specfem::kokkos::DevScratchSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  /**
   * @brief Deleted default constructor
   *
   */
  dynamic_quadrature_points() = delete;

  /**
   * @brief Construct a new dynamic quadrature points object
   *
   * @param ngllz Number of quadrature points in the z direction
   * @param ngllx Number of quadrature points in the x direction
   */
  dynamic_quadrature_points(const int &ngllz, const int &ngllx)
      : ngllx(ngllx), ngllz(ngllz){};

  /**
   * @brief Destroy the dynamic quadrature points object
   *
   */
  ~dynamic_quadrature_points() = default;

  /**
   * @brief Get the required size of the scratch memory space for a given type T
   * and axes ax1 and ax2
   *
   * @tparam T Tyoe of the scratch view
   * @tparam ax1 Axis 1
   * @tparam ax2 Axis 2
   * @return std::size_t Size of the scratch memory space
   */
  template <typename T, int N, specfem::enums::axes ax1,
            specfem::enums::axes ax2>
  std::size_t shmem_size() const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return ScratchViewType<T, N>::shmem_size(this->ngllx, this->ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return ScratchViewType<T, N>::shmem_size(this->ngllz, this->ngllz);
    } else {
      return ScratchViewType<T, N>::shmem_size(this->ngllz, this->ngllx);
    }
  }

  /**
   * @brief Get the scratch view for a given type T and axes ax1 and ax2
   *
   * @tparam T Type of the scratch view
   * @tparam ax1 Axis 1
   * @tparam ax2 Axis 2
   * @param ptr Address in the scratch memory space to allocate the scratch view
   * @return ScratchViewType<T> Scratch view allocated at the address ptr
   */
  template <typename T, int N, specfem::enums::axes ax1,
            specfem::enums::axes ax2>
  KOKKOS_INLINE_FUNCTION ScratchViewType<T, N>
  ScratchView(scratch_memory_space &ptr) const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return ScratchViewType<T, N>(ptr, this->ngllx, this->ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return ScratchViewType<T, N>(ptr, this->ngllz, this->ngllz);
    } else {
      return ScratchViewType<T, N>(ptr, this->ngllz, this->ngllx);
    }
  };

  /**
   * @brief Get the team thread range for a given axes ax1 and ax2
   *
   * @tparam ax1 Axis 1
   * @tparam ax2 Axis 2
   * @param team_member Team member
   * @return Kokkos::TeamThreadRange Team thread range
   */
  template <specfem::enums::axes ax1, specfem::enums::axes ax2>
  KOKKOS_INLINE_FUNCTION auto
  TeamThreadRange(const member_type &team_member) const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return Kokkos::TeamThreadRange(team_member, ngllx * ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return Kokkos::TeamThreadRange(team_member, ngllz * ngllz);
    } else {
      return Kokkos::TeamThreadRange(team_member, ngllz * ngllx);
    }
  }

  /**
   * @brief Get the number of quadrature points in the x and z directions
   *
   * @param ngllx Number of quadrature points in the x direction
   * @param ngllz Number of quadrature points in the z direction
   */
  KOKKOS_INLINE_FUNCTION void get_ngll(int *ngllx, int *ngllz) const {
    *ngllx = this->ngllx;
    *ngllz = this->ngllz;
  }
};

// Define the number of quadrature points at compile time
template <int NGLL> class static_quadrature_points {

public:
  /**
   * @brief Scratch memory space type
   *
   */
  using scratch_memory_space =
      specfem::kokkos::DevExecSpace::scratch_memory_space;

  /**
   * @brief Team member type. See kokkos TeamHandle documentation for more
   * details.
   *
   */
  using member_type = specfem::kokkos::DeviceTeam::member_type;

  /**
   * @brief Scratch view type for a given type T.
   *
   * For static quadrature points, the scratch view is a 2D view of size NGLL x
   * NGLL defined at compile time.
   *
   * @tparam T Type of the scratch view
   * @tparam N Number of components
   */
  template <typename T, int N>
  using ScratchViewType =
      specfem::kokkos::StaticDeviceScratchView3d<T, NGLL, NGLL, N>;

  /**
   * @brief Construct a new static quadrature points object
   *
   */
  constexpr static_quadrature_points() = default;
  /**
   * @brief Destroy the static quadrature points object
   *
   */
  ~static_quadrature_points() = default;

  /**
   * @brief Get the required size of the scratch memory space for a given type T
   * and axes ax1 and ax2
   *
   * @tparam T Type of the scratch view
   * @tparam ax_1 Axis 1
   * @tparam ax_2 Axis 2
   * @return std::size_t Size of the scratch memory space
   */
  template <typename T, int N, specfem::enums::axes ax_1,
            specfem::enums::axes ax_2>
  std::size_t shmem_size() const {
    return ScratchViewType<T, N>::shmem_size();
  }

  /**
   * @brief Get the scratch view for a given type T and axes ax1 and ax2
   *
   * @tparam T Type of the scratch view
   * @tparam ax_1 Axis 1
   * @tparam ax_2 Axis 2
   * @param ptr Address in the scratch memory space to allocate the scratch view
   * @return ScratchViewType<T> Scratch view allocated at the address ptr
   */
  template <typename T, int N, specfem::enums::axes ax_1,
            specfem::enums::axes ax_2>
  KOKKOS_INLINE_FUNCTION ScratchViewType<T, N>
  ScratchView(const scratch_memory_space &ptr) const {
    return ScratchViewType<T, N>(ptr);
  }

  /**
   * @brief Get the team thread range for a given axes ax1 and ax2
   *
   * @tparam ax_1 Axis 1
   * @tparam ax_2 Axis 2
   * @param team_member Team member
   * @return Kokkos::TeamThreadRange Team thread range
   */
  template <specfem::enums::axes ax_1, specfem::enums::axes ax_2>
  KOKKOS_INLINE_FUNCTION auto
  TeamThreadRange(const member_type &team_member) const {
    return Kokkos::TeamThreadRange(team_member, NGLL * NGLL);
  }

  /**
   * @brief Get the number of quadrature points in the x and z directions
   *
   * @param ngllx Number of quadrature points in the x direction
   * @param ngllz Number of quadrature points in the z direction
   */
  KOKKOS_INLINE_FUNCTION constexpr void get_ngll(int *ngllx, int *ngllz) const {
    *ngllx = NGLL;
    *ngllz = NGLL;
  }
};

} // namespace quadrature

/**
 * @brief medium property of the element
 *
 */
namespace medium {
/**
 * @brief Elastic medium
 *
 */
class elastic {
public:
  /**
   * @brief constexpr defining the type of the element
   *
   */
  constexpr static specfem::enums::element::type value =
      specfem::enums::element::elastic;
  /**
   * @brief Number of components for this medium
   *
   */
  constexpr static int components = 2;
};

/**
 * @brief Acoustic medium
 *
 */
class acoustic {
public:
  /**
   * @brief constexpr defining the type of the element
   *
   */
  constexpr static specfem::enums::element::type value =
      specfem::enums::element::acoustic;
  /**
   * @brief constexpr defining number of components for this medium.
   *
   */
  constexpr static int components = 1;
};

} // namespace medium

/**
 * @brief Elemental properties
 *
 * Properties can be utilized to distinguish elements based on physics or to
 * optimize kernel calculations for specific elements.
 */
namespace property {
class isotropic {};
} // namespace property
} // namespace element

namespace coupling {
namespace edge {
enum type {
  TOP,    ///< Top edge
  BOTTOM, ///< Bottom edge
  LEFT,   ///< Left edge
  RIGHT   ///< Right edge
};

constexpr int num_edges = 4;
} // namespace edge
} // namespace coupling
} // namespace enums
} // namespace specfem

#endif
