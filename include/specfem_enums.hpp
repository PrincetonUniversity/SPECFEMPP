#ifndef SPECFEM_ENUM_HPP
#define SPECFEM_ENUM_HPP

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace enums {

enum class axes {
  x, ///< X axis
  y, ///< Y axis
  z  ///< Z axis
};

namespace element {

enum type {
  elastic,    ///< elastic element
  acoustic,   ///< acoustic element
  poroelastic ///< poroelastic element
};

namespace dimension {
class dim2 {
  constexpr static int dim = 2;
};
class dim3 {
  constexpr static int dim = 3;
};
} // namespace dimension

namespace quadrature {

class dynamic_quadrature_points {
public:
  int ngllx;
  int ngllz;

  using scratch_memory_space =
      specfem::kokkos::DevExecSpace::scratch_memory_space;

  using member_type = specfem::kokkos::DeviceTeam::member_type;

  template <typename T>
  using ScratchViewType = specfem::kokkos::DeviceScratchView2d<T>;

  dynamic_quadrature_points() = delete;

  dynamic_quadrature_points(const int &ngllz, const int &ngllx)
      : ngllx(ngllx), ngllz(ngllz){};

  ~dynamic_quadrature_points() = default;

  template <typename T, specfem::enums::axes ax1, specfem::enums::axes ax2>
  std::size_t shmem_size() const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return ScratchViewType<T>::shmem_size(this->ngllx, this->ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return ScratchViewType<T>::shmem_size(this->ngllz, this->ngllz);
    } else {
      return ScratchViewType<T>::shmem_size(this->ngllz, this->ngllx);
    }
  }

  template <typename T, specfem::enums::axes ax1, specfem::enums::axes ax2>
  KOKKOS_INLINE_FUNCTION ScratchViewType<T>
  ScratchView(scratch_memory_space &ptr) const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return ScratchViewType<T>(ptr, this->ngllx, this->ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return ScratchViewType<T>(ptr, this->ngllz, this->ngllz);
    } else {
      return ScratchViewType<T>(ptr, this->ngllz, this->ngllx);
    }
  };

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

  KOKKOS_INLINE_FUNCTION void get_ngll(int *ngllx, int *ngllz) const {
    *ngllx = this->ngllx;
    *ngllz = this->ngllz;
  }
};

// Define the number of quadrature points at compile time
template <int N> class static_quadrature_points {

public:
  constexpr static int NGLL = N;

  using scratch_memory_space =
      specfem::kokkos::DevExecSpace::scratch_memory_space;

  using member_type = specfem::kokkos::DeviceTeam::member_type;

  template <typename T>
  using ScratchViewType =
      specfem::kokkos::StaticDeviceScratchView2d<T, NGLL, NGLL>;

  constexpr static_quadrature_points() = default;
  ~static_quadrature_points() = default;

  template <typename T, specfem::enums::axes ax_1, specfem::enums::axes ax_2>
  std::size_t shmem_size() const {
    return ScratchViewType<T>::shmem_size();
  }

  template <typename T, specfem::enums::axes ax_1, specfem::enums::axes ax_2>
  KOKKOS_INLINE_FUNCTION ScratchViewType<T>
  ScratchView(const scratch_memory_space &ptr) const {
    return ScratchViewType<T>(ptr);
  }

  template <specfem::enums::axes ax_1, specfem::enums::axes ax_2>
  KOKKOS_INLINE_FUNCTION auto
  TeamThreadRange(const member_type &team_member) const {
    return Kokkos::TeamThreadRange(team_member, NGLL * NGLL);
  }

  KOKKOS_INLINE_FUNCTION constexpr void get_ngll(int *ngllx, int *ngllz) const {
    *ngllx = NGLL;
    *ngllz = NGLL;
  }
};

} // namespace quadrature

namespace medium {
class elastic {
public:
  constexpr static specfem::enums::element::type value =
      specfem::enums::element::elastic;
  constexpr static int components = 2;
};
} // namespace medium

namespace property {
class isotropic {};
} // namespace property
} // namespace element
} // namespace enums
} // namespace specfem

#endif
