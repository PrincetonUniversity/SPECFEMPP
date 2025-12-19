#pragma once

namespace specfem::chunk_edge {

namespace impl {

template <specfem::dimension::type DimensionTag, int NumberElements,
          int NQuadIntersection, int NQuadElement,
          specfem::data_access::DataClassType DataClass,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct transfer_function;

template <int NumberElements, int NQuadIntersection, int NQuadElement,
          specfem::data_access::DataClassType DataClass,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits>
struct transfer_function<specfem::dimension::type::dim2, NumberElements,
                         NQuadIntersection, NQuadElement, DataClass,
                         InterfaceTag, BoundaryTag, MemorySpace, MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge, DataClass,
          specfem::dimension::type::dim2, false> {

public:
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_element = NQuadElement;
  static constexpr int n_quad_intersection = NQuadIntersection;
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  using TransferViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, specfem::dimension::type::dim2, NumberElements, NQuadElement,
      NQuadIntersection, false, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data of the transfer
                     ///< function.

private:
  TransferViewType data_;

public:
  template <typename U = TransferViewType,
            typename std::enable_if_t<
                std::is_convertible<TransferViewType, U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION transfer_function(const U &transfer_function)
      : data_(transfer_function) {}

  KOKKOS_INLINE_FUNCTION
  transfer_function() = default;

  template <typename MemberType, typename U = TransferViewType,
            typename std::enable_if_t<U::memory_traits::is_unmanaged == true,
                                      int> = 0>
  KOKKOS_INLINE_FUNCTION transfer_function(const MemberType &team)
      : data_(team.team_scratch(0)) {}

  constexpr static int shmem_size() { return TransferViewType::shmem_size(); }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) const {
    return data_(indices...);
  }
};

} // namespace impl

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection, int NQuadElement>
using transfer_function_self = impl::transfer_function<
    DimensionTag, NumberElements, NQuadIntersection, NQuadElement,
    specfem::data_access::DataClassType::transfer_function_self, InterfaceTag,
    BoundaryTag>;

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection, int NQuadElement>
using transfer_function_coupled = impl::transfer_function<
    DimensionTag, NumberElements, NQuadIntersection, NQuadElement,
    specfem::data_access::DataClassType::transfer_function_coupled,
    InterfaceTag, BoundaryTag>;

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct intersection_factor;

template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection, typename MemorySpace, typename MemoryTraits>
struct intersection_factor<specfem::dimension::type::dim2, InterfaceTag,
                           BoundaryTag, NumberElements, NQuadIntersection,
                           MemorySpace, MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::intersection_factor,
          specfem::dimension::type::dim2, false> {

public:
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_intersection = NQuadIntersection;
  using IntersectionFactorViewType =
      Kokkos::View<type_real[NumberElements][NQuadIntersection], MemorySpace,
                   MemoryTraits>;

private:
  IntersectionFactorViewType data_;

public:
  template <
      typename U = IntersectionFactorViewType,
      typename std::enable_if_t<
          std::is_convertible<IntersectionFactorViewType, U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_factor(const U &intersection_factor)
      : data_(intersection_factor) {}

  KOKKOS_INLINE_FUNCTION
  intersection_factor() = default;

  template <typename MemberType, typename U = IntersectionFactorViewType,
            typename std::enable_if_t<U::memory_traits::is_unmanaged == true,
                                      int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_factor(const MemberType &team)
      : data_(team.team_scratch(0)) {}

  constexpr static int shmem_size() {
    return IntersectionFactorViewType::shmem_size();
  }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) const {
    return data_(indices...);
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct intersection_normal;

template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection, typename MemorySpace, typename MemoryTraits>
struct intersection_normal<specfem::dimension::type::dim2, InterfaceTag,
                           BoundaryTag, NumberElements, NQuadIntersection,
                           MemorySpace, MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::intersection_normal,
          specfem::dimension::type::dim2, false> {

public:
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_intersection = NQuadIntersection;
  using IntersectionNormalViewType =
      Kokkos::View<type_real[NumberElements][NQuadIntersection][2], MemorySpace,
                   MemoryTraits>; ///< Underlying view used to
                                  ///< store data of the transfer
                                  ///< function.

private:
  IntersectionNormalViewType data_;

public:
  template <
      typename U = IntersectionNormalViewType,
      typename std::enable_if_t<
          std::is_convertible<IntersectionNormalViewType, U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_normal(const U &intersection_normal)
      : data_(intersection_normal) {}

  KOKKOS_INLINE_FUNCTION
  intersection_normal() = default;

  template <typename MemberType, typename U = IntersectionNormalViewType,
            typename std::enable_if_t<U::memory_traits::is_unmanaged == true,
                                      int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_normal(const MemberType &team)
      : data_(team.team_scratch(0)) {}

  constexpr static int shmem_size() {
    return IntersectionNormalViewType::shmem_size();
  }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) const {
    return data_(indices...);
  }
};

template <typename... Accessors>
struct NonconformingAccessorPack
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::dimension::type::dim2, false>,
      public Accessors... {

  constexpr static auto dimension_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::dimension_tag;
  constexpr static auto interface_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::interface_tag;
  constexpr static auto boundary_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::boundary_tag;
  constexpr static size_t n_accessors = sizeof...(Accessors);
  using packed_accessors = std::tuple<Accessors...>;
  constexpr static auto connection_tag =
      specfem::connections::type::nonconforming;

  constexpr static auto data_class =
      specfem::data_access::DataClassType::nonconforming_interface;

  static_assert(
      (std::is_same_v<
           std::integral_constant<specfem::dimension::type,
                                  Accessors::dimension_tag>,
           std::integral_constant<specfem::dimension::type, dimension_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "dimension_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::interface::interface_tag,
                                             Accessors::interface_tag>,
                      std::integral_constant<specfem::interface::interface_tag,
                                             interface_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "interface_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::element::boundary_tag,
                                             Accessors::boundary_tag>,
                      std::integral_constant<specfem::element::boundary_tag,
                                             boundary_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "boundary_tag");

  KOKKOS_INLINE_FUNCTION NonconformingAccessorPack() = default;

  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPack(const Accessors &...accessors)
      : Accessors(accessors)... {};

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION type_real operator()(Indices... indices) const =
      delete;

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION NonconformingAccessorPack(const MemberType &team)
      : Accessors(team)... {}

  constexpr static int shmem_size() {
    return (Accessors::shmem_size() + ... + 0);
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection, int NQuadElement>
using coupling_terms_pack = NonconformingAccessorPack<
    transfer_function_coupled<DimensionTag, InterfaceTag, BoundaryTag,
                              NumberElements, NQuadIntersection, NQuadElement>,
    intersection_normal<DimensionTag, InterfaceTag, BoundaryTag, NumberElements,
                        NQuadIntersection> >;

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection>
using integral_data_pack = NonconformingAccessorPack<
    intersection_factor<DimensionTag, InterfaceTag, BoundaryTag, NumberElements,
                        NQuadIntersection> >;

} // namespace specfem::chunk_edge
