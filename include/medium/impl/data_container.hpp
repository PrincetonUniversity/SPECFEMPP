#pragma once

#include "domain_view.hpp"
#include "enumerations/medium.hpp"

#define _CREATE_NAMED_VARIABLE(prefix, postfix)                                \
  BOOST_PP_CAT(prefix, BOOST_PP_CAT(_, postfix))

#define _ACCESS_ELEMENT_ON_DEVICE_CONST(elem, index)                           \
  BOOST_PP_CAT(_, BOOST_PP_SEQ_ELEM(0, elem))(index)

#define _ACCESS_ELEMENT_ON_DEVICE(elem, index) BOOST_PP_SEQ_ELEM(0, elem)[index]

#define _ACCESS_ELEMENT_ON_HOST(elem, index)                                   \
  BOOST_PP_CAT(h_, BOOST_PP_SEQ_ELEM(0, elem))[index]

#define _CALL_FUNCTOR_ON_DEVICE_CONST(r, data, elem)                           \
  Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space, \
               Kokkos::MemoryTraits<Kokkos::RandomAccess> >                    \
      BOOST_PP_CAT(_, BOOST_PP_SEQ_ELEM(0, elem)) =                            \
          BOOST_PP_SEQ_ELEM(0, elem).get_base_view();                          \
  BOOST_PP_TUPLE_ELEM(0, data)                                                 \
  (_ACCESS_ELEMENT_ON_DEVICE_CONST(elem, BOOST_PP_TUPLE_ELEM(1, data)),        \
   static_cast<std::size_t>(BOOST_PP_SEQ_ELEM(1, elem)));

#define _CALL_FUNCTOR_ON_DEVICE(r, data, elem)                                 \
  BOOST_PP_TUPLE_ELEM(0, data)                                                 \
  (_ACCESS_ELEMENT_ON_DEVICE(elem, BOOST_PP_TUPLE_ELEM(1, data)),              \
   BOOST_PP_SEQ_ELEM(1, elem));

#define _CALL_FUNCTOR_ON_HOST(r, data, elem)                                   \
  BOOST_PP_TUPLE_ELEM(0, data)                                                 \
  (_ACCESS_ELEMENT_ON_HOST(elem, BOOST_PP_TUPLE_ELEM(1, data)),                \
   BOOST_PP_SEQ_ELEM(1, elem));

#define _DATA_ACCESSOR(seq)                                                    \
  template <typename FunctorType, typename IndexType>                          \
  KOKKOS_INLINE_FUNCTION std::enable_if_t<                                     \
      std::is_invocable_v<FunctorType, const type_real &, std::size_t>, void>  \
  for_each_on_device(const IndexType &index, FunctorType f) const {            \
    const auto &mapping =                                                      \
        BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, seq)).get_mapping();         \
    const std::size_t _index = mapping(index.ispec, index.iz, index.ix);       \
    BOOST_PP_SEQ_FOR_EACH(_CALL_FUNCTOR_ON_DEVICE_CONST, (f, _index), seq)     \
  }                                                                            \
  template <typename FunctorType, typename IndexType>                          \
  KOKKOS_INLINE_FUNCTION std::enable_if_t<                                     \
      (!std::is_invocable_v<FunctorType, const type_real &, std::size_t> &&    \
       std::is_invocable_v<FunctorType, type_real &, std::size_t>),            \
      void>                                                                    \
  for_each_on_device(const IndexType &index, FunctorType f) const {            \
    const auto &mapping =                                                      \
        BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, seq)).get_mapping();         \
    const std::size_t _index = mapping(index.ispec, index.iz, index.ix);       \
    BOOST_PP_SEQ_FOR_EACH(_CALL_FUNCTOR_ON_DEVICE, (f, _index), seq)           \
  }                                                                            \
  template <typename FunctorType, typename IndexType>                          \
  void for_each_on_host(const IndexType &index, FunctorType f) const {         \
    const auto &mapping =                                                      \
        BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, seq)).get_mapping();         \
    const std::size_t _index = mapping(index.ispec, index.iz, index.ix);       \
    BOOST_PP_SEQ_FOR_EACH(_CALL_FUNCTOR_ON_HOST, (f, _index), seq)             \
  }

#define _DEFINE_DOMAIN_VIEW(r, data, elem)                                     \
  specfem::kokkos::DomainView2d<type_real, 3,                                  \
                                Kokkos::DefaultExecutionSpace::memory_space>   \
      BOOST_PP_SEQ_ELEM(0, elem);                                              \
  typename decltype(BOOST_PP_SEQ_ELEM(0, elem))::HostMirror BOOST_PP_CAT(      \
      h_, BOOST_PP_SEQ_ELEM(0, elem));

#define _INSTANCE_DEVICE_VIEW(r, data, elem)                                   \
  BOOST_PP_SEQ_ELEM(0, elem)                                                   \
  (BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)), nspec, ngllz, ngllx)

#define _INSTANCE_HOST_VIEW(r, data, elem)                                     \
  BOOST_PP_CAT(h_, BOOST_PP_SEQ_ELEM(0, elem))                                 \
  (specfem::kokkos::create_mirror_view(BOOST_PP_SEQ_ELEM(0, elem)))

#define _DATA_DEFINITION(seq) BOOST_PP_SEQ_FOR_EACH(_DEFINE_DOMAIN_VIEW, _, seq)

#define _DATA_CONSTRUCTORS(seq)                                                \
  data_container() = default;                                                  \
  data_container(const int nspec, const int ngllz, const int ngllx)            \
      : BOOST_PP_SEQ_ENUM(                                                     \
            BOOST_PP_SEQ_TRANSFORM(_INSTANCE_DEVICE_VIEW, _, seq)),            \
        BOOST_PP_SEQ_ENUM(                                                     \
            BOOST_PP_SEQ_TRANSFORM(_INSTANCE_HOST_VIEW, _, seq)) {}

#define _SYNC_DEVICE(r, data, elem)                                            \
  specfem::kokkos::deep_copy(BOOST_PP_SEQ_ELEM(0, elem),                       \
                             BOOST_PP_CAT(h_, BOOST_PP_SEQ_ELEM(0, elem)));

#define _SYNC_HOST(r, data, elem)                                              \
  specfem::kokkos::deep_copy(BOOST_PP_CAT(h_, BOOST_PP_SEQ_ELEM(0, elem)),     \
                             BOOST_PP_SEQ_ELEM(0, elem));

#define _DATA_SYNCHRONIZE(seq)                                                 \
  void copy_to_device() { BOOST_PP_SEQ_FOR_EACH(_SYNC_DEVICE, _, seq) }        \
  void copy_to_host() { BOOST_PP_SEQ_FOR_EACH(_SYNC_HOST, _, seq) }

#define _ACCESS_DEVICE_VIEW(r, data, elem)                                     \
  BOOST_PP_TUPLE_ELEM(0, data)                                                 \
  (BOOST_PP_SEQ_ELEM(0, elem), BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)));

#define _ACCESS_HOST_VIEW(r, data, elem)                                       \
  BOOST_PP_TUPLE_ELEM(0, data)                                                 \
  (BOOST_PP_CAT(h_, BOOST_PP_SEQ_ELEM(0, elem)),                               \
   BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)));

#define _VIEW_ACCESSOR(seq)                                                    \
  template <typename FunctorType>                                              \
  void for_each_device_view(FunctorType f) const {                             \
    BOOST_PP_SEQ_FOR_EACH(_ACCESS_DEVICE_VIEW, (f), seq)                       \
  }                                                                            \
  template <typename FunctorType>                                              \
  void for_each_host_view(FunctorType f) const {                               \
    BOOST_PP_SEQ_FOR_EACH(_ACCESS_HOST_VIEW, (f), seq)                         \
  }

#define _DATA_CONTAINER_NUMBERED_SEQ(seq)                                      \
  _DATA_DEFINITION(seq)                                                        \
  _DATA_CONSTRUCTORS(seq)                                                      \
  _DATA_ACCESSOR(seq)                                                          \
  _VIEW_ACCESSOR(seq)                                                          \
  _DATA_SYNCHRONIZE(seq)

#define _CREATE_SEQUENCE(r, data, i, elem) ((elem)(i))

#define _CREATE_NUMBERED_SEQ(seq)                                              \
  (BOOST_PP_SEQ_FOR_EACH_I(_CREATE_SEQUENCE, _, seq))

#define _DATA_CONTAINER_SEQ(seq)                                               \
  BOOST_PP_EXPAND(_DATA_CONTAINER_NUMBERED_SEQ _CREATE_NUMBERED_SEQ(seq))

#define _ARGS(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)

/**
 * @brief Generate a data container where each element within the variadic
 * argument list is a DomainView2d.
 *
 * @param ... Variadic arguments representing the names of the DomainView2d
 * elements.
 *
 * @details This macro creates the following structure:
 * - A data container with a set of DomainView2d elements, each named according
 * to the provided arguments.
 * - Data container constructor that initializes each DomainView2d with the
 * specified number of spectral elements (nspec), number of GLL points in z
 * (ngllz), and number of GLL points in x (ngllx).
 * - Accessor methods for iterating over the elements on both device and host.
 * - Synchronization methods to copy data between device and host views.
 * - Accessor methods to iterate over the DomainView2d elements.
 *
 * Example usage:
 * @code
 * DATA_CONTAINER(rho, kappa)
 * @endcode
 * Generated code :
 * @code
 * DomainView2d rho;
 * DomainView2d kappa;
 * typename decltype(rho)::HostMirror h_rho;
 * typename decltype(kappa)::HostMirror h_kappa;
 * data_container() = default;
 * data_container(const int nspec, const int ngllz, const int ngllx)
 *     : rho("rho", nspec, ngllz, ngllx),
 *       kappa("kappa", nspec, ngllz, ngllx),
 *       h_rho(specfem::kokkos::create_mirror_view(rho)),
 *       h_kappa(specfem::kokkos::create_mirror_view(kappa)) {}
 * template <typename FunctorType, typename IndexType>
 * KOKKOS_INLINE_FUNCTION
 * void for_each_on_device(const IndexType &index, FunctorType f) const {
 *   const auto mapping = rho.get_mapping();
 *   const std::size_t _index = mapping(index.ispec, index.iz, index.ix);
 *   f(rho[_index], "rho");
 *   f(kappa[_index], "kappa");
 * }
 * template <typename FunctorType, typename IndexType>
 * void for_each_on_host(const IndexType &index, FunctorType f) const {
 *   const auto mapping = rho.get_mapping();
 *   const std::size_t _index = mapping(index.ispec, index.iz, index.ix);
 *   f(h_rho[_index], "rho");
 *   f(h_kappa[_index], "kappa");
 * }
 * void copy_to_device() {
 *   specfem::kokkos::deep_copy(rho, h_rho);
 *   specfem::kokkos::deep_copy(kappa, h_kappa);
 * }
 * void copy_to_host() {
 *   specfem::kokkos::deep_copy(h_rho, rho);
 *   specfem::kokkos::deep_copy(h_kappa, kappa);
 * }
 * template <typename FunctorType>
 * void for_each_device_view(FunctorType f) const {
 *   f(rho, "rho");
 *   f(kappa, "kappa");
 * }
 * template <typename FunctorType>
 * void for_each_host_view(FunctorType f) const {
 *   f(h_rho, "h_rho");
 *   f(h_kappa, "h_kappa");
 * }
 * @endcode
 *
 */
#define DATA_CONTAINER(...) _DATA_CONTAINER_SEQ(_ARGS(__VA_ARGS__))

namespace specfem {
namespace medium {
namespace properties {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename Enable = void>
struct data_container;

} // namespace properties

namespace kernels {
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename Enable = void>
struct data_container;
} // namespace kernels
} // namespace medium
} // namespace specfem
