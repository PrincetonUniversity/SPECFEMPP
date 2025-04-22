#pragma once

#include "domain_view.hpp"
#include "enumerations/medium.hpp"

#define _CREATE_NAMED_VARIABLE(prefix, postfix)                                \
  BOOST_PP_CAT(prefix, BOOST_PP_CAT(_, postfix))

#define _ACCESS_ELEMENT_ON_DEVICE(elem, index)                                 \
  BOOST_PP_SEQ_ELEM(0, elem)(index.ispec, index.iz, index.ix)

#define _ACCESS_ELEMENT_ON_HOST(elem, index)                                   \
  BOOST_PP_CAT(h_, BOOST_PP_SEQ_ELEM(0, elem))(index.ispec, index.iz, index.ix)

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
  KOKKOS_INLINE_FUNCTION void for_each_on_device(const IndexType &index,       \
                                                 FunctorType f) const {        \
    BOOST_PP_SEQ_FOR_EACH(_CALL_FUNCTOR_ON_DEVICE, (f, index), seq)            \
  }                                                                            \
  template <typename FunctorType, typename IndexType>                          \
  void for_each_on_host(const IndexType &index, FunctorType f) const {         \
    BOOST_PP_SEQ_FOR_EACH(_CALL_FUNCTOR_ON_HOST, (f, index), seq)              \
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
  BOOST_PP_TUPLE_ELEM(0, data)(BOOST_PP_SEQ_ELEM(0, elem),                     \
                               BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)));

#define _ACCESS_HOST_VIEW(r, data, elem)                                       \
  BOOST_PP_TUPLE_ELEM(0, data)(BOOST_PP_CAT(h_, BOOST_PP_SEQ_ELEM(0, elem)),   \
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
