#pragma once

#define POINT_VALUE_ACCESSOR(r, data, elem)                                    \
  KOKKOS_INLINE_FUNCTION                                                       \
  const value_type BOOST_PP_SEQ_ELEM(0, elem)() const {                        \
    return _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)];                  \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION                                                       \
  value_type &BOOST_PP_SEQ_ELEM(0, elem)() {                                   \
    return _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)];                  \
  }

#define POINT_VALUE_ACCESSORS(seq)                                             \
  BOOST_PP_SEQ_FOR_EACH(POINT_VALUE_ACCESSOR, _, seq)

#define POINT_OPERATOR_DEFINITION(seq)                                         \
  KOKKOS_INLINE_FUNCTION const value_type operator[](const int i) const {      \
    return _point_data_container[i];                                           \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION value_type &operator[](const int i) {                 \
    return _point_data_container[i];                                           \
  }

#define POINT_BOOLEAN_OPERATOR_DEFINITION(seq)                                 \
  KOKKOS_INLINE_FUNCTION bool operator==(const point_container &other) const { \
    if (nprops != other.nprops) {                                              \
      return false;                                                            \
    }                                                                          \
    for (int i = 0; i < nprops; ++i) {                                         \
      if (!specfem::utilities::is_close(_point_data_container[i],              \
                                        other._point_data_container[i])) {     \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  }

#define POINT_CONSTRUCTOR(seq)                                                 \
  KOKKOS_INLINE_FUNCTION point_container() = default;                          \
  template <typename... Args,                                                  \
            typename std::enable_if_t<sizeof...(Args) == nprops, int> = 0>     \
  KOKKOS_INLINE_FUNCTION point_container(Args... args)                         \
      : _point_data_container{ static_cast<value_type>(args)... } {}           \
  KOKKOS_INLINE_FUNCTION                                                       \
  point_container(const value_type *value) {                                   \
    for (int i = 0; i < nprops; ++i) {                                         \
      _point_data_container[i] = value[i];                                     \
    }                                                                          \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION point_container(const value_type &value) {            \
    for (int i = 0; i < nprops; ++i) {                                         \
      _point_data_container[i] = value;                                        \
    }                                                                          \
  }

#define POINT_PRINT_VALUE(r, message, elem)                                    \
  message << "\n\t\t" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)) << " = " << _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)];

#define POINT_PRINT(seq)                                                       \
  template <typename U = simd>                                                 \
  typename std::enable_if_t<!U::using_simd, void> print(                       \
      std::ostringstream &message) const {                                     \
    message << "\n\t Point data: ";                                            \
    BOOST_PP_SEQ_FOR_EACH(POINT_PRINT_VALUE, message, seq)                     \
    message << "\n";                                                           \
  }

#define POINT_PRINT_VALUE_SIMD(r, data, elem)                                  \
  BOOST_PP_TUPLE_ELEM(0, data)                                                 \
      << "\n\t\t" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)) << " = "                                                 \
                      << _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)]     \
                                              [BOOST_PP_TUPLE_ELEM(1, data)];

#define POINT_PRINT_SIMD(seq)                                                  \
  template <typename U = simd>                                                 \
  typename std::enable_if_t<U::using_simd, void> print(                        \
      std::ostringstream &message) const {                                     \
    message << "\n\t Point data: ";                                            \
    for (std::size_t lane = 0; lane < simd::size(); ++lane) {                  \
      message << "\n\t Lane " << lane << ": ";                                 \
      BOOST_PP_SEQ_FOR_EACH(POINT_PRINT_VALUE_SIMD, (message, lane), seq)      \
    }                                                                          \
    message << "\n";                                                           \
  }

#define POINT_DATA_CONTAINER_NUMBERED_SEQ(seq)                                 \
public:                                                                        \
  constexpr static int nprops = BOOST_PP_SEQ_SIZE(seq);                        \
  value_type _point_data_container[nprops] = { 0 };                            \
                                                                               \
private:                                                                       \
  POINT_PRINT(seq)                                                             \
  POINT_PRINT_SIMD(seq)                                                        \
public:                                                                        \
  POINT_CONSTRUCTOR(seq)                                                       \
  POINT_BOOLEAN_OPERATOR_DEFINITION(seq)                                       \
  POINT_VALUE_ACCESSORS(seq)                                                   \
  POINT_OPERATOR_DEFINITION(seq)                                               \
  KOKKOS_INLINE_FUNCTION bool operator!=(const point_container &other) const { \
    return !(*this == other);                                                  \
  }                                                                            \
  std::string print() const {                                                  \
    std::ostringstream message;                                                \
    print(message);                                                            \
    return message.str();                                                      \
  }

#define POINT_CREATE_SEQUENCE(r, data, i, elem) ((elem)(i))

#define POINT_CREATE_NUMBERED_SEQ(seq)                                         \
  (BOOST_PP_SEQ_FOR_EACH_I(POINT_CREATE_SEQUENCE, _, seq))

#define POINT_DATA_CONTAINER_SEQ(seq)                                          \
  BOOST_PP_EXPAND(                                                             \
      POINT_DATA_CONTAINER_NUMBERED_SEQ POINT_CREATE_NUMBERED_SEQ(seq))

#define POINT_ARGS(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)

#define POINT_CONTAINER(...) POINT_DATA_CONTAINER_SEQ(POINT_ARGS(__VA_ARGS__))
