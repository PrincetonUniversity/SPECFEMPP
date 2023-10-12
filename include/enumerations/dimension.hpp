#ifndef _ENUMERATIONS_DIMENSION_HPP_
#define _ENUMERATIONS_DIMENSION_HPP_

namespace specfem {
namespace enums {
namespace element {
/**
 * @namespace dimensionality property of the element
 *
 */
namespace dimension {
/**
 * @brief 2D element
 *
 */
class dim2 {
public:
  constexpr static int dim = 2; ///< Dimensionality of the element

  /**
   * @brief Array to store temporary values when doing Kokkos reductions
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
  constexpr static int dim = 3; ///< Dimensionality of the element

  /**
   * @brief Array to store temporary values when doing Kokkos reductions
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
} // namespace element
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_DIMENSION_HPP_ */
