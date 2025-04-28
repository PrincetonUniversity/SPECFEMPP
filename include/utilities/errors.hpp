#pragma once

namespace specfem {
namespace utilities {

template <auto... T> constexpr bool always_false = false;

}
} // namespace specfem
