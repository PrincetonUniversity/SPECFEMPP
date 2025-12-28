#pragma once

#include <regex>
#include <string>
#include <type_traits>

namespace specfem::test_fixture::impl {

template <typename T, typename = void> struct description {
  static constexpr bool has = false;
  static std::string get(const int &indent = 0) {
    std::string indent_str(indent, ' ');
    return indent_str = "<no description>";
  }
};

template <typename T>
struct description<
    T, std::enable_if_t<std::is_same_v<decltype(T::description()), std::string>,
                        void> > {
  static constexpr bool has = true;
  static std::string get(const int &indent = 0) {
    std::string indent_str(indent, ' ');
    return std::regex_replace(
        T::description(), std::regex("^(.+)$", std::regex_constants::multiline),
        indent_str + "$&");
  }
};

template <typename T, typename = void> struct name {
  static constexpr bool has = false;
  static std::string get() { return "<unnamed>"; }
};

template <typename T>
struct name<T, std::enable_if_t<
                   std::is_same_v<decltype(T::name()), std::string>, void> > {
  static constexpr bool has = true;
  static std::string get() { return T::name(); }
};
} // namespace specfem::test_fixture::impl
