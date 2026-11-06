#pragma once

#include "specfem/injection/method/injection_method.hpp"
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace specfem {
namespace injection {

/**
 * @brief Injection methods selectable by name in the parameter file.
 *
 * The program uses specfem::injection::default_method_registry(); tests build
 * their own registry so test-only methods never reach production.
 *
 * @code
 * specfem::injection::MethodRegistry registry;
 * registry.add<MyMethod>(); // MyMethod().name() is "my-method"
 * const auto method = registry.make("my-method");
 * @endcode
 */
class MethodRegistry {
public:
  using Method = InjectionMethod<specfem::element::dimension_tag::dim3>;

  /**
   * @brief Register a method under its name().
   * @tparam M Default-constructible method
   * @throws std::runtime_error if the name is already registered
   */
  template <typename M> void add() {
    const std::string name(M().name());
    if (!factories_.emplace(name, [] { return std::make_shared<M>(); }).second)
      throw std::runtime_error("Injection method '" + name +
                               "' is already registered.");
  }

  /**
   * @brief Whether a method is registered.
   * @param name Method name
   * @return True if @p name is registered
   */
  [[nodiscard]] bool contains(std::string_view name) const;

  /**
   * @brief Create a method.
   * @param name Method name
   * @return New method instance
   * @throws std::runtime_error naming the registered methods if @p name is
   * unknown
   */
  [[nodiscard]] std::shared_ptr<Method> make(std::string_view name) const;

  /**
   * @brief Registered method names, sorted.
   * @return Names
   */
  [[nodiscard]] std::vector<std::string> names() const;

  /**
   * @brief Error message for an unknown method name.
   * @param name Requested method name
   * @return Message naming the registered methods
   */
  [[nodiscard]] std::string unknown_method_message(std::string_view name) const;

private:
  std::map<std::string, std::function<std::shared_ptr<Method>()>, std::less<>>
      factories_; ///< Method factories by name
};

/**
 * @brief Injection methods built into SPECFEM++.
 * @return Registry of the built-in methods
 */
const MethodRegistry &default_method_registry();

} // namespace injection
} // namespace specfem
