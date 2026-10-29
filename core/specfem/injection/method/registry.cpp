#include "specfem/injection/method/registry.hpp"
#include "specfem/injection/method/pointwise_surface_method.hpp"
#include "specfem/injection/method/stacey_injection.hpp"

bool specfem::injection::MethodRegistry::contains(
    const std::string_view name) const {
  return factories_.find(name) != factories_.end();
}

std::shared_ptr<specfem::injection::MethodRegistry::Method>
specfem::injection::MethodRegistry::make(const std::string_view name) const {
  const auto found = factories_.find(name);
  if (found == factories_.end())
    throw std::runtime_error(unknown_method_message(name));
  return found->second();
}

std::vector<std::string> specfem::injection::MethodRegistry::names() const {
  std::vector<std::string> names;
  for (const auto &[name, factory] : factories_)
    names.push_back(name);
  return names;
}

std::string specfem::injection::MethodRegistry::unknown_method_message(
    const std::string_view name) const {
  std::string message = "Unknown injection method '" + std::string(name) +
                        "'. Registered methods:";
  if (factories_.empty())
    message += " (none)";
  for (const auto &[registered, factory] : factories_)
    message += " " + registered;
  return message + ".";
}

const specfem::injection::MethodRegistry &
specfem::injection::default_method_registry() {
  static const MethodRegistry registry = [] {
    MethodRegistry built_in;
    built_in.add<PointwiseSurfaceMethod<StaceyInjection>>();
    return built_in;
  }();
  return registry;
}
