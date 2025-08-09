#pragma once
#include "enumerations/dimension.hpp"
#include "interface_modules.hpp"
#include "interface_modules/edge_container.hpp"

namespace specfem::assembly::interface {
namespace module {
struct type_util {
  static constexpr bool is_edge_container(const type &module) {
    return module == type::SINGLE_EDGE_CONTAINER ||
           module == type::DOUBLE_EDGE_CONTAINER;
  }
  template <specfem::dimension::type DimensionType, type module>
  using edge_container_from_enum = std::conditional_t<
      module == type::SINGLE_EDGE_CONTAINER,
      specfem::assembly::interface::module::single_edge_container<
          DimensionType>,
      std::conditional_t<module == type::DOUBLE_EDGE_CONTAINER,
                         specfem::assembly::interface::module::
                             double_edge_container<DimensionType>,
                         void> >;

  // hold mixins in a template argument
  template <typename... values> struct mixins_pack {};

  /* =============================================================================================
   * pass mixins through deduction. We are getting the module
   * list from `Modules...` inside this:
   * `container_args<...>::module_type_pack == mixins_pack<Modules...>`.
   *
   * The only public facing part is moduled_interface_container<container_args>
   */
private:
  template <typename container_args,
            typename = typename container_args::module_type_pack>
  struct moduled_interface_container_impl;

  template <typename container_args, typename... Modules>
  struct moduled_interface_container_impl<container_args,
                                          mixins_pack<Modules...> >
      : public Modules... {
  public:
    static constexpr specfem::dimension::type DimensionType =
        container_args::DimensionType;
    moduled_interface_container_impl(const initializer &init)
        : Modules(init)... {}
    moduled_interface_container_impl()
        : moduled_interface_container_impl(initializer()) {}

    template <int medium, bool access_from_host = false, typename... Args>
    inline auto &index_at(Args... args) {
      return container_args::EdgeContainerType::template get_edge_index_view<
          medium, access_from_host>()(args...);
    }
    template <int medium, bool access_from_host = false, typename... Args>
    inline auto &edge_type_at(Args... args) {
      return container_args::EdgeContainerType::template get_edge_type_view<
          medium, access_from_host>()(args...);
    }
  };

public:
  template <typename containerspec>
  using moduled_interface_container =
      moduled_interface_container_impl<containerspec>;
  // =============================================================================================
};

template <specfem::dimension::type DimensionType_, type... modules>
struct container_args {
public:
  static constexpr specfem::dimension::type DimensionType = DimensionType_;

private:
  // recursive template for determining mixins
  template <type...> struct partial_processed_args {
    using EdgeContainerType = void;
    static constexpr bool defines_edge_container = false;
    static constexpr bool incompatibly_defined = false;
  };

  // process one at a time
  template <type current_type, type... remaining_to_process>
  struct partial_processed_args<current_type, remaining_to_process...> {
    using prior_partial = partial_processed_args<remaining_to_process...>;

    using EdgeContainerType = std::conditional_t<
        type_util::is_edge_container(current_type),
        type_util::edge_container_from_enum<DimensionType, current_type>,
        typename prior_partial::EdgeContainerType>;

    static constexpr bool defines_edge_container =
        prior_partial::defines_edge_container ||
        type_util::is_edge_container(current_type);
    static constexpr bool incompatibly_defined =
        // extend prior: once incompatible, always incompatible
        prior_partial::incompatibly_defined ||
        // incompatible if edge container was multiply defined
        (prior_partial::defines_edge_container &&
         type_util::is_edge_container(current_type));
  };

  using processed_pack = partial_processed_args<modules...>;

public:
  static constexpr bool well_formed = !processed_pack::incompatibly_defined;
  using EdgeContainerType = typename processed_pack::EdgeContainerType;

  using module_type_pack = type_util::mixins_pack<EdgeContainerType>;
};
} // namespace module

/**
 * @brief The container paralleling specfem::assembly::interface_container.
 * This implements our interfaces using a mixin approach, as different schemes
 * need different information.
 *
 * @tparam DimensionType - dimension of the domain
 * @tparam Modules - A pack of module types.
 */
template <specfem::dimension::type DimensionType, module::type... Modules>
using moduled_interface_container =
    module::type_util::moduled_interface_container<
        module::container_args<DimensionType, Modules...> >;
} // namespace specfem::assembly::interface
