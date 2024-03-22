#ifndef _MATERIAL_HPP
#define _MATERIAL_HPP

#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
#include "point/properties.hpp"
#include "properties.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include <ostream>
#include <tuple>

namespace specfem {
namespace material {

/**
 * @brief Base material class
 *
 */
template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
class material : public specfem::material::properties<type, property> {
public:
  constexpr static auto value_type = type;
  constexpr static auto property_type = property;
  /**
   * @brief Construct a new material object
   *
   */
  material() = default;

  /**
   * @brief Destroy the material object
   *
   */
  ~material() = default;

  template <
      specfem::element::medium_tag U = value_type,
      std::enable_if_t<U == specfem::element::medium_tag::elastic, int> = 0>
  material(const type_real &density, const type_real &cs, const type_real &cp,
           const type_real &Qkappa, const type_real &Qmu,
           const type_real &compaction_grad)
      : specfem::material::properties<type, property>(density, cs, cp, Qkappa,
                                                      Qmu, compaction_grad){};

  template <
      specfem::element::medium_tag U = value_type,
      std::enable_if_t<U == specfem::element::medium_tag::acoustic, int> = 0>
  material(const type_real &density, const type_real &cp,
           const type_real &Qkappa, const type_real &Qmu,
           const type_real &compaction_grad)
      : specfem::material::properties<type, property>(density, cp, Qkappa, Qmu,
                                                      compaction_grad){};

  /**
   * @brief Get the material type
   *
   * @return specfem::enums::material::type
   */
  constexpr specfem::element::medium_tag get_type() const { return type; };
};

} // namespace material
} // namespace specfem

#endif
