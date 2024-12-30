#pragma once

#include "acoustic_isotropic2d/acoustic_isotropic2d.hpp"
#include "elastic_anisotropic2d/elastic_anisotropic2d.hpp"
#include "elastic_isotropic2d/elastic_isotropic2d.hpp"
#include "enumerations/specfem_enums.hpp"
#include "medium/properties.hpp"
#include "point/properties.hpp"
#include "specfem_setup.hpp"
#include <ostream>
#include <tuple>

namespace specfem {
namespace medium {

/**
 * @brief Material properties for a given medium and property
 *
 * @tparam MediumTag Medium tag for the material
 * @tparam PropertyTag Property tag for the material
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
class material : public specfem::material::properties<MediumTag, PropertyTag> {
public:
  constexpr static auto medium_tag = MediumTag;     ///< Medium tag
  constexpr static auto property_tag = PropertyTag; ///< Property tag

  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Construct a new material object
   *
   */
  material() = default;

  /**
   * @brief Construct a new material object
   *
   * @tparam Args Arguments to forward to the properties constructor
   * @param args Properties of the material (density, wave speeds, etc.)
   */
  template <typename... Args>
  material(Args &&...args)
      : specfem::material::properties<MediumTag, PropertyTag>(
            std::forward<Args>(args)...) {}
  ///@}

  ~material() = default;

  /**
   * @brief Get the medium tag of the material
   *
   * @return constexpr specfem::element::medium_tag Medium tag
   */
  constexpr specfem::element::medium_tag get_type() const {
    return medium_tag;
  };
};

} // namespace medium
} // namespace specfem
