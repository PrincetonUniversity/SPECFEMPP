#pragma once

#include "enumerations/specfem_enums.hpp"
#include "specfem/point.hpp"
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
          specfem::element::property_tag PropertyTag, typename Enable = void>
struct material;

// /**
//  * @brief Material properties for a given medium and property
//  *
//  * @tparam MediumTag Medium tag for the material
//  * @tparam PropertyTag Property tag for the material
//  */
// template <specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag>
// class impl_material;

// /**
//  * @brief Material properties for a given medium and property
//  *
//  * @tparam MediumTag Medium tag for the material
//  * @tparam PropertyTag Property tag for the material
//  */
// template <specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag>
// class material : public impl_material<MediumTag, PropertyTag> {
// public:
//   constexpr static auto medium_tag = MediumTag;     ///< Medium tag
//   constexpr static auto property_tag = PropertyTag; ///< Property tag

//   /**
//    * @name Constructors
//    */
//   ///@{

//   /**
//    * @brief Construct a new material object
//    *
//    */
//   material() = default;

//   /**
//    * @brief Construct a new material object
//    *
//    * @tparam Args Arguments to forward to the properties constructor
//    * @param args Properties of the material (density, wave speeds, etc.)
//    */
//   template <typename... Args>
//   material(Args &&...args)
//       : specfem::medium::impl_material<MediumTag, PropertyTag>(
//             std::forward<Args>(args)...) {}
//   ///@}

//   ~material() = default;

//   /**
//    * @brief Get the medium tag of the material
//    *
//    * @return constexpr specfem::element::medium_tag Medium tag
//    */
//   constexpr specfem::element::medium_tag get_type() const {
//     return medium_tag;
//   };
// };

} // namespace medium
} // namespace specfem

#include "dim2/acoustic/isotropic/material.hpp"
#include "dim2/elastic/anisotropic/material.hpp"
#include "dim2/elastic/isotropic/material.hpp"
#include "dim2/elastic/isotropic_cosserat/material.hpp"
#include "dim2/electromagnetic/isotropic/material.hpp"
#include "dim2/poroelastic/isotropic/material.hpp"

#include "dim3/acoustic/isotropic/material.hpp"
#include "dim3/elastic/isotropic/material.hpp"
