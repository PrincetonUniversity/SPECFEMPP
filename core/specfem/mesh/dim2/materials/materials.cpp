#include "materials.hpp"
#include "enumerations/interface.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros.hpp"

void specfem::mesh::materials<specfem::element::dimension_tag::dim2>::print()
    const {
  std::ostringstream message;
  message << "Total number of materials: " << this->n_materials << "\n";
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T,
                  ELECTROMAGNETIC_TE),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       ATTENUATION_TAG(NONE, CONSTANT_ISOTROPIC)),
      CAPTURE() {
        const auto &material_container =
            this->get_container<_medium_tag_, _property_tag_,
                                _attenuation_tag_>();

        if (material_container.n_materials > 0) {

          message << "Material Type: \n"
                  << "\t Medium Tag: "
                  << specfem::element::to_string(_medium_tag_) << "\n"
                  << "\tProperty Tag: "
                  << specfem::element::to_string(_property_tag_) << "\n"
                  << "\tAttenuation Tag: "
                  << specfem::element::to_string(_attenuation_tag_) << "\n";

          for (int i = 0; i < material_container.n_materials; ++i) {
            message << "Material Index: " << i << "\n";
            message << material_container.element_materials[i].print();
          }
        }
      })

  specfem::Logger::info(message.str());

  return;
}
