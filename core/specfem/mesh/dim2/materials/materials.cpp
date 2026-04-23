#include "materials.hpp"
#include "specfem/enums.hpp"
#include "specfem/logger.hpp"
#include "specfem/tag_dispatch/for_each.hpp"

void specfem::mesh::materials<specfem::element::dimension_tag::dim2>::print()
    const {
  std::ostringstream message;
  message << "Total number of materials: " << this->n_materials << "\n";
  specfem::tag_dispatch::for_each(
      decltype(combinations){}, [this, &message]<typename TagsType>() {
        const auto &material_container =
            this->material_containers.template get<TagsType>();
        if (material_container.n_materials > 0) {
          message << "Material Type: \n"
                  << "\t Medium Tag: "
                  << specfem::element::to_string(TagsType::medium_tag) << "\n"
                  << "\tProperty Tag: "
                  << specfem::element::to_string(TagsType::property_tag) << "\n"
                  << "\tAttenuation Tag: "
                  << specfem::element::to_string(TagsType::attenuation_tag)
                  << "\n";
          for (int i = 0; i < material_container.n_materials; ++i) {
            message << "Material Index: " << i << "\n";
            message << material_container.element_materials[i].print();
          }
        }
      });

  specfem::Logger::info(message.str());

  return;
}
