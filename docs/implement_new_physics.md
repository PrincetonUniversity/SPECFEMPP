# Implementing new physics -- Example (Anisotropy)


## Glossary for original specfem devs

Some terms that are very specific to the C++ implementation

- Member
- Team
- Iterator
- ...



## Create new material class

- [ ] Add `property` enumeration `anisotropic` to `include/enumerations/medium.hpp`
- [ ] Modify `include/material/material.hpp`.
- [ ] Add new material to `include/mesh/materials/materials.hpp::materials`.
- [ ] Add new material to `include/mesh/materials/materials.hpp::materials::operator[]`.
- [ ] Add material/property template specification
  ```cpp
  template <>
  class properties<specfem::element::medium_tag::elastic, specfem::element::property_tag::anisotropic>
  ```
  to file `include/material/elastic_properties.hpp`.
- [ ] Add to file `src/mesh/materials/materials.cpp`
- [ ] For new media (poro-elastic, e.g.) add coupled interface description
      between the respective interfaces
  * [ ] Add to file `include/mesh/coupled_interfaces.hpp`
  * [ ] Add to file `include/mesh/coupled_interfaces.tpp`
  * [ ] Add to file `src/mesh/coupled_interfaces/coupled_interfaces.cpp`
  * [ ] Add to file `src/mesh/coupled_interfaces/interface_container.cpp`


## Create material reading

In the `IO/mesh/impl/read_materials.cpp` a section needs to be added to
add the correct values to the new material this is done in the local function
`read_materials()`, which returns the specification. This would use the
implemented material class from above to populate the materials

- [ ] Add conditional for new material properties in
      `IO/mesh/impl/read_materials.cpp::read_materials()` from line `73`-ish


## Create new point property & Material Assembly

Store values related to GLL point wihtin the assembly class.

Files that need updating

- [ ] Add new material `load_device_properties` to `compute/properties/properties.hpp`
- [ ] Add new material `load_host_properties` to `compute/properties/properties.hpp`
- [ ] Add new material property containers and corresponding `load_device_

> **_NOTE:_** It's really the
`specfem::compute::properties::material_property<type, property>::medium_property()`
> constructor defined in `include/compute/properties/impl/material_properties.tpp`
> that assigns properties to the GLL points from the input mesh.

We need to hold different properties and update the corresponding
`load_on_device` function.
`compute/properties/impl/properties_container.hpp`




## Create Medium class

How to compute forces and stresses in the medium.

`medium/elastic_isotropic/elastic_isotropic.hpp`


## Create Examples

From `specfem2d`:
- Anisotropic_zinc_crystals
- Anisotropy_isotropy


Kernels for the medium
----------------------

How do we compute misfit kernels.
medium/elastic_isotropic/elasticisotropic.hpp
