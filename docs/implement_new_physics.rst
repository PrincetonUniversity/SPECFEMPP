Implementing new physics -- Example (Anisotropy)
================================================

Glossary for original specfem devs
----------------------------------


Create new material class
-------------------------

material/material.hpp

Create material reading
-----------------------

IO/mesh/impl/read_materials.hpp

Create new point property & Material Assembly
---------------------------------------------

Store values related to GLL point wihtin the assembly class.

Files that need updating

compute/properties/properties.hpp

We need to hold different properties and update the corresponding
load_on_device function.
compute/properties/impl/properties_container.hpp


Create Medium class
-------------------

How to compute forces and stresses in the medium.
medium/elastic_isotropic/elasticisotropic.hpp

Create Examples
---------------

- Anisotropic_zinc_crystals
- Anisotropy_isotropy


Kernels for the medium
----------------------

How do we compute misfit kernels.
medium/elastic_isotropic/elasticisotropic.hpp
