#pragma once

template <typename Tags>
auto compute_derivatives_test_impl::ComputeDerivativesTest<
    Tags>::make_assembly() -> specfem::assembly::assembly<dimension> {
  if constexpr (dimension == specfem::element::dimension_tag::dim3) {
    return {};
  } else {
    // The 2D assembly has no default constructor. Bootstrap it through its
    // public constructor using an existing tiny mesh, then replace the kernel
    // inputs with the same reference elements used by the 3D cases below.
    const auto mesh =
        specfem::io::read_2d_mesh("data/dim2/3_elem_nonconforming/database.bin",
                                  specfem::enums::elastic_wave::psv,
                                  specfem::enums::electromagnetic_wave::te,
                                  specfem::attenuation::Setup{});
    const specfem::quadrature::quadratures quadrature(
        specfem::quadrature::gll::gll{});
    std::vector<std::shared_ptr<specfem::sources::source<dimension>>> sources;
    return { mesh,
             quadrature,
             sources,
             {},
             {},
             0.0,
             dt,
             1,
             1,
             1,
             specfem::simulation::type::combined,
             false,
             nullptr };
  }
}

template <typename Tags>
template <typename Functor>
void compute_derivatives_test_impl::ComputeDerivativesTest<
    Tags>::for_each_point(Functor &&functor) const {
  for (int ispec = 0; ispec < nspec; ++ispec) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int ix = 0; ix < ngll; ++ix) {
        if constexpr (dimension == specfem::element::dimension_tag::dim2) {
          functor(Index(ispec, iz, ix));
        } else {
          for (int iy = 0; iy < ngll; ++iy) {
            functor(Index(ispec, iz, iy, ix));
          }
        }
      }
    }
  }
}

template <typename Tags>
auto compute_derivatives_test_impl::ComputeDerivativesTest<
    Tags>::make_properties() -> Properties {
  if constexpr (medium == specfem::element::medium_tag::acoustic) {
    return Properties(0.5, 5.0);
  } else if constexpr (medium == specfem::element::medium_tag::poroelastic) {
    // Positive Biot moduli, nonsingular permeability, and nonzero porosity.
    return Properties(0.2, 3.0, 1.0, 2.0, 2.0, 8.0, 1.0, 3.0, 1.0, 0.0, 1.0,
                      0.1);
  } else if constexpr (property == specfem::element::property_tag::isotropic) {
    return Properties(5.0, 2.0, 3.0);
  } else {
    Properties result(static_cast<type_real>(0.0));
    result.rho() = 3.0;
    result.c11() = 8.0;
    result.c33() = 10.0;
    result.c13() = 2.0;
    result.c12() = 2.0;
    result.c23() = 2.0;
    result.c55() = 2.0;
    if constexpr (dimension == specfem::element::dimension_tag::dim3) {
      result.c22() = 9.0;
      result.c44() = 2.0;
      result.c66() = 2.0;
    }
    return result;
  }
}

template <typename Tags>
void compute_derivatives_test_impl::ComputeDerivativesTest<Tags>::SetUp() {
  auto &mesh = assembly.mesh;
  mesh.nspec = nspec;
  mesh.ngnod = dimension == specfem::element::dimension_tag::dim2 ? 4 : 8;
  using Mapping =
      typename specfem::assembly::mesh<dimension>::IndexMappingViewType;
  if constexpr (dimension == specfem::element::dimension_tag::dim2) {
    mesh.element_grid = { ngll, ngll };
    mesh.nglob = nspec * ngll * ngll;
    mesh.index_mapping = Mapping("index_mapping", nspec, ngll, ngll);
    assembly.jacobian_matrix = { nspec, ngll, ngll };
  } else {
    mesh.element_grid = { ngll, ngll, ngll };
    mesh.nglob = nspec * ngll * ngll * ngll;
    mesh.index_mapping = Mapping("index_mapping", nspec, ngll, ngll, ngll);
    assembly.jacobian_matrix = { nspec, ngll, ngll, ngll };
    Kokkos::deep_copy(assembly.jacobian_matrix.etay, 1.0);
  }
  mesh.h_index_mapping = Kokkos::create_mirror_view(mesh.index_mapping);
  int iglob = 0;
  for_each_point([&](const Index &index) {
    if constexpr (dimension == specfem::element::dimension_tag::dim2) {
      mesh.h_index_mapping(index.ispec, index.iz, index.ix) = iglob++;
    } else {
      mesh.h_index_mapping(index.ispec, index.iz, index.iy, index.ix) = iglob++;
    }
  });
  Kokkos::deep_copy(mesh.index_mapping, mesh.h_index_mapping);

  // Disjoint reference elements: identity Jacobians and normal GLL operators.
  Kokkos::deep_copy(assembly.jacobian_matrix.xix, 1.0);
  Kokkos::deep_copy(assembly.jacobian_matrix.gammaz, 1.0);
  Kokkos::deep_copy(assembly.jacobian_matrix.jacobian, 1.0);
  static_cast<specfem::assembly::mesh_impl::quadrature<dimension> &>(mesh) =
      specfem::assembly::mesh_impl::quadrature<dimension>(
          specfem::quadrature::quadratures(specfem::quadrature::gll::gll{}));

  Kokkos::View<specfem::element::medium_tag *,
               Kokkos::DefaultHostExecutionSpace>
      media("medium_tags", nspec);
  Kokkos::View<specfem::element::property_tag *,
               Kokkos::DefaultHostExecutionSpace>
      properties("property_tags", nspec);
  Kokkos::View<specfem::element::attenuation_tag *,
               Kokkos::DefaultHostExecutionSpace>
      attenuation("attenuation_tags", nspec);
  Kokkos::View<specfem::element::boundary_tag *,
               Kokkos::DefaultHostExecutionSpace>
      boundaries("boundary_tags", nspec);
  Kokkos::deep_copy(media, medium);
  Kokkos::deep_copy(properties, property);
  Kokkos::deep_copy(attenuation, Tags::attenuation_tag);
  Kokkos::deep_copy(boundaries, specfem::element::boundary_tag::none);
  assembly.element_types = { nspec,      mesh.element_grid, media,
                             properties, attenuation,       boundaries };
  assembly.fields = { mesh, assembly.element_types,
                      specfem::simulation::type::combined };
  assembly.kernels = { assembly.element_types };

  // Kernels read material properties and wavefields, not attenuation memory.
  using DomainProperties =
      specfem::assembly::impl::domain_properties<dimension, medium, property>;
  using ContainerTags = specfem::tags::Tags<dimension, medium, property>;
  auto &container = assembly.properties.value.template get<ContainerTags>();
  container = DomainProperties(nspec, mesh.element_grid);
  container.element_range = { 0, nspec };
  assembly.properties.nspec = nspec;
  assembly.properties.element_grid = mesh.element_grid;
  assembly.properties.element_ranges.template get<ContainerTags>() = { 0,
                                                                       nspec };
  for_each_point([&](const Index &index) {
    specfem::assembly::store_on_host(index, make_properties(),
                                     assembly.properties);
  });
  assembly.properties.copy_to_device();
}

template <typename Tags>
void compute_derivatives_test_impl::ComputeDerivativesTest<Tags>::set_fields(
    const type_real acceleration, const type_real displacement) {
  const auto &adjoint = assembly.fields.adjoint.template get_field<medium>();
  const auto &backward = assembly.fields.backward.template get_field<medium>();
  Kokkos::deep_copy(adjoint.get_field_dot_dot(), acceleration);
  Kokkos::deep_copy(backward.get_field(), displacement);
}

template <typename Tags>
void compute_derivatives_test_impl::ComputeDerivativesTest<Tags>::compute() {
  specfem::compute::compute_derivatives<ngll, specfem::tags::Tags<dimension>>(
      assembly, dt);
  assembly.kernels.copy_to_host();
}

template <typename Tags>
type_real
compute_derivatives_test_impl::ComputeDerivativesTest<Tags>::expected_kernel() {
  const auto properties = make_properties();
  if constexpr (medium == specfem::element::medium_tag::acoustic) {
    return 6.0 * dt / properties.kappa();
  } else if constexpr (medium == specfem::element::medium_tag::poroelastic) {
    return -12.0 * dt * properties.rho_bar();
  } else if constexpr (dimension == specfem::element::dimension_tag::dim3 &&
                       property ==
                           specfem::element::property_tag::anisotropic) {
    // 3D anisotropic kernels use absolute, rather than relative, parameters.
    return -18.0 * dt;
  } else {
    constexpr int components =
        specfem::element::attributes<dimension, medium>::components;
    return -6.0 * dt * components * properties.rho();
  }
}

template <typename Tags>
type_real
compute_derivatives_test_impl::ComputeDerivativesTest<Tags>::selected_kernel(
    const Kernels &kernels) {
  if constexpr (medium == specfem::element::medium_tag::acoustic) {
    return kernels.kappa();
  } else if constexpr (medium == specfem::element::medium_tag::poroelastic) {
    return kernels.rhot();
  } else {
    return kernels.rho();
  }
}
