#include "../test_fixture/test_fixture.hpp"
#include "algorithms/locate_point.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::wavefield::simulation_field WavefieldType>
void check_store(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  specfem::assembly::sources<DimensionTag> &sources = assembly.sources;
  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int ngllx = assembly.mesh.element_grid.ngllx;

  // the structured binding ([element_indices, source_indices]) is not
  // supported by the intel compiler
  const auto elements_and_sources = sources.get_sources_on_device(
      MediumTag, PropertyTag, BoundaryTag, WavefieldType);
  const Kokkos::View<int *, Kokkos::DefaultExecutionSpace> &element_indices =
      std::get<0>(elements_and_sources);
  const Kokkos::View<int *, Kokkos::DefaultExecutionSpace> &source_indices =
      std::get<1>(elements_and_sources);

  const int nelements = element_indices.size();

  constexpr int num_components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  if (nelements == 0) {
    return;
  }

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store", nelements);

  const auto h_values_to_store = Kokkos::create_mirror_view(values_to_store);

  for (int i = 0; i < nelements; i++) {
    h_values_to_store(i) = 1.0 + i;
  }

  Kokkos::deep_copy(values_to_store, h_values_to_store);

  using PointSourceType =
      specfem::point::source<DimensionTag, MediumTag, WavefieldType>;
  using mapped_chunk_index_type =
      specfem::point::mapped_index<DimensionTag, false>;
  Kokkos::parallel_for(
      "check_store_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { nelements, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        // element indices and source indices from elements_and_sources
        const int ielement = element_indices(i);
        const int isource = source_indices(i);

        const auto index =
            specfem::point::index<DimensionTag, false>(ielement, iz, ix);
        const auto mapped_iterator_index =
            mapped_chunk_index_type(index, isource);
        specfem::datatype::VectorPointViewType<type_real, num_components, false>
            stf;
        specfem::datatype::VectorPointViewType<type_real, num_components, false>
            lagrange_interpolant;
        for (int ic = 0; ic < num_components; ic++) {
          stf(ic) = values_to_store(i);
          lagrange_interpolant(ic) = values_to_store(i);
        }
        PointSourceType point(stf, lagrange_interpolant);
        specfem::assembly::store_on_device(mapped_iterator_index, point,
                                           sources);
      });

  Kokkos::fence();
}

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::wavefield::simulation_field WavefieldType>
void check_load(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  specfem::assembly::sources<DimensionTag> &sources = assembly.sources;
  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int ngllx = assembly.mesh.element_grid.ngllx;

  const auto elements_and_sources = sources.get_sources_on_device(
      MediumTag, PropertyTag, BoundaryTag, WavefieldType);
  const Kokkos::View<int *, Kokkos::DefaultExecutionSpace> &element_indices =
      std::get<0>(elements_and_sources);
  const Kokkos::View<int *, Kokkos::DefaultExecutionSpace> &source_indices =
      std::get<1>(elements_and_sources);

  const int nelements = element_indices.size();

  constexpr int num_components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store", nelements);

  auto h_values_to_store = Kokkos::create_mirror_view(values_to_store);

  for (int i = 0; i < nelements; i++) {
    h_values_to_store(i) = 1.0 + i;
  }

  Kokkos::deep_copy(values_to_store, h_values_to_store);

  using PointSourceType =
      specfem::point::source<DimensionTag, MediumTag, WavefieldType>;

  using mapped_chunk_index_type =
      specfem::point::mapped_index<DimensionTag, false>;

  Kokkos::View<PointSourceType ***, Kokkos::DefaultExecutionSpace>
      point_sources("point_sources", ngllz, ngllx, nelements);

  auto h_point_sources = Kokkos::create_mirror_view(point_sources);

  Kokkos::parallel_for(
      "check_load_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { nelements, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        // element indices and source indices from elements_and_sources
        const int ielement = element_indices(i);
        const int isource = source_indices(i);

        const auto index =
            specfem::point::index<DimensionTag, false>(ielement, iz, ix);

        const auto mapped_iterator_index =
            mapped_chunk_index_type(index, isource);

        PointSourceType point;
        specfem::assembly::load_on_device(mapped_iterator_index, sources,
                                          point);

        point_sources(iz, ix, i) = point;
      });

  Kokkos::fence();
  Kokkos::deep_copy(h_point_sources, point_sources);

  for (int i = 0; i < nelements; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const auto &point_source = h_point_sources(iz, ix, i);
        for (int ic = 0; ic < num_components; ic++) {
          const auto stf = point_source.stf(ic);
          const auto expected = h_values_to_store(i);
          if (expected != stf) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << i << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << stf << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }

          const auto lagrange_interpolant =
              point_source.lagrange_interpolant(ic);
          if (expected != lagrange_interpolant) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << i << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << lagrange_interpolant << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }
}

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
void check_assembly_source_construction(
    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim2> > > &sources,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int ngllx = assembly.mesh.element_grid.ngllx;

  constexpr auto components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  using PointSourceType =
      specfem::point::source<DimensionTag, MediumTag,
                             specfem::wavefield::simulation_field::forward>;

  const int nsources = sources.size();
  for (int isource = 0; isource < nsources; isource++) {
    const auto &source = sources[isource];
    const auto coord = source->get_global_coordinates();

    const auto lcoord = specfem::algorithms::locate_point(coord, assembly.mesh);

    source->set_local_coordinates(lcoord);
    source->set_medium_tag(assembly.element_types.get_medium_tag(lcoord.ispec));

    if (assembly.element_types.get_medium_tag(lcoord.ispec) != MediumTag) {
      continue;
    }

    Kokkos::View<type_real ***, Kokkos::LayoutRight,
                 Kokkos::DefaultHostExecutionSpace>
        source_array("source_array", components,
                     assembly.mesh.element_grid.ngllz,
                     assembly.mesh.element_grid.ngllx);

    specfem::assembly::compute_source_array(
        source, assembly.mesh, assembly.jacobian_matrix, source_array);
    Kokkos::View<type_real **, Kokkos::LayoutRight,
                 Kokkos::DefaultHostExecutionSpace>
        stf("stf", 1, components);

    source->compute_source_time_function(1.0, 0.0, 1, stf);
    using mapped_chunk_index_type =
        specfem::point::mapped_index<DimensionTag, false>;

    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        specfem::point::index<DimensionTag, false> index(lcoord.ispec, iz, ix);
        const auto mapped_iterator_index =
            mapped_chunk_index_type(index, isource);
        PointSourceType point;
        specfem::assembly::load_on_host(mapped_iterator_index, assembly.sources,
                                        point);

        for (int ic = 0; ic < components; ic++) {
          const auto lagrange_interpolant = point.lagrange_interpolant(ic);
          const auto expected = source_array(ic, iz, ix);
          if (lagrange_interpolant != expected) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << lcoord.ispec << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << lagrange_interpolant << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }
        }

        for (int ic = 0; ic < components; ic++) {
          const auto computed_stf = point.stf(ic);
          const auto expected_stf = stf(0, ic);
          if (computed_stf != expected_stf) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << lcoord.ispec << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << computed_stf << "\n"
                    << "  expected = " << expected_stf;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }
}

void test_assembly_source_construction(
    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim2> > > &sources,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        check_assembly_source_construction<_dimension_tag_, _medium_tag_>(
            sources, assembly);
      })
}

void test_sources(specfem::assembly::assembly<specfem::dimension::type::dim2>
                      &assembly){ FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T),
     PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
     BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                  COMPOSITE_STACEY_DIRICHLET)),
    {
      check_store<_dimension_tag_, _medium_tag_, _property_tag_, _boundary_tag_,
                  specfem::wavefield::simulation_field::forward>(assembly);
      check_load<_dimension_tag_, _medium_tag_, _property_tag_, _boundary_tag_,
                 specfem::wavefield::simulation_field::forward>(assembly);
    }) }

TEST_F(Assembly2D, sources) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    auto sources = std::get<2>(parameters);
    specfem::assembly::assembly<specfem::dimension::type::dim2> assembly =
        std::get<5>(parameters);

    try {
      test_assembly_source_construction(sources, assembly);
      test_sources(assembly);

      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << Test.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
};
