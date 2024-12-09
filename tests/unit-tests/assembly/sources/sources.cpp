
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
void check_store(specfem::compute::assembly &assembly) {

  const auto elements =
      sources.get_elements_on_device(MediumTag, WavefieldType);

  constexpr int num_components =
      specfem::element::attributes<Dimension, MediumTag>::components();

  if (elements.size() == 0) {
    return;
  }

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store", elements.size());

  for (int i = 0; i < elements.size(); i++) {
    values_to_store(i) = 1.0 + i;
  }

  using PointType =
      specfem::point::sources<Dimension, MediumTag, WavefieldType>;

  Kokkos::parallel_for(
      "check_store_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { N, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = elements(i);
        auto &kernels_l = kernels;

        const auto index =
            specfem::point::index<Dimension, false>(ielement, iz, ix);
        specfem::datatype::ScalarPointViewType<type_real, num_components, false>
            stf;
        specfem::datatype::ScalarPointViewType<type_real, num_components, false>
            lagrange_interpolant;
        for (int ic = 0; ic < num_components; ic++) {
          stf(ic) = 1.0;
          lagrange_interpolant(ic) = 1.0;
        }
        PointType point(stf, lagrange_interpolant);
        specfem::compute::store_on_device(index, point, sources);
      });

  Kokkos::fence();
}

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
void check_load(specfem::compute::assembly &assembly) {

  const auto elements =
      sources.get_elements_on_device(MediumTag, WavefieldType);

  constexpr int num_components =
      specfem::element::attributes<Dimension, MediumTag>::components();

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store", elements.size());

  auto h_values_to_store = Kokkos::create_mirror_view(values_to_store);

  for (int i = 0; i < elements.size(); i++) {
    h_values_to_store(i) = 1.0 + i;
  }

  Kokkos::deep_copy(values_to_store, h_values_to_store);

  using PointType =
      specfem::point::sources<Dimension, MediumTag, WavefieldType>;

  Kokkos::View<PointType **[N], Kokkos::DefaultExecutionSpace> point_sources(
      "point_sources", ngllz, ngllx);

  auto h_point_sources = Kokkos::create_mirror_view(point_sources);

  Kokkos::parallel_for(
      "check_load_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { N, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = elements(i);
        auto &kernels_l = kernels;

        const auto index =
            specfem::point::index<Dimension, false>(ielement, iz, ix);

        PointType point;
        specfem::compute::load_on_device(index, sources, point);

        point_sources(iz, ix, i) = point;
      });

  Kokkos::fence();
  Kokkos::deep_copy(h_point_sources, point_sources);

  for (int i = 0; i < N; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const auto &point_kernel = h_point_sources(iz, ix, i);
        for (int ic = 0; ic < num_components; ic++) {
          const auto stf = point_kernel.stf(ic);
          const auto expected = values_to_store(i);
          if (computed != stf) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << i << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << computed << "\n"
                    << "  expected = " << stf;
            throw std::runtime_error(message.str());
          }

          const auto lagrange_interpolant =
              point_kernel.lagrange_interpolant(ic);
          if (computed != lagrange_interpolant) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << i << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << computed << "\n"
                    << "  expected = " << lagrange_interpolant;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }
}

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
void check_load(specfem::compute::assembly &assembly) {}

void test_kernels(specfem::compute::assembly &assembly) {

  auto &sources = assembly.sources;

  check_store<specfem::dimension::type::dim2,
              specfem::element::medium_tag::elastic,
              specfem::wavefield::simulation_field::forward>(sources);

  check_load<specfem::dimension::type::dim2,
             specfem::element::medium_tag::elastic,
             specfem::wavefield::simulation_field::forward>(sources);

  check_store<specfem::dimension::type::dim2,
              specfem::element::medium_tag::acoustic,
              specfem::wavefield::simulation_field::forward>(sources);

  check_load<specfem::dimension::type::dim2,
             specfem::element::medium_tag::acoustic,
             specfem::wavefield::simulation_field::forward>(sources);
}

TEST_F(ASSEMBLY, sources) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    auto assembly = std::get<1>(parameters);

    try {
      test_kernels(assembly);

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
}
