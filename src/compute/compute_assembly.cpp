
#include "compute/assembly/assembly.hpp"
#include "mesh/mesh.hpp"

namespace {

template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::component Component>
class field_type_parameters;

template <specfem::element::medium_tag MediumTag>
class field_type_parameters<MediumTag,
                            specfem::wavefield::component::displacement> {
public:
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto store_displacement = true;
  constexpr static auto store_velocity = false;
  constexpr static auto store_acceleration = false;
  constexpr static auto store_mass_matrix = false;
  constexpr static auto num_components = 2;
};

template <specfem::element::medium_tag MediumTag>
class field_type_parameters<MediumTag,
                            specfem::wavefield::component::velocity> {
public:
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto store_displacement = false;
  constexpr static auto store_velocity = true;
  constexpr static auto store_acceleration = false;
  constexpr static auto store_mass_matrix = false;
  constexpr static auto num_components = 2;
};

template <specfem::element::medium_tag MediumTag>
class field_type_parameters<MediumTag,
                            specfem::wavefield::component::acceleration> {
public:
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto store_displacement = false;
  constexpr static auto store_velocity = false;
  constexpr static auto store_acceleration = true;
  constexpr static auto store_mass_matrix = false;
  constexpr static auto num_components = 2;
};

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::wavefield::component Component>
class helper;

template <specfem::element::property_tag PropertyTag,
          specfem::wavefield::component Component>
class helper<specfem::element::medium_tag::elastic, PropertyTag, Component> {
public:
  using field_parameters =
      field_type_parameters<specfem::element::medium_tag::elastic, Component>;
  helper(const specfem::compute::assembly &assembly,
         Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                      Kokkos::DefaultExecutionSpace>
             wavefield_on_entire_grid)
      : assembly(assembly), wavefield_on_entire_grid(wavefield_on_entire_grid) {
    return;
  }

  void compute_wavefield() {
    const auto buffer = assembly.fields.buffer;

    const int nspec = assembly.mesh.nspec;
    const int ngllz = assembly.mesh.ngllz;
    const int ngllx = assembly.mesh.ngllx;

    const auto elements =
        assembly.properties.get_elements_on_device(MediumTag, PropertyTag);

    using PointFieldType =
        specfem::point::field<specfem::dimension::type::dim2, MediumTag,
                              field_parameters::store_displacement,
                              field_parameters::store_velocity,
                              field_parameters::store_acceleration,
                              field_parameters::store_mass_matrix, false>;

    using ParallelConfig = specfem::parallel_config::point_config<
        specfem::dimension::type::dim2,
        specfem::datatype::simd<type_real, false>,
        Kokkos::DefaultExecutionSpace>;

    using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfig>;

    ChunkPolicyType chunk_policy(elements, ngllz, ngllx);
    const int nelements = elements.extent(0);

    Kokkos::parallel_for(
        "specfem::domain::impl::kernels::elements::compute_mass_matrix",
        static_cast<const typename ChunkPolicyType::policy_type &>(
            chunk_policy),
        KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
          for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
               tile += ChunkPolicyType::chunk_size * simd_size) {
            const int starting_element_index =
                team.league_rank() * ChunkPolicyType::tile_size * simd_size +
                tile;

            if (starting_element_index >= nelements) {
              break;
            }

            const auto iterator =
                chunk_policy.league_iterator(starting_element_index);

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, iterator.chunk_size()),
                [&](const int i) {
                  const auto iterator_index = iterator(i);
                  const auto index = iterator_index.index;

                  PointFieldType field;

                  specfem::compute::load_on_device(index, buffer, field);

                  for (int icomponent = 0;
                       icomponent < field_parameters::num_components;
                       icomponent++) {
                    wavefield_on_entire_grid(index.ispec, index.iz, index.ix,
                                             icomponent) = field(icomponent);
                  }
                });
          }
        });

    return;
  }

private:
  const specfem::compute::assembly &assembly;
  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid;
};

template <specfem::element::property_tag PropertyTag,
          specfem::wavefield::component Component>
class helper<specfem::element::medium_tag::acoustic, PropertyTag, Component> {
public:
  using field_parameters =
      field_type_parameters<specfem::element::medium_tag::acoustic, Component>;
  helper(const specfem::compute::assembly &assembly,
         Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                      Kokkos::DefaultExecutionSpace>
             wavefield_on_entire_grid)
      : assembly(assembly), wavefield_on_entire_grid(wavefield_on_entire_grid) {
  }

  void compute_wavefield() {
    const auto buffer = assembly.fields.buffer;

    const int nspec = assembly.mesh.nspec;
    const int ngllz = assembly.mesh.ngllz;
    const int ngllx = assembly.mesh.ngllx;

    const auto elements =
        assembly.properties.get_elements_on_device(MediumTag, PropertyTag);
    const int nelements = elements.extent(0);

    using ChunkElementFieldType = specfem::chunk_element::field<
        ParallelConfig::chunk_size, ngll, specfem::dimension::type::dim2,
        MediumTag, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>,
        field_parameters::store_displacement, field_parameters::store_velocity,
        field_parameters::store_acceleration,
        field_parameters::store_mass_matrix, false>;

    using QuadratureType = specfem::element::quadrature<
        ngll, specfem::dimension::type::dim2, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

    using simd = specfem::datatype::simd<type_real, false>;
    using ParallelConfig = specfem::parallel_config::chunk_config<
        specfem::dimension::type::dim2, simd, Kokkos::DefaultExecutionSpace>;
    using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfig>;

    int scratch_size =
        ChunkElementFieldType::shmem_size() + QuadratureType::shmem_size();
    ChunkPolicyType chunk_policy(elements, ngllz, ngllx);

    Kokkos::parallel_for(
        "compute_wavefield",
        chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
          QuadratureType quadrature(team);
          ChunkElementFieldType field(team);

          specfem::compute::load_on_device(team, assembly.mesh.quadratures,
                                           quadrature);

          for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
               tile += ChunkPolicyType::chunk_size * simd_size) {
            const int starting_element_index =
                team.league_rank() * ChunkPolicyType::tile_size * simd_size +
                tile;

            if (starting_element_index >= nelements) {
              break;
            }

            const auto iterator =
                chunk_policy.league_iterator(starting_element_index);
            specfem::compute::load_on_device(team, iterator, buffer, field);
            team.team_barrier();

            const auto &active_field = [&]() {
              if constexpr (Component ==
                            specfem::wavefield::component::displacement) {
                return field.displacement;
              } else if constexpr (Component ==
                                   specfem::wavefield::component::velocity) {
                return field.velocity;
              } else if constexpr (Component == specfem::wavefield::component::
                                                    acceleration) {
                return field.acceleration;
              } else {
                static_assert("component not supported");
              }
            }();

            specfem::algorithms::gradient(
                team, iterator, quadrature.hprime, active_field,
                [&](const typename ChunkPolicyType::iterator_type::index_type
                        &iterator_index,
                    const typename PointFieldDerivatives::ViewType &du) {
                  PointPropertyType point_property;
                  PointFieldDerivativesType field_derivatives(du);

                  PointStressType stress = specfem::element::compute_stress(
                      point_property, field_derivatives);

                  for (int icomponent = 0; icomponent < num_components;
                       icomponent++) {
                    wavefield_on_entire_grid(
                        iterator_index.index.ispec, iterator_index.index.iz,
                        iterator_index.index.ix, icomponent) =
                        stress(icomponent, 0);
                  }
                });
          }
        });

    return;
  }

private:
  const specfem::compute::assembly &assembly;
  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid;
};

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::wavefield::component Component>
void get_wavefield_on_entire_grid(
    const specfem::compute::assembly &assembly,
    Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                 Kokkos::DefaultExecutionSpace>
        wavefield_on_entire_grid) {

  helper<MediumTag, PropertyTag, Component> handle(assembly,
                                                   wavefield_on_entire_grid);

  handle.compute_wavefield();
  return;
}

} // namespace

// template <>
// get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic,
//                              specfem::wavefield::component::displacement>(
//     const specfem::compute::assembly &assembly,
//     Kokkos::View<type_real ****, Kokkos::LayoutLeft,
//                  Kokkos::DefaultExecutionSpace>
//         wavefield_on_entire_grid) {

//   const auto buffer = assembly.fields.buffer;

//   const int nspec = assembly.mesh.nspec;
//   const int ngllz = assembly.mesh.ngllz;
//   const int ngllx = assembly.mesh.ngllx;
//   // Get parameters for the chunk field type
//   using field_parameters = field_parameters<MediumTag, Component>;

//   using simd = specfem::datatype::simd<type_real, false>;
//   using parallel_config = specfem::parallel_config::chunk_config<
//       specfem::dimension::type::dim2, simd, Kokkos::DefaultExecutionSpace>;
//   using ChunkPolicyType = specfem::policy::element_chunk<parallel_config>;

//   using ChunkElementFieldType = specfem::chunk_element::field<
//       ParallelConfig::chunk_size, ngll, specfem::dimension::type::dim2,
//       MediumTag, specfem::kokkos::DevScratchSpace,
//       Kokkos::MemoryTraits<Kokkos::Unmanaged>,
//       field_parameters::store_displacement, field_parameters::store_velocity,
//       field_parameters::store_acceleration,
//       field_parameters::store_mass_matrix, false>;
//   using QuadratureType = specfem::element::quadrature<
//       ngll, specfem::dimension::type::dim2, specfem::kokkos::DevScratchSpace,
//       Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

//   using ScalarPointViewType = specfem::datatype::ScalarPointViewType<
//       type_real, field_parameters::num_components, false>;

//   const auto elements =
//   assembly.properties.get_elements_on_device(MediumTag); const
//   ChunkPolicyType chunk_policy(elements, ngllz, ngllx); const int nelements =
//   elements.extent(0);

//   Kokkos::parallel_for(
//       "get_wavefield_on_entire_grid",
//       chunk_policy.set_scratch_size(
//           0, Kokkos::PerTeam(ChunkElementFieldType::shmem_size())),
//       KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team)
//       {
//         QuadratureType quadrature(team);
//         ChunkElementFieldType field(team);

//         specfem::compute::load_on_device(team, assembly.mesh.quadratures,
//                                          quadrature);

//         for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
//              tile += ChunkPolicyType::chunk_size * simd_size) {
//           const int starting_element_index =
//               team.league_rank() * ChunkPolicyType::tile_size * simd_size +
//               tile;

//           if (starting_element_index >= nelements) {
//             break;
//           }

//           const auto iterator =
//               chunk_policy.league_iterator(starting_element_index);
//           specfem::compute::load_on_device(team, iterator, buffer, field);
//           team.team_barrier();

//           specfem::element::compute_wavefield<Component>(
//               iterator, quadrature, field,
//               [&](const ChunkPolicyType::iterator_type::index_type
//                       &iterator_index,
//                   const ScalarPointViewType &point_field) {
//                 const int ispec = iterator_index.index.ispec;
//                 const int iz = iterator_index.index.iz;
//                 const int ix = iterator_index.index.ix;
//                 for (int icomponent = 0; icomponent < num_components;
//                      icomponent++) {
//                   wavefield_on_entire_grid(ispec, iz, ix) =
//                       point_field(icomponent);
//                 }
//               });
//         }
//       });

//   return;
// }

} // namespace

specfem::compute::assembly::assembly(
    const specfem::mesh::mesh &mesh,
    const specfem::quadrature::quadratures &quadratures,
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const std::vector<std::shared_ptr<specfem::receivers::receiver> >
        &receivers,
    const std::vector<specfem::enums::seismogram::type> &stypes,
    const type_real t0, const type_real dt, const int max_timesteps,
    const int max_sig_step, const specfem::simulation::type simulation) {
  this->mesh = { mesh.tags, mesh.control_nodes, quadratures };
  this->partial_derivatives = { this->mesh };
  this->properties = { this->mesh.nspec,   this->mesh.ngllz, this->mesh.ngllx,
                       this->mesh.mapping, mesh.tags,        mesh.materials };
  this->kernels = { this->mesh.nspec, this->mesh.ngllz, this->mesh.ngllx,
                    this->mesh.mapping, mesh.tags };
  this->sources = { sources,          this->mesh, this->partial_derivatives,
                    this->properties, t0,         dt,
                    max_timesteps };
  this->receivers = { max_sig_step, receivers, stypes, this->mesh };
  this->boundaries = { this->mesh.nspec,   this->mesh.ngllz,
                       this->mesh.ngllx,   mesh,
                       this->mesh.mapping, this->mesh.quadratures,
                       this->properties,   this->partial_derivatives };
  this->coupled_interfaces = { mesh,
                               this->mesh.points,
                               this->mesh.quadratures,
                               this->partial_derivatives,
                               this->properties,
                               this->mesh.mapping };
  this->fields = { this->mesh, this->properties, simulation };
  this->boundary_values = { max_timesteps, this->mesh, this->properties,
                            this->boundaries };
  return;
}

Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::compute::assembly::generate_wavefield_on_entire_grid(
    const specfem::wavefield::type wavefield,
    const specfem::wavefield::component component) {

  const int ncomponents = [&]() -> int {
    if (component == specfem::wavefield::component::displacement) {
      return 2;
    } else if (component == specfem::wavefield::component::velocity) {
      return 2;
    } else if (component == specfem::wavefield::component::acceleration) {
      return 2;
    } else {
      throw std::runtime_error("Wavefield component not supported");
    }
  }();

  // Copy the required wavefield into the buffer
  if (wavefield == specfem::wavefield::type::forward) {
    Kokkos::deep_copy(this->fields.buffer, this->fields.forward);
  } else if (wavefield == specfem::wavefield::type::adjoint) {
    Kokkos::deep_copy(this->fields.buffer, this->fields.adjoint);
  } else if (wavefield == specfem::wavefield::type::backward) {
    Kokkos::deep_copy(this->fields.buffer, this->fields.backward);
  } else {
    throw std::runtime_error("Wavefield type not supported");
  }

  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid("wavefield_on_entire_grid", this->mesh.nspec,
                               this->mesh.ngllz, this->mesh.ngllx, ncomponents);

  const auto h_wavefield_on_entire_grid =
      Kokkos::create_mirror_view(wavefield_on_entire_grid);

  if (component == specfem::wavefield::component::displacement) {
    get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic,
                                 specfem::wavefield::component::displacement>(
        *this, wavefield_on_entire_grid);
    get_wavefield_on_entire_grid<specfem::element::medium_tag::acoustic,
                                 specfem::wavefield::component::displacement>(
        *this, wavefield_on_entire_grid);
  } else if (component == specfem::wavefield::component::velocity) {
    get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic,
                                 specfem::wavefield::component::velocity>(
        *this, wavefield_on_entire_grid);
    get_wavefield_on_entire_grid<specfem::element::medium_tag::acoustic,
                                 specfem::wavefield::component::velocity>(
        *this, wavefield_on_entire_grid);
  } else if (component == specfem::wavefield::component::acceleration) {
    get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic,
                                 specfem::wavefield::component::acceleration>(
        *this, wavefield_on_entire_grid);
    get_wavefield_on_entire_grid<specfem::element::medium_tag::acoustic,
                                 specfem::wavefield::component::acceleration>(
        *this, wavefield_on_entire_grid);
  } else {
    throw std::runtime_error("Wavefield component not supported");
  }

  Kokkos::deep_copy(h_wavefield_on_entire_grid, wavefield_on_entire_grid);

  return h_wavefield_on_entire_grid;
}
