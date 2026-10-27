#include "specfem/injection/points/points.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/mpi.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

specfem::injection::InjectionPoints specfem::injection::make_injection_points(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const QuantityPolicy &quantities, const std::string_view method_name) {
  using Surface = specfem::assembly::InjectionSurface<
      specfem::element::dimension_tag::dim3>;
  static_assert(
      std::is_same_v<Surface::RowView, InjectionPoints::view_type>,
      "Injection surface rows must share the type of InjectionPoints arrays");
  const auto &surface = assembly.injection_surface;

  std::vector<long long> global_rows;
  specfem::tag_dispatch::for_each(
      Surface::combinations, [&]<typename TagsType>() {
        global_rows.push_back(surface.row_ranges.get<TagsType>().size());
      });
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(
      MPI_IN_PLACE, global_rows.data(), static_cast<int>(global_rows.size()),
      MPI_LONG_LONG, MPI_SUM, specfem::MPI::communicator()));

  InjectionPoints points;
  points.x = surface.x;
  points.y = surface.y;
  points.z = surface.z;
  points.nx = surface.nx;
  points.ny = surface.ny;
  points.nz = surface.nz;

  std::string unsupported;
  std::size_t system = 0;
  specfem::tag_dispatch::for_each(
      Surface::combinations, [&]<typename TagsType>() {
        constexpr auto medium = TagsType::medium_tag;
        constexpr auto property = TagsType::property_tag;
        const auto rows = surface.row_ranges.get<TagsType>();
        const auto &container =
            assembly.properties.get_container<medium, property>();

        PointGroup group;
        group.medium = static_cast<specfem_medium>(medium);
        group.property = static_cast<specfem_property>(property);
        group.begin = rows.begin_index();
        group.count = rows.size();
        container.for_each_host_view(
            [&](const auto &, const std::string &name) {
              group.property_names.push_back(name);
            });
        group.properties = PointGroup::properties_view(
            "specfem::injection::properties", group.count,
            group.property_names.size());
        int k = 0;
        container.for_each_host_view(
            [&](const auto &view, const std::string &) {
              for (int r = 0; r < rows.size(); ++r) {
                const int entry = surface.h_row_entry(rows(r));
                const int ispec =
                    assembly.properties
                        .template property_index_mapping<medium, property>(
                            surface.h_ispec(entry));
                group.properties(r, k) =
                    view(ispec, surface.h_iz(entry), surface.h_iy(entry),
                         surface.h_ix(entry));
              }
              ++k;
            });

        if (global_rows[system] > 0) {
          if (auto requested = quantities(medium, property)) {
            group.quantities = std::move(*requested);
          } else {
            if (!unsupported.empty())
              unsupported += ", ";
            unsupported += std::to_string(global_rows[system]) + " points in " +
                           specfem::element::to_string(medium) + "/" +
                           specfem::element::to_string(property);
          }
        }
        points.groups.push_back(std::move(group));
        ++system;
      });

  if (!unsupported.empty())
    throw std::runtime_error("The injection surface has " + unsupported +
                             " elements, but the " + std::string(method_name) +
                             " injection method does not support them.");
  return points;
}

specfem::injection::SimulationParameters
specfem::injection::make_simulation_parameters(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const int nstep) {
  SimulationParameters parameters{
    .t0 = assembly.t0, .dt = assembly.dt, .nstep = nstep, .bounds = {}
  };
  const auto &bounds = assembly.info.domain_bounds.bounds_array;
  for (int d = 0; d < 3; ++d) {
    parameters.bounds[2 * d] = bounds(d).min;
    parameters.bounds[2 * d + 1] = bounds(d).max;
  }
  return parameters;
}
