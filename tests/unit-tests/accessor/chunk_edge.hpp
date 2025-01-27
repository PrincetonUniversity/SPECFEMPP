#include "chunk_edge/field.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk_edge.hpp"

#include <string.h>

KOKKOS_INLINE_FUNCTION
std::string edge_to_string(const specfem::enums::edge::type edge) {
  switch (edge) {
  case specfem::enums::edge::type::RIGHT:
    return std::string("RIGHT");
  case specfem::enums::edge::type::TOP:
    return std::string("TOP");
  case specfem::enums::edge::type::LEFT:
    return std::string("LEFT");
  case specfem::enums::edge::type::BOTTOM:
    return std::string("BOTTOM");
  default:
    return std::string("NONE");
  }
}

std::string wrap_text(std::string str, int maxcols,
                      std::string newline = std::string("\n")) {
  int rowstart_loc = 0;
  int last_space = 0;
  int ind = 1;
  // go through the string, creating newlines when necessary
  while (ind < str.length()) {
    if (str.at(ind) == ' ') {
      last_space = ind; // mark this as where we can wrap
    } else if (str.at(ind) == '\n') {
      rowstart_loc = last_space =
          ind; // new line in given string. Replace with our newline
      str =
          str.substr(0, ind) + newline + str.substr(ind + 1, std::string::npos);
    }

    if (ind - rowstart_loc >= maxcols) {
      // we need to insert a newline

      if (last_space <= rowstart_loc) {
        // a super long word; just cut here
        rowstart_loc = last_space = ind;
        str = str.substr(0, ind) + newline + str.substr(ind, std::string::npos);
      } else {
        // we have a word wrap somewhere, cut there
        rowstart_loc = ind = last_space;
        int newstart =
            (str.at(ind) == ' ') ? ind + 1 : ind; // replace whitespace
        str = str.substr(0, ind) + newline +
              str.substr(newstart, std::string::npos);
      }
    }
    ind++;
  }
  return str;
}

struct access_failcond {

  char message[256];
  int league_rank;
  int team_rank;

  bool isfail;
  template <typename MemberType>
  access_failcond(const MemberType team, const char *message)
      : isfail(true), league_rank(team.league_rank()),
        team_rank(team.team_rank()) {
    strcpy(this->message, message);
  }

  access_failcond() : isfail(false) {}

  void handle() {
    if (isfail) {
      std::string message = " - Error: " + std::string(this->message);
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - Chunk Edge\n"
             << wrap_text(message, 50, "\n -   ") << "\n"
             << " - (team / league) = (" << team_rank << "," << league_rank
             << ")\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
  }
};

template <int CHUNK_SIZE, int NGLL, specfem::dimension::type DimensionType,
          bool USE_SIMD, typename ExecutionSpace, typename FieldValFunction>
void verify_chunk_edges(std::shared_ptr<specfem::compute::assembly> assembly,
                        FieldValFunction &fieldval) {
  constexpr bool DISPLACEMENT = true;
  constexpr bool VELOCITY = true;
  constexpr bool ACCEL = true;
  constexpr bool MASS_MATRIX = false;

  using ChunkEdgeAcoustic = specfem::chunk_edge::field<
      CHUNK_SIZE, NGLL, DimensionType, specfem::element::medium_tag::acoustic,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      DISPLACEMENT, VELOCITY, ACCEL, MASS_MATRIX, USE_SIMD>;
  using ChunkEdgeElastic = specfem::chunk_edge::field<
      CHUNK_SIZE, NGLL, DimensionType, specfem::element::medium_tag::elastic,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      DISPLACEMENT, VELOCITY, ACCEL, MASS_MATRIX, USE_SIMD>;

  using SIMD = specfem::datatype::simd<type_real, USE_SIMD>;
  using ParallelConfig =
      specfem::parallel_config::default_chunk_config<DimensionType, SIMD,
                                                     ExecutionSpace>;
  using ChunkPolicyType = specfem::policy::chunk_edge<ParallelConfig>;

  int scratch_size =
      ChunkEdgeAcoustic::shmem_size() + ChunkEdgeElastic::shmem_size();

  //===========================================================================
  // craete a policy over all edges
  using IspecView =
      Kokkos::View<int *,
                   typename ParallelConfig::execution_space::memory_space>;
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *,
                   typename ParallelConfig::execution_space::memory_space>;

  const auto element_type = assembly->properties.element_types;
  const auto &simfield = assembly->fields.forward;
  const int nspec = simfield.nspec;
  const int ngll = simfield.ngllx;
  const auto index_mapping = simfield.index_mapping;

  const int nelements = nspec * 4;
  IspecView ispec_view("chunk_edge test: ispec", nelements);
  EdgeTypeView edgetype_view("chunk_edge test: edge", nelements);
  Kokkos::parallel_for(
      "test accessor/chunk_edge.hpp: init element indices", nelements,
      KOKKOS_LAMBDA(const int i) {
        ispec_view(i) = i / 4;
        const specfem::enums::edge::type edge[4] = {
          specfem::enums::edge::type::RIGHT, specfem::enums::edge::type::TOP,
          specfem::enums::edge::type::LEFT, specfem::enums::edge::type::BOTTOM
        };
        edgetype_view(i) = edge[i % 4];
      });
  ChunkPolicyType chunk_policy(ispec_view, edgetype_view, NGLL);

  constexpr int simd_size = SIMD::size();
  //===========================================================================

  // store fail conditions to print outside of loop
  Kokkos::View<access_failcond[1],
               typename ParallelConfig::execution_space::memory_space>
      failcontainer("failreduction");
  auto h_failcontainer = Kokkos::create_mirror_view(failcontainer);

  // store all hit indices.
  Kokkos::View<bool *[NGLL],
               typename ParallelConfig::execution_space::memory_space>
      hit_indices("indices hit check", nelements);
  auto h_hit_indices = Kokkos::create_mirror_view(hit_indices);

  // "specfem::domain::domain::compute_seismogram",
  // specfem::kokkos::DeviceTeam(nreceivers * nseismograms, Kokkos::AUTO, 1)
  //     .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
  Kokkos::parallel_for(
      "test accessor/chunk_edge.hpp: valdate iter and loads",
      static_cast<typename ChunkPolicyType::policy_type &>(chunk_policy)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
        ChunkEdgeAcoustic edge_acoustic(team);
        ChunkEdgeElastic edge_elastic(team);

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
                const int ielem = iterator_index.ielement;
                const int ispec = index.ispec;
                const int ix = index.ix;
                const int iz = index.iz;
                const int igll = iterator_index.igll;
                const auto edge = iterator_index.edge;
                hit_indices(starting_element_index + ielem, igll) = true;

                const int expected_ispec =
                    ispec_view(starting_element_index + ielem);
                const auto expected_edge =
                    edgetype_view(starting_element_index + ielem);

                if (expected_ispec != ispec) {
                  failcontainer(0) = access_failcond(
                      team, ("iter index " + std::to_string(i) + ": ielement " +
                             std::to_string(ielem) + " should map to ispec=" +
                             std::to_string(expected_ispec) + ". Got " +
                             std::to_string(ispec) + " instead." +
                             "\n(Starting element index:" +
                             std::to_string(starting_element_index) + ")")
                                .c_str());
                }
                if (expected_edge != edge) {
                  failcontainer(0) = access_failcond(
                      team, ("iter index " + std::to_string(i) + ": ielement " +
                             std::to_string(ielem) + " should map to edge=" +
                             edge_to_string(expected_edge) + ". Got " +
                             edge_to_string(edge) + " instead." +
                             "\n(Starting element index:" +
                             std::to_string(starting_element_index) + ")")
                                .c_str());
                }

                int expected_iz;
                int expected_ix;
                switch (edge) {
                case specfem::enums::edge::type::RIGHT:
                  expected_iz = igll;
                  expected_ix = NGLL - 1;
                  break;
                case specfem::enums::edge::type::TOP:
                  expected_iz = NGLL - 1;
                  expected_ix = igll;
                  break;
                case specfem::enums::edge::type::LEFT:
                  expected_iz = igll;
                  expected_ix = 0;
                  break;
                case specfem::enums::edge::type::BOTTOM:
                  expected_iz = 0;
                  expected_ix = igll;
                  break;
                default:
                  failcontainer(0) = access_failcond(
                      team, ("indexing array has NONE at index " +
                             std::to_string(starting_element_index + ielem) +
                             ". Fix the test.")
                                .c_str());
                }
                if (expected_ix != ix || expected_iz != iz) {
                  failcontainer(0) = access_failcond(
                      team, ("iter index " + std::to_string(i) + ": igll " +
                             std::to_string(igll) + " with edge " +
                             edge_to_string(edge) + " should map to (iz,ix)=(" +
                             std::to_string(expected_iz) + "," +
                             std::to_string(expected_ix) + "). Got (" +
                             std::to_string(iz) + "," + std::to_string(ix) +
                             ") instead." + "\n(Starting element index:" +
                             std::to_string(starting_element_index) + ")")
                                .c_str());
                }
                int iglob = index_mapping(ispec, iz, ix);

#define test_against_true(medium)                                              \
  {                                                                            \
    specfem::point::field<DimensionType, medium, DISPLACEMENT, VELOCITY,       \
                          ACCEL, MASS_MATRIX, USE_SIMD>                        \
        pointfield;                                                            \
    specfem::compute::load_on_device(index, assembly->fields.forward,          \
                                     pointfield);                              \
    for (int icomp = 0;                                                        \
         icomp <                                                               \
         specfem::element::attributes<DimensionType, medium>::components();    \
         icomp++) {                                                            \
                                                                               \
      int failderiv = -1;                                                      \
      type_real got;                                                           \
      if constexpr (DISPLACEMENT) {                                            \
        if (pointfield.displacement(icomp) != fieldval(iglob, icomp, 0)) {     \
          failderiv = 0;                                                       \
          got = pointfield.displacement(icomp);                                \
        }                                                                      \
      }                                                                        \
      if constexpr (VELOCITY) {                                                \
        if (pointfield.velocity(icomp) != fieldval(iglob, icomp, 1)) {         \
          failderiv = 1;                                                       \
          got = pointfield.velocity(icomp);                                    \
        }                                                                      \
      }                                                                        \
      if constexpr (ACCEL) {                                                   \
        if (pointfield.acceleration(icomp) != fieldval(iglob, icomp, 2)) {     \
          failderiv = 2;                                                       \
          got = pointfield.acceleration(icomp);                                \
        }                                                                      \
      }                                                                        \
      if constexpr (MASS_MATRIX) {                                             \
        if (pointfield.mass_matrix(icomp) != fieldval(iglob, icomp, 3)) {      \
          failderiv = 3;                                                       \
          got = pointfield.mass_matrix(icomp);                                 \
        }                                                                      \
      }                                                                        \
      if (failderiv != -1) {                                                   \
        failcontainer(0) = access_failcond(                                    \
            team, ("iter index " + std::to_string(i) + ": index(" +            \
                   std::to_string(ispec) + "," + std::to_string(iz) + "," +    \
                   std::to_string(ix) +                                        \
                   ") giving iglob = " + std::to_string(iglob) +               \
                   " got a failed read at icomp = " + std::to_string(icomp) +  \
                   " and deriv order/ trait = " + std::to_string(failderiv) +  \
                   ". Expected " +                                             \
                   std::to_string(fieldval(iglob, icomp, failderiv)) +         \
                   " and got " + std::to_string(got) + ".")                    \
                      .c_str());                                               \
      }                                                                        \
    }                                                                          \
  }

                switch (element_type(ispec)) {
                case specfem::element::medium_tag::acoustic: {
                  test_against_true(specfem::element::medium_tag::acoustic);
                  break;
                }
                case specfem::element::medium_tag::elastic: {
                  test_against_true(specfem::element::medium_tag::elastic);
                  break;
                }
                }
#undef test_against_true
                // const auto point_property = [&]() -> PointPropertyType {
                //   PointPropertyType point_property;

                //   specfem::compute::load_on_device(index, properties,
                //                                    point_property);
                //   return point_property;
                // }();

                // const auto point_partial_derivatives =
                //     [&]() -> PointPartialDerivativesType {
                //   PointPartialDerivativesType point_partial_derivatives;
                //   specfem::compute::load_on_device(index,
                //   partial_derivatives,
                //                                    point_partial_derivatives);
                //   return point_partial_derivatives;
                // }();
              });
        }
      });
  Kokkos::deep_copy(h_failcontainer, failcontainer);
  // if an error was generated, fail it
  h_failcontainer(0).handle();

  int misses;
  Kokkos::parallel_reduce(
      "all-hit check", nelements * NGLL,
      KOKKOS_LAMBDA(const int i, int &res) {
        if (!hit_indices(i / NGLL, i % NGLL)) {
          res++;
        }
      },
      Kokkos::Sum<int>(misses));

  if (misses != 0) {
    // should we say which ones? we can do that once the problem arises.
    FAIL() << "--------------------------------------------------\n"
           << "\033[0;31m[FAILED]\033[0m Test failed\n"
           << " - Chunk Edge\n"
           << " - iterator missed " << misses << " entries\n"
           << "--------------------------------------------------\n\n"
           << std::endl;
  }
}
