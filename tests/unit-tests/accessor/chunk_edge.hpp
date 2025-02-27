#include "chunk_edge/field.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk_edge.hpp"

#include <bitset>
#include <cstdio>
#include <string.h>

// KOKKOS_INLINE_FUNCTION
// std::string edge_to_string(const specfem::enums::edge::type edge) {
//   switch (edge) {
//   case specfem::enums::edge::type::RIGHT:
//     return std::string("RIGHT");
//   case specfem::enums::edge::type::TOP:
//     return std::string("TOP");
//   case specfem::enums::edge::type::LEFT:
//     return std::string("LEFT");
//   case specfem::enums::edge::type::BOTTOM:
//     return std::string("BOTTOM");
//   default:
//     return std::string("NONE");
//   }
// }

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
namespace failcodes {
constexpr int FAILURE_SIZE_INTDATA = 5;
enum ID {
  NO_ERROR,
  ISPEC_MISMATCH,
  IEDGE_MISMATCH,
  EDGE_NONE,
  IZ_IX_MISMATCH,
  POINTFIELD_READ_MISMATCH,
  EDGEFIELD_READ_MISMATCH
};
std::string id_str(ID id) {
  return (std::string[]){ "NO_ERROR",
                          "ISPEC_MISMATCH",
                          "IEDGE_MISMATCH",
                          "EDGE_NONE",
                          "IZ_IX_MISMATCH",
                          "POINTFIELD_READ_MISMATCH",
                          "EDGEFIELD_READ_MISMATCH" }[id];
};
struct failure {
  ID id;
  int league_rank;
  int team_rank;
  bool stored_index;
  int ispec, iz, ix;
  int int_data[FAILURE_SIZE_INTDATA];

  failure() : id(ID::NO_ERROR) {}

  KOKKOS_INLINE_FUNCTION
  failure(ID id, int league_rank, int team_rank)
      : id(id), league_rank(league_rank), team_rank(team_rank),
        stored_index(false) {}
  KOKKOS_INLINE_FUNCTION
  failure(ID id, int league_rank, int team_rank, int ispec, int iz, int ix)
      : id(id), league_rank(league_rank), team_rank(team_rank), ispec(ispec),
        iz(iz), ix(ix), stored_index(true) {}

  std::string get_message() {
    char buf[256];
    std::snprintf(buf, 256, " (ispec, iz, ix) = (%d, %d, %d) - ", ispec, iz,
                  ix);
    std::string msg = id_str(id) + ((stored_index) ? buf : " - ");
    switch (id) {
    case ID::IZ_IX_MISMATCH:
      std::snprintf(buf, 256, "igll=%d computed (iz,ix) = (%d,%d)", int_data[0],
                    int_data[1], int_data[2]);
      return msg + buf;
    case ID::NO_ERROR:
      return "No error.";
    default:
      return msg + "Message not yet implemented.";
    }
  }
};

template <typename MemberType>
KOKKOS_INLINE_FUNCTION failure ispec_mismatch(const MemberType team) {
  return failure(ID::ISPEC_MISMATCH, team.league_rank(), team.team_rank());
}
template <typename MemberType>
KOKKOS_INLINE_FUNCTION failure iedge_mismatch(const MemberType team) {
  failure fail(ID::IEDGE_MISMATCH, team.league_rank(), team.team_rank());
  return fail;
}
template <typename MemberType>
KOKKOS_INLINE_FUNCTION failure edge_none(const MemberType team) {
  return failure(ID::EDGE_NONE, team.league_rank(), team.team_rank());
}
template <typename MemberType>
KOKKOS_INLINE_FUNCTION failure iz_ix_mismatch(const MemberType team, int ispec,
                                              int iz, int ix, int igll,
                                              int found_iz, int found_ix) {
  failure fail(ID::IZ_IX_MISMATCH, team.league_rank(), team.team_rank(), ispec,
               iz, ix);
  fail.int_data[0] = igll;
  fail.int_data[1] = found_iz;
  fail.int_data[2] = found_ix;
  return fail;
}
template <typename MemberType>
KOKKOS_INLINE_FUNCTION failure pointfield_read_mismatch(const MemberType team,
                                                        int dcomp) {
  return failure(ID::POINTFIELD_READ_MISMATCH, team.league_rank(),
                 team.team_rank());
}
template <typename MemberType>
KOKKOS_INLINE_FUNCTION failure edgefield_read_mismatch(const MemberType team,
                                                       int dcomp) {
  return failure(ID::EDGEFIELD_READ_MISMATCH, team.league_rank(),
                 team.team_rank());
}

}; // namespace failcodes

template <int CHUNK_SIZE, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool USE_SIMD,
          typename FieldValFunction>
void verify_chunk_edges(std::shared_ptr<specfem::compute::assembly> assembly,
                        FieldValFunction &fieldval) {
  constexpr bool DISPLACEMENT = true;
  constexpr bool VELOCITY = true;
  constexpr bool ACCEL = true;
  constexpr bool MASS_MATRIX = false;

  constexpr int NUM_COMPONENTS =
      specfem::element::attributes<DimensionType, MediumTag>::components();

  using ChunkEdgeFieldType = specfem::chunk_edge::field<
      CHUNK_SIZE, NGLL, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      DISPLACEMENT, VELOCITY, ACCEL, MASS_MATRIX, USE_SIMD>;

  using SIMD = specfem::datatype::simd<type_real, USE_SIMD>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      DimensionType, SIMD, Kokkos::DefaultExecutionSpace>;
  using ChunkPolicyType = specfem::policy::chunk_edge<ParallelConfig>;

  int scratch_size = ChunkEdgeFieldType::shmem_size();

  //===========================================================================
  // craete a policy over all edges
  using EdgeIndexView =
      Kokkos::View<specfem::edge::index<DimensionType, USE_SIMD> *,
                   typename ParallelConfig::execution_space::memory_space>;

  const auto &simfield = assembly->fields.forward;
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> elems_to_test =
      assembly->element_types.get_elements_on_device(MediumTag);
  const int nspec_medium = elems_to_test.extent(0);
  // const int nspec = simfield.nspec;
  const auto index_mapping = simfield.index_mapping;

  const int nelements = nspec_medium * 4;
  EdgeIndexView edge_index_view("chunk_edge test: indices", nelements);
  Kokkos::parallel_for(
      "test accessor/chunk_edge.hpp: init element indices", nelements,
      KOKKOS_LAMBDA(const int i) {
        edge_index_view(i).ispec = elems_to_test(i / 4);
        const specfem::enums::edge::type edge[4] = {
          specfem::enums::edge::type::RIGHT, specfem::enums::edge::type::TOP,
          specfem::enums::edge::type::LEFT, specfem::enums::edge::type::BOTTOM
        };
        edge_index_view(i).edge_type = edge[i % 4];
      });
  ChunkPolicyType chunk_policy(edge_index_view, NGLL);

  constexpr int simd_size = SIMD::size();
  //===========================================================================

  // store fail conditions to print outside of loop
  Kokkos::View<failcodes::failure[1],
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
        ChunkEdgeFieldType edgefield(team);

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

          specfem::compute::load_on_device(team, iterator, simfield, edgefield);

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
                    edge_index_view(starting_element_index + ielem).ispec;
                const auto expected_edge =
                    edge_index_view(starting_element_index + ielem).edge_type;

                if (expected_ispec != ispec) {
                  failcontainer(0) = failcodes::ispec_mismatch(team);
                }
                if (expected_edge != edge) {
                  failcontainer(0) = failcodes::iedge_mismatch(team);
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
                  failcontainer(0) = failcodes::edge_none(team);
                }
                if (expected_ix != ix || expected_iz != iz) {
                  failcontainer(0) = failcodes::iz_ix_mismatch(
                      team, ispec, expected_iz, expected_ix, igll, iz, ix);
                }
                int iglob = index_mapping(ispec, iz, ix);

                specfem::point::field<DimensionType, MediumTag, DISPLACEMENT,
                                      VELOCITY, ACCEL, MASS_MATRIX, USE_SIMD>
                    pointfield;
                // specfem::compute::load_on_device(index, simfield,
                // pointfield); for (int icomp = 0;
                //      icomp < NUM_COMPONENTS;
                //      icomp++) {

                //   // if constexpr (DISPLACEMENT)
                //   {
                //     if (pointfield.displacement(icomp) != fieldval(iglob,
                //     icomp, 0)){
                //       failcontainer(0) =
                //       failcodes::pointfield_read_mismatch(team, 0);
                //     }else
                //     if (edgefield.displacement(ielem, igll, icomp) !=
                //     fieldval(iglob, icomp, 0)) {
                //       failcontainer(0) =
                //       failcodes::edgefield_read_mismatch(team, 0);
                //     }
                //   }
                //   // if constexpr (VELOCITY)
                //   {
                //     if (pointfield.velocity(icomp) != fieldval(iglob, icomp,
                //     1)){
                //       failcontainer(0) =
                //       failcodes::pointfield_read_mismatch(team, 1);
                //     }else
                //     if (edgefield.velocity(ielem, igll, icomp) !=
                //     fieldval(iglob, icomp, 1)) {
                //       failcontainer(0) =
                //       failcodes::edgefield_read_mismatch(team, 1);
                //     }
                //   }
                //   // if constexpr (ACCEL)
                //   {
                //     if (pointfield.acceleration(icomp) != fieldval(iglob,
                //     icomp, 2)){
                //       failcontainer(0) =
                //       failcodes::pointfield_read_mismatch(team, 2);
                //     }else
                //     if (edgefield.acceleration(ielem, igll, icomp) !=
                //     fieldval(iglob, icomp, 2)) {
                //       failcontainer(0) =
                //       failcodes::edgefield_read_mismatch(team, 2);
                //     }
                //   }
                //   // if constexpr (MASS_MATRIX) {
                //   //   if (pointfield.mass_matrix(icomp) !=
                //   //           fieldval(iglob, icomp, 3) ||
                //   //       pointfield.mass_matrix(icomp) !=
                //   //           edgefield.mass_matrix(ielem, igll, icomp)) {
                //   //     failderiv = 3;
                //   //     got_pt = pointfield.mass_matrix(icomp);
                //   //     got_edge = edgefield.mass_matrix(ielem, igll,
                //   icomp);
                //   //   }
                //   // }
                // }
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
  Kokkos::fence();
  Kokkos::deep_copy(h_failcontainer, failcontainer);
  // if an error was generated, fail it
  if (h_failcontainer(0).id != failcodes::ID::NO_ERROR) {
    failcodes::failure &fail = h_failcontainer(0);
    FAIL() << "--------------------------------------------------\n"
           << "\033[0;31m[FAILED]\033[0m Test failed\n"
           << " - Chunk Edge\n"
           << " - error code " << failcodes::id_str(fail.id)
           << " ; full error\n"
           << "      " << fail.get_message() << "\n"
           << "--------------------------------------------------\n\n"
           << std::endl;
  }

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
           << " - iterator missed " << misses << " / " << nelements * NGLL
           << " entries\n"
           << "--------------------------------------------------\n\n"
           << std::endl;
  }
}
