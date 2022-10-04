#ifndef JACOBIAN_H
#define JaCOBIAN_H

#include "../include/kokkos_abstractions.h"

namespace jacobian {

std::tuple<type_real, type_real>
compute_locations(const specfem::HostTeam::member_type &teamMember,
                  const specfem::HostScratchView2d<type_real> s_coorg,
                  const int ngnod, const type_real xi, const type_real gamma);

std::tuple<type_real, type_real>
compute_locations(const specfem::HostTeam::member_type &teamMember,
                  const specfem::HostScratchView2d<type_real> s_coorg,
                  const int ngnod,
                  const specfem::HostView1d<type_real> shape2D);

std::tuple<type_real, type_real, type_real, type_real>
compute_partial_derivatives(const specfem::HostTeam::member_type &teamMember,
                            const specfem::HostScratchView2d<type_real> s_coorg,
                            const int ngnod, const type_real xi,
                            const type_real gamma);

std::tuple<type_real, type_real, type_real, type_real>
compute_partial_derivatives(const specfem::HostTeam::member_type &teamMember,
                            const specfem::HostScratchView2d<type_real> s_coorg,
                            const int ngnod,
                            const specfem::HostView2d<type_real> dershape2D);

type_real compute_jacobian(const type_real xxi, const type_real zxi,
                           const type_real xgamma, const type_real zgamma);

type_real compute_jacobian(const specfem::HostTeam::member_type &teamMember,
                           const specfem::HostScratchView2d<type_real> s_coorg,
                           const int ngnod, const type_real xi,
                           const type_real gamma);

type_real compute_jacobian(const specfem::HostTeam::member_type &teamMember,
                           const specfem::HostScratchView2d<type_real> s_coorg,
                           const int ngnod,
                           const specfem::HostView2d<type_real> dershape2D);

// HostArray2d compute_partial_derivatives(const HostArray1d greekcoord,
//                                         const HostArray2d coorg,
//                                         const int ngod);
// double jacobian::compute_jacobian(const HostArray2d partial_der);
// double compute_jacobian(const HostArray1d greekcoord, const HostArray2d
// coorg,
//                         const int ngod);

// HostArray2d compute_inverse_partial_derivatives(const HostArray1d greekcoord,
//                                                 const HostArray2d coorg,
//                                                 const int ngod);
} // namespace jacobian

#endif
