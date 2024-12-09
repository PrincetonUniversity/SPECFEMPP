#ifndef __UTIL_EDGE_STORAGES_TPP_
#define __UTIL_EDGE_STORAGES_TPP_

#include "_util/edge_storages.hpp"
#include <array>
#include <cmath>
#include <iostream>

namespace _util {
namespace edge_manager {

template <int ngll, int datacapacity>
edge_data<ngll,datacapacity>& edge_storage<ngll, datacapacity>::get_edge_on_host(int edge){
  return h_edge_data_container(edge);
}
template <int ngll, int datacapacity>
edge_intersection<ngll>& edge_storage<ngll, datacapacity>::get_intersection_on_host(int intersection){
  if(!intersections_built){
    build_intersections_on_host();
  }
  return h_intersection_container(intersection);
}

template <int ngll, int datacapacity>
edge_storage<ngll, datacapacity>::edge_storage(const std::vector<edge> edges)
    : n_edges(edges.size()), edges(edges), intersections_built(false),
      edge_data_container(
          specfem::kokkos::DeviceView1d<edge_data<ngll, datacapacity> >(
              "_util::edge_manager::edge_storage::edge_data", n_edges)),
      h_edge_data_container(Kokkos::create_mirror_view(edge_data_container)) {
  for (int i = 0; i < n_edges; i++) {
    h_edge_data_container(i).parent = edges[i];
  }
  Kokkos::deep_copy(edge_data_container, h_edge_data_container);
}

template <int ngll, int datacapacity>
void edge_storage<ngll, datacapacity>::initialize_intersection_data(int capacity){
  if (!intersections_built){
    build_intersections_on_host();
  }
  intersection_data = specfem::kokkos::DeviceView2d<type_real>(
              "_util::edge_manager::edge_storage::edge_data", n_intersections, capacity);
  h_intersection_data = Kokkos::create_mirror_view(intersection_data);

  intersection_data_built = true;
}

template <int ngll, int datacapacity>
void edge_storage<ngll, datacapacity>::foreach_edge_on_host(
    const std::function<void(edge_data<ngll, datacapacity> &)> &func) {
  for (int i = 0; i < n_edges; i++) {
    func(h_edge_data_container(i));
  }
  Kokkos::deep_copy(edge_data_container, h_edge_data_container);
}

template <int ngll, int datacapacity>
void edge_storage<ngll, datacapacity>::foreach_intersection_on_host(
    const std::function<void(edge_intersection<ngll> &,
                             edge_data<ngll, datacapacity> &,
                             edge_data<ngll, datacapacity> &)> &func) {
  if (!intersections_built) {
    build_intersections_on_host();
  }
  for (int i = 0; i < n_intersections; i++) {
    edge_intersection<ngll> &ei = h_intersection_container(i);
    func(ei, h_edge_data_container(ei.a_ref_ind),
         h_edge_data_container(ei.b_ref_ind));
  }
  Kokkos::deep_copy(intersection_container, h_intersection_container);
  Kokkos::deep_copy(edge_data_container, h_edge_data_container);
}

template <int ngll, int datacapacity>
void edge_storage<ngll, datacapacity>::foreach_intersection_on_host(
    const std::function<void(edge_intersection<ngll> &,
                              edge_data<ngll, datacapacity> &,
                              edge_data<ngll, datacapacity> &,
          decltype(Kokkos::subview(std::declval<specfem::kokkos::HostView2d<type_real>>(), 1u, Kokkos::ALL)))> &func){

  if (!intersection_data_built) {
    throw std::runtime_error("Attempting a foreach_intersection_on_host() with data access before the intersections array was built!");
  }
  for (int i = 0; i < n_intersections; i++) {
    edge_intersection<ngll> &ei = h_intersection_container(i);
    func(ei, h_edge_data_container(ei.a_ref_ind),
         h_edge_data_container(ei.b_ref_ind), Kokkos::subview(h_intersection_data,i,Kokkos::ALL));
  }
  Kokkos::deep_copy(intersection_container, h_intersection_container);
  Kokkos::deep_copy(edge_data_container, h_edge_data_container);
  Kokkos::deep_copy(intersection_data, h_intersection_data);
}

/**
 * @brief Checks whether or not these edges intersect. If they do, intersection
 * is modified to contain the intersection. That is, the parameter_start/end
 * fields in intersection& are set.
 *
 * @param a the first edge
 * @param b the second edge
 * @param intersection the struct to store the intersection into if there is a
 * nonzero intersection. This reference may update the parameter_start/end
 * variables even if there's no intersection.
 * @return true if a nonzero intersection occurs between these two edges
 * @return false if no nonzero intersection occurs between these two edges
 */
template <int ngll, int datacapacity>
bool intersect(edge_data<ngll, datacapacity> &a,
               edge_data<ngll, datacapacity> &b,
               edge_intersection<ngll> &intersection) {
#define intersect_eps 1e-3
#define intersect_eps2 (intersect_eps * intersect_eps)
  quadrature_rule gll = gen_GLL(ngll);

  // make sure ngll is correct
  if (a.ngll != ngll || b.ngll != ngll) {
    throw std::runtime_error(
        "bool intersect(edge_data,edge_data,edge_intersection): ngll different "
        "from expected.");
  }
  //maybe do an AABB check for performance? the box can be precomputed per edge.


  //approximate edges as linear segments
  constexpr int subdivisions = 10;
  constexpr type_real h = 2.0/subdivisions;

  type_real ax[subdivisions+1];
  type_real bx[subdivisions+1];
  type_real az[subdivisions+1];
  type_real bz[subdivisions+1];

  for(int i = 0; i < subdivisions+1; i++){
    ax[i] = gll.interpolate(a.x, i*h);
    az[i] = gll.interpolate(a.z, i*h);
    bx[i] = gll.interpolate(b.x, i*h);
    bz[i] = gll.interpolate(b.z, i*h);
  }

  const auto line_intersection = [&gll](type_real ax0, type_real az0, type_real ax1, type_real az1,
                type_real bx0, type_real bz0, type_real bx1, type_real bz1,
                type_real& alow, type_real& ahigh, type_real& blow, type_real& bhigh) -> bool {
    type_real adx = ax1-ax0;
    type_real bdx = bx1-bx0;
    type_real adz = az1-az0;
    type_real bdz = bz1-bz0;
    type_real cross = adx * bdz - bdx * adz;
    type_real sin2 = cross * cross / ((adx * adx + adz * adz) * (bdx * bdx + bdz * bdz));
    if (sin2 > 1e-2){
      //not parallel, but there is an interval for which the lines are within eps distance
      //parameters where intersections occur: use cramer's rule
      type_real c1 = bx0-ax0;
      type_real c2 = bz0-az0;
      type_real ta = (bdz*c1 - bdx*c2)/cross;
      type_real tb = (adz*c1 - adx*c2)/cross;

      //this is the distance from the intersection where the lines are intersect_eps distance apart
      type_real permitted_dist = intersect_eps / sqrt(2 - 2*sqrt(1 - sin2));
      //and the distance in parameter space to achieve that:
      type_real a_shift = permitted_dist / sqrt(adx*adx + adz*adz);
      type_real b_shift = permitted_dist / sqrt(bdx*bdx + bdz*bdz);
      alow = std::max((type_real)0.0, ta - a_shift);
      blow = std::max((type_real)0.0, tb - b_shift);
      ahigh = std::min((type_real)1.0, ta + a_shift);
      bhigh = std::min((type_real)1.0, tb + b_shift);

    //confirm that the intersection occurs inside the segments
      if(ahigh - alow <= 0 || bhigh - blow <= 0){
        return false;
      }

      return true;
    }
    // sin^2 theta <= eps, so parallel; orth project points to find distance between lines


    //(a0-b0) - ad( (a0 - b0) . (ad)/|ad|^2 )

    // initial_point_deviation . a_direction
    type_real dot_over_mag2 =
        ((ax0 - bx0) * adx + (az0 - bz0) * adz) / (adx * adx + adz * adz);
    //proj initial_point_deviation perp to {a_direction}
    type_real orthx = (ax0 - bx0) - adx * dot_over_mag2;
    type_real orthz = (az0 - bz0) - adz * dot_over_mag2;

    if (orthx * orthx + orthz * orthz > intersect_eps2) {
      // distance between lines is greater than eps, so no intersection
      return false;
    }

    //the lines intersect. Find start and end parameters. We use [0,1] for parameter
    // (a0 + ad*ta = b0 + bd*tb)


    // find (a0 + ad*tb0 ~ b0) and (a0 + ad*tb0 ~ b1): do a projection
    type_real tb0 = ((bx0 - ax0) * adx + (bz0 - az0) * adz) / (adx * adx + adz * adz);
    type_real tb1 = ((bx1 - ax0) * adx + (bz1 - az0) * adz) / (adx * adx + adz * adz);

    alow = std::max((type_real)0.0, std::min(tb0, tb1));
    ahigh = std::min((type_real)1.0, std::max(tb0, tb1));

    //confirm that the intersection occurs inside the segment
    if(ahigh - alow <= 0){
      return false;
    }

    //we have an intersection. Set blow,bhigh and return true

    if (fabs(bdx) > fabs(bdz)) {
      blow = (ax0 + alow * adx - bx0) / bdx;
      bhigh = (ax0 + ahigh * adx - bx0) / bdx;
    } else {
      blow = (az0 + alow * adz - bz0) / bdz;
      bhigh = (az0 + ahigh * adz - bz0) / bdz;
    }
    return true;
  };

  float a_param_start = 1, a_param_end = -1;
  float b_param_start = 1, b_param_end = -1;



  //this is O(n^2) check. this really needs to be optimized
  float alow,ahigh,blow,bhigh;
  for(int ia = 0; ia < subdivisions; ia++){
    for(int ib = 0; ib < subdivisions; ib++){
      if (line_intersection(ax[ia],az[ia],ax[ia+1],az[ia+1],
                           bx[ib],bz[ib],bx[ib+1],bz[ib+1],
                           alow,ahigh,blow,bhigh)){
        //push out parameters. We assume intersection is convex
        a_param_start = std::min(a_param_start, (ia+alow)*h-1);
        a_param_end = std::max(a_param_end, (ia+ahigh)*h-1);
        b_param_start = std::min(b_param_start, (ib+blow)*h-1);
        b_param_end = std::max(b_param_end, (ib+bhigh)*h-1);
      }
    }
  }
  if(a_param_end - a_param_start < intersect_eps && b_param_end - b_param_start < intersect_eps){
    //intersection (within segments) is too small
    return false;
  }
  intersection.a_param_start = a_param_start;
  intersection.a_param_end = a_param_end;
  intersection.b_param_start = b_param_start;
  intersection.b_param_end = b_param_end;

  // populate mortar transfer functions by computing reference parameters for even spacing
  type_real t_samples[ngll];

  //this uses the linear approximations, but we should consider a different scheme (locate_point?)
  type_real a_len = 0, b_len = 0;
  type_real a_sublens[subdivisions];
  type_real b_sublens[subdivisions];
  for(int i = 0; i < subdivisions; i++){
    type_real dx = ax[i+1] - ax[i];
    type_real dz = az[i+1] - az[i];
    a_sublens[i] = h*sqrt(dx*dx + dz*dz);
    a_len += a_sublens[i];
    dx = bx[i+1] - bx[i];
    dz = bz[i+1] - bz[i];
    b_sublens[i] = h*sqrt(dx*dx + dz*dz);
    b_len += b_sublens[i];
  }

  //even spacing is len_desired = i * len * (t+1)/2
  //find parameters int_-1^{t_desired} |dr|  = len_desired
  type_real len_inc = a_sublens[0];
  int segment = 1;
  for (int i = 0; i < ngll; i++) {
    type_real len_desired = 0.5*(1+gll.t[i])*a_len;
    while(len_inc < len_desired){
      len_inc += a_sublens[segment];
      segment++;
    }
    // len_inc = len_desired + delta
    // t_desired = h*segment - delta/sublen[segment-1] - 1
    // since h * segment -> len_inc, and we use linear approx
    t_samples[i] = h*(segment + (len_desired - len_inc)/a_sublens[segment-1]) - 1;
  }
  gll.sample_L(intersection.a_mortar_trans, t_samples, ngll);


  len_inc = b_sublens[0];
  segment = 1;
  for (int i = 0; i < ngll; i++) {
    type_real len_desired = 0.5*(1+gll.t[i])*b_len;
    while(len_inc < len_desired){
      len_inc += b_sublens[segment];
      segment++;
    }
    // len_inc = len_desired + delta
    // t_desired = h*segment + delta/sublen[segment-1]
    t_samples[i] = h*(segment + (len_desired - len_inc)/b_sublens[segment-1]) - 1;
  }
  gll.sample_L(intersection.b_mortar_trans, t_samples, ngll);
  intersection.ngll = ngll;
  intersection.a_ngll = a.ngll;
  intersection.b_ngll = b.ngll;
  return true;
#undef intersect_eps2
#undef intersect_eps
}

template <int ngll, int datacapacity>
void edge_storage<ngll, datacapacity>::build_intersections_on_host() {
  std::vector<edge_intersection<ngll> > intersections;
  edge_intersection<ngll> intersection;
  // foreach unordered pair (edge[i], edge[j]), j != i
  for (int i = 0; i < n_edges; i++) {
    for (int j = i + 1; j < n_edges; j++) {
      // if there is an intersection, store it.
      if (intersect(h_edge_data_container(i), h_edge_data_container(j),
                    intersection)) {
        intersection.a_ref_ind = i;
        intersection.b_ref_ind = j;
        intersections.push_back(intersection);

        // edge_data<ngll,datacapacity> a = h_edge_data_container(i);
        // edge_data<ngll,datacapacity> b = h_edge_data_container(j);
        // type_real x1a = 0.5*(a.x[ngll-1] + a.x[0]);
        // type_real z1a = 0.5*(a.z[ngll-1] + a.z[0]);
        // type_real x1b = 0.5*(b.x[ngll-1] + b.x[0]);
        // type_real z1b = 0.5*(b.z[ngll-1] + b.z[0]);
        // //deriv
        // type_real x2a = 0.5*(a.x[ngll-1] - a.x[0]);
        // type_real z2a = 0.5*(a.z[ngll-1] - a.z[0]);
        // type_real x2b = 0.5*(b.x[ngll-1] - b.x[0]);
        // type_real z2b = 0.5*(b.z[ngll-1] - b.z[0]);
        // if((x1a-x1b)*(x1a-x1b) + (z1a-z1b)*(z1a-z1b) < 1e-6){continue;}
        // printf("inter: (%.2f,%.2f)-(%.2f,%.2f) :
        // (%.2f,%.2f)-(%.2f,%.2f)\n",x1a-x2a,z1a-z2a,x1a+x2a,z1a+z2a,x1b-x2b,z1b-z2b,x1b+x2b,z1b+z2b);
      }
    }
  }
  n_intersections = intersections.size();
  intersection_container =
      specfem::kokkos::DeviceView1d<edge_intersection<ngll> >(
          "_util::edge_manager::edge_storage::edge_data", n_intersections);
  h_intersection_container = Kokkos::create_mirror_view(intersection_container);
  for (int i = 0; i < n_intersections; i++) {
    h_intersection_container(i) = intersections[i];
  }
  Kokkos::deep_copy(intersection_container, h_intersection_container);
  intersections_built = true;
}

type_real quadrature_rule::integrate(type_real *f) {
  type_real sum = 0;
  for (int i = 0; i < nquad; i++) {
    sum += f[i] * w[i];
  }
  return sum;
}
type_real quadrature_rule::deriv(type_real *f, type_real t) {
  // f(t) = sum_{j} f[j] L_j(t) = sum_{ij} f[j] L_{ji} i t^(i-1)
  type_real tim1 = 1; // t^(i-1)
  type_real sum = 0;
  for (int i = 1; i < nquad; i++) {
    for (int j = 0; j < nquad; j++) {
      sum += f[j] * L[j * nquad + i] * i * tim1;
    }
    tim1 *= t;
  }
  return sum;
}
type_real quadrature_rule::interpolate(type_real *f, type_real t) {
  // f(t) = sum_{j} f[j] L_j(t) = sum_{ij} f[j] L_{ji} t^i
  type_real ti = 1; // t^i
  type_real sum = 0;
  for (int i = 0; i < nquad; i++) {
    for (int j = 0; j < nquad; j++) {
      sum += f[j] * L[j * nquad + i] * ti;
    }
    ti *= t;
  }
  return sum;
}

quadrature_rule gen_GLL(int ngll) {
  // TODO should we set a builder for a general rule?
  if (ngll == 5) {
    quadrature_rule gll(ngll);
    gll.t[0] = -1.000000000000000;
    gll.t[1] = -0.654653670707977;
    gll.t[2] = 0.000000000000000;
    gll.t[3] = 0.654653670707977;
    gll.t[4] = 1.000000000000000;
    gll.w[0] = 0.100000000000000;
    gll.w[1] = 0.544444444444444;
    gll.w[2] = 0.711111111111111;
    gll.w[3] = 0.544444444444444;
    gll.w[4] = 0.100000000000000;
    gll.L[0] = -0.000000000000000;
    gll.L[1] = 0.375000000000000;
    gll.L[2] = -0.375000000000000;
    gll.L[3] = -0.875000000000000;
    gll.L[4] = 0.875000000000000;
    gll.L[5] = 0.000000000000000;
    gll.L[6] = -1.336584577695453;
    gll.L[7] = 2.041666666666667;
    gll.L[8] = 1.336584577695453;
    gll.L[9] = -2.041666666666667;
    gll.L[10] = 1.000000000000000;
    gll.L[11] = -0.000000000000000;
    gll.L[12] = -3.333333333333333;
    gll.L[13] = 0.000000000000000;
    gll.L[14] = 2.333333333333333;
    gll.L[15] = -0.000000000000000;
    gll.L[16] = 1.336584577695453;
    gll.L[17] = 2.041666666666667;
    gll.L[18] = -1.336584577695453;
    gll.L[19] = -2.041666666666667;
    gll.L[20] = 0.000000000000000;
    gll.L[21] = -0.375000000000000;
    gll.L[22] = -0.375000000000000;
    gll.L[23] = 0.875000000000000;
    gll.L[24] = 0.875000000000000;

    return gll;
  }

  throw std::runtime_error("gen_GLL only supports ngll=5 right now.");
}

template <int ngllcapacity>
void quadrature_rule::sample_L(type_real buf[][ngllcapacity], type_real *t_vals,
                               int t_size) {
  for (int it = 0; it < t_size; it++) { // foreach t
    type_real tpow = 1;
    // reset accum
    for (int iL = 0; iL < nquad; iL++) {
      buf[it][iL] = 0;
    }
    for (int ipow = 0; ipow < nquad; ipow++) {
      // buf += L_{:,ipow} * t^ipow
      for (int iL = 0; iL < nquad; iL++) {
        buf[it][iL] += tpow * L[iL * nquad + ipow];
      }
      tpow *= t_vals[it];
    }
  }
}

template <int ngllcapacity>
type_real edge_intersection<ngllcapacity>::a_to_mortar(int mortar_index,
                                                       type_real *quantity) {
  type_real val = 0;
  for (int i = 0; i < a_ngll; i++) {
    val += a_mortar_trans[mortar_index][i] * quantity[i];
  }
  return val;
}
template <int ngllcapacity>
type_real edge_intersection<ngllcapacity>::b_to_mortar(int mortar_index,
                                                       type_real *quantity) {
  type_real val = 0;
  for (int i = 0; i < b_ngll; i++) {
    val += b_mortar_trans[mortar_index][i] * quantity[i];
  }
  return val;
}

} // namespace edge_manager
} // namespace _util

#endif
