#ifndef __UTIL_EDGE_STORAGES_CPP_
#define __UTIL_EDGE_STORAGES_CPP_

#include "_util/edge_storages.hpp"
#include <cmath>
#include <iostream>

namespace _util {
namespace edge_manager {

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
    const std::function<void(edge_intersection &,
                             edge_data<ngll, datacapacity> &,
                             edge_data<ngll, datacapacity> &)> &func) {
  if (!intersections_built) {
    build_intersections_on_host();
  }
  for (int i = 0; i < n_intersections; i++) {
    edge_intersection &ei = h_intersection_container(i);
    func(ei, h_edge_data_container(ei.a_ref_ind),
         h_edge_data_container(ei.b_ref_ind));
  }
  Kokkos::deep_copy(intersection_container, h_intersection_container);
  Kokkos::deep_copy(edge_data_container, h_edge_data_container);
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
               edge_intersection &intersection) {
#define intersect_eps 1e-5
#define intersect_eps2 (intersect_eps * intersect_eps)
  quadrature_rule gll = gen_GLL(ngll);

  // make sure second derivative is zero and ngll is correct
  if (a.ngll != ngll || b.ngll != ngll) {
    throw std::runtime_error(
        "bool intersect(edge_data,edge_data,edge_intersection): ngll different "
        "from expected.");
  }
  for (int i = 2; i < ngll; i++) {
    type_real xa = 0, xb = 0, za = 0, zb = 0;
    for (int j = 0; j < ngll; j++) {
      xa += a.x[j] * gll.L[j * ngll + i];
      xb += b.x[j] * gll.L[j * ngll + i];
      za += a.z[j] * gll.L[j * ngll + i];
      zb += b.z[j] * gll.L[j * ngll + i];
    }
    if (abs(xa) > 1e-4 || abs(xb) > 1e-4 || abs(za) > 1e-4 || abs(zb) > 1e-4) {
      std::cout << "t^" << i << " term fail\n";
      std::cout << "  a x-deriv " << xa << " from "
                << "[";
      for (int j = 0; j < ngll; j++) {
        std::cout << a.x[j] << ", ";
      }
      std::cout << "]\n";
      std::cout << "  a z-deriv " << za << " from "
                << "[";
      for (int j = 0; j < ngll; j++) {
        std::cout << a.z[j] << ", ";
      }
      std::cout << "]\n";
      std::cout << "  b x-deriv " << xb << " from "
                << "[";
      for (int j = 0; j < ngll; j++) {
        std::cout << b.x[j] << ", ";
      }
      std::cout << "]\n";
      std::cout << "  b z-deriv " << zb << " from "
                << "[";
      for (int j = 0; j < ngll; j++) {
        std::cout << b.z[j] << ", ";
      }
      std::cout << "]\n";
      throw std::runtime_error(
          "bool intersect(edge_data,edge_data,edge_intersection) currently "
          "supports only linear segments.");
    }
  }
  // center (param = 0)
  type_real x1a = 0.5 * (a.x[ngll - 1] + a.x[0]);
  type_real z1a = 0.5 * (a.z[ngll - 1] + a.z[0]);
  type_real x1b = 0.5 * (b.x[ngll - 1] + b.x[0]);
  type_real z1b = 0.5 * (b.z[ngll - 1] + b.z[0]);
  // deriv
  type_real x2a = 0.5 * (a.x[ngll - 1] - a.x[0]);
  type_real z2a = 0.5 * (a.z[ngll - 1] - a.z[0]);
  type_real x2b = 0.5 * (b.x[ngll - 1] - b.x[0]);
  type_real z2b = 0.5 * (b.z[ngll - 1] - b.z[0]);
  type_real cross = x2a * z2b - x2b * z2a;

  if (cross * cross / ((x2a * x2a + z2a * z2a) * (x2b * x2b + z2b * z2b)) <
      1e-8) {
    // sin^2 theta < 1e-8, so parallel; orth project points to find distance
    // between lines
    type_real dot_over_mag2 =
        ((x1a - x1b) * x2a + (z1a - z1b) * z2a) / (x2a * x2a + z2a * z2a);
    type_real orthx = (x1a - x1b) - x2a * dot_over_mag2;
    type_real orthz = (z1a - z1b) - z2a * dot_over_mag2;
    //(a1-b1) - a2( (a1 - b1) . (a2)/|a2|^2 )
    if (orthx * orthx + orthz * orthz > intersect_eps2) {
      // distance between lines is greater than eps, so no intersection
      return false;
    }
    // map b endpoints into a parameter
    type_real tb1 =
        ((b.x[0] - x1a) * x2a + (b.z[0] - x1a) * z2a) / (x2a * x2a + z2a * z2a);
    type_real tb2 =
        ((b.x[ngll - 1] - x1a) * x2a + (b.z[ngll - 1] - x1a) * z2a) /
        (x2a * x2a + z2a * z2a);
    intersection.a_param_start = std::max((type_real)-1.0, std::min(tb1, tb2));
    intersection.a_param_end = std::min((type_real)1.0, std::max(tb1, tb2));
    if (intersection.a_param_end - intersection.a_param_start < intersect_eps) {
      // intersection is of (reference coordinate) length < eps.
      return false;
    }

    // a param -> b param (a1 + ta*a2 = b1 + tb*b2)
    if (abs(x2b) > abs(z2b)) {
      intersection.b_param_start =
          (x1a + intersection.a_param_start * x2a - x1b) / x2b;
      intersection.b_param_end =
          (x1a + intersection.a_param_end * x2a - x1b) / x2b;
    } else {
      intersection.b_param_start =
          (z1a + intersection.a_param_start * z2a - z1b) / z2b;
      intersection.b_param_end =
          (z1a + intersection.a_param_end * z2a - z1b) / z2b;
    }
    return true;
  }
  // not parallel, so intersection has zero measure (single point).
  return false;
#undef intersect_eps2
#undef intersect_eps
}

template <int ngll, int datacapacity>
void edge_storage<ngll, datacapacity>::build_intersections_on_host() {
  std::vector<edge_intersection> intersections;
  edge_intersection intersection;
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
  intersection_container = specfem::kokkos::DeviceView1d<edge_intersection>(
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
} // namespace edge_manager
} // namespace _util

#endif
