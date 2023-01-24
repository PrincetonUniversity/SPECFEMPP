#ifndef DOMAIN_H
#define DOMAIN_H

#include "../include/compute.h"
#include "../include/config.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace Domain {
class Domain {

public:
  virtual specfem::DeviceView2d<type_real> get_field() const {
    return this->field;
  }
  virtual specfem::HostMirror2d<type_real> get_host_field() const {
    return this->h_field;
  }
  virtual specfem::DeviceView2d<type_real> get_field_dot() const {
    return this->field_dot;
  }
  virtual specfem::HostMirror2d<type_real> get_host_field_dot() const {
    return this->h_field_dot;
  }
  virtual specfem::DeviceView2d<type_real> get_field_dot_dot() const {
    return this->field_dot_dot;
  }
  virtual specfem::HostMirror2d<type_real> get_host_field_dot_dot() const {
    return this->h_field_dot_dot;
  }
  virtual specfem::DeviceView2d<type_real> get_rmass_inverse() const {
    return this->rmass_inverse;
  }
  virtual specfem::HostMirror2d<type_real> get_host_rmass_inverse() const {
    return this->h_rmass_inverse;
  }

  virtual void compute_stiffness_interaction(){};
  virtual void divide_mass_matrix(){};
  virtual void compute_source_interaction(const type_real timeval){};

  virtual void sync_field(specfem::sync::kind kind){};
  virtual void sync_field_dot(specfem::sync::kind kind){};
  virtual void sync_field_dot_dot(specfem::sync::kind kind){};
  virtual void sync_rmass_inverse(specfem::sync::kind kind){};

private:
  specfem::DeviceView2d<type_real> field;
  specfem::HostMirror2d<type_real> h_field;
  specfem::DeviceView2d<type_real> field_dot;
  specfem::HostMirror2d<type_real> h_field_dot;
  specfem::DeviceView2d<type_real> field_dot_dot;
  specfem::HostMirror2d<type_real> h_field_dot_dot;
  specfem::DeviceView2d<type_real> rmass_inverse;
  specfem::HostMirror2d<type_real> h_rmass_inverse;
};

class Elastic : public Domain {
public:
  /**
   * @brief Get a view of displacement
   *
   * @return specfem::DeviceView2d<type_real> View of the displacement
   */
  specfem::DeviceView2d<type_real> get_field() const override {
    return this->field;
  }
  specfem::HostMirror2d<type_real> get_host_field() const override {
    return this->h_field;
  }
  /**
   * @brief Get a view of velocity
   *
   * @return specfem::DeviceView2d<type_real> view of the velocity
   */
  specfem::DeviceView2d<type_real> get_field_dot() const override {
    return this->field_dot;
  }
  specfem::HostMirror2d<type_real> get_host_field_dot() const override {
    return this->h_field_dot;
  }
  /**
   * @brief Get a view of acceleration
   *
   * @return specfem::DeviceView2d<type_real> view of the acceleration
   */
  specfem::DeviceView2d<type_real> get_field_dot_dot() const override {
    return this->field_dot_dot;
  }
  specfem::HostMirror2d<type_real> get_host_field_dot_dot() const override {
    return this->h_field_dot_dot;
  }
  /**
   * @brief Get a view of acceleration
   *
   * @return specfem::DeviceView2d<type_real> view of the inverse of diagonal
   * elements of mass matrix (For GLL / GLJ quadrature mass matrix is diagonal)
   */
  specfem::DeviceView2d<type_real> get_rmass_inverse() const override {
    return this->rmass_inverse;
  }
  specfem::HostMirror2d<type_real> get_host_rmass_inverse() const override {
    return this->h_rmass_inverse;
  }

  /**
   * @brief Construct a new Elastic domain object
   *
   * @param ndim Number of dimensions
   * @param nglob Total number of distinct quadrature points inside the domain
   * @param material_properties Pointer to specfem::compute::properties struct
   * used to store material properties
   * @param quadx Pointer to quadrature object in x-dimension
   * @param quadx Pointer to quadrature object in z-dimension
   */
  Elastic(const int ndim, const int nglob, specfem::compute::compute *compute,
          specfem::compute::properties *material_properties,
          specfem::compute::partial_derivatives *partial_derivatives,
          specfem::compute::sources *sources, quadrature::quadrature *quadx,
          quadrature::quadrature *quadz);

  void compute_stiffness_interaction() override;
  void divide_mass_matrix() override;
  void compute_source_interaction(const type_real timeval) override;
  void assign_views();

  void sync_field(specfem::sync::kind kind) override;
  void sync_field_dot(specfem::sync::kind kind) override;
  void sync_field_dot_dot(specfem::sync::kind kind) override;
  void sync_rmass_inverse(specfem::sync::kind kind) override;

private:
  specfem::DeviceView2d<type_real> field; ///< Displacement inside elastic
                                          ///< domain
  specfem::HostMirror2d<type_real> h_field;
  specfem::DeviceView2d<type_real> field_dot; ///< Velocity inside elastic
                                              ///< domain
  specfem::HostMirror2d<type_real> h_field_dot;
  specfem::DeviceView2d<type_real> field_dot_dot; ///< Acceleration inside
                                                  ///< elastic domain
  specfem::HostMirror2d<type_real> h_field_dot_dot;
  specfem::DeviceView2d<type_real> rmass_inverse; ///< Inverse of mass matrix
                                                  ///< inside elastic domain
  specfem::HostMirror2d<type_real> h_rmass_inverse;
  specfem::compute::compute *compute;
  specfem::compute::properties *material_properties; ///< Pointer to struct used
                                                     ///< to store material
                                                     ///< properties
  specfem::compute::partial_derivatives *partial_derivatives;
  specfem::compute::sources *sources;
  quadrature::quadrature *quadx; ///< Pointer to quadrature object in
                                 ///< x-dimension
  quadrature::quadrature *quadz; ///< Pointer to quadrature object in
                                 ///< z-dimension
  int nelem_domain;              ///< Total number of elements in this domain
  specfem::DeviceView1d<int> ispec_domain; ///< Array containing global
                                           ///< indices(ispec) of all elements
                                           ///< in this domain
  specfem::HostMirror1d<int> h_ispec_domain;
};
} // namespace Domain
} // namespace specfem

#endif
