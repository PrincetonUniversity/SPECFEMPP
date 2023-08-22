#ifndef _ELASTIC_DOMAIN_TPP
#define _ELASTIC_DOMAIN_TPP

#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/domain.hpp"
#include "domain/elastic/elastic_domain.hpp"
#include "domain/impl/elements/interface.hpp"
#include "domain/impl/receivers/interface.hpp"
#include "domain/impl/sources/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

// Type aliases
// ----------------------------------------------------------------------------

template <typename element_type>
using element_container =
    typename specfem::domain::impl::elements::container<element_type>;

template <class qp_type, class... traits>
using element_type = typename specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic, qp_type, traits...>;

template <typename source_type>
using source_container =
    typename specfem::domain::impl::sources::container<source_type>;

template <class qp_type, class... traits>
using source_type = typename specfem::domain::impl::sources::source<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic, qp_type, traits...>;

template <typename receiver_type>
using receiver_container =
    typename specfem::domain::impl::receivers::container<receiver_type>;

template <class qp_type, class... traits>
using receiver_type = typename specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic, qp_type, traits...>;

// ----------------------------------------------------------------------------

template <class medium>
void initialize_views(
    const int &nglob,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        rmass_inverse) {

  constexpr int components = medium::components;

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::initiaze_views",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 }, { nglob, components }),
      KOKKOS_LAMBDA(const int iglob, const int idim) {
        field(iglob, idim) = 0;
        field_dot(iglob, idim) = 0;
        field_dot_dot(iglob, idim) = 0;
        rmass_inverse(iglob, idim) = 0;
      });
}

template <class medium>
void initialize_rmass_inverse(
    const specfem::kokkos::DeviceView3d<int> ibool,
    const specfem::kokkos::DeviceView1d<specfem::enums::element::type>
        ispec_type,
    const specfem::kokkos::DeviceView1d<type_real> wxgll,
    const specfem::kokkos::DeviceView1d<type_real> wzgll,
    const specfem::kokkos::DeviceView3d<type_real> rho,
    const specfem::kokkos::DeviceView3d<type_real> jacobian,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        rmass_inverse) {
  // Compute the mass matrix

  constexpr int components = medium::components;
  constexpr auto value = medium::value;

  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  const int nglob = rmass_inverse.extent(0);

  specfem::kokkos::DeviceScatterView2d<type_real, Kokkos::LayoutLeft> results(
      rmass_inverse);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_mass_matrix",
      specfem::kokkos::DeviceMDrange<3, Kokkos::Iterate::Left>(
          { 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int ix) {
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = rho(ispec, iz, ix);
        auto access = results.access();
        if (ispec_type(ispec) == value) {
          for (int icomponent = 0; icomponent < components; icomponent++) {
            access(iglob, icomponent) +=
                wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, iz, ix);
          }
        }
      });

  Kokkos::Experimental::contribute(rmass_inverse, results);

  // invert the mass matrix
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::Invert_mass_matrix",
      specfem::kokkos::DeviceRange(0, nglob), KOKKOS_LAMBDA(const int iglob) {
        if (rmass_inverse(iglob, 0) > 0.0) {
          for (int icomponent = 0; icomponent < components; icomponent++) {
            rmass_inverse(iglob, icomponent) =
                1.0 / rmass_inverse(iglob, icomponent);
          }
        } else {
          for (int icomponent = 0; icomponent < components; icomponent++) {
            rmass_inverse(iglob, icomponent) = 1.0;
          }
        }
      });
}

template <class qp_type>
void instantialize_element(
    specfem::compute::partial_derivatives partial_derivatives,
    specfem::compute::properties properties,
    specfem::kokkos::DeviceView1d<int> ispec_domain,
    specfem::kokkos::DeviceView1d<element_container<element_type<qp_type> > >
        elements) {
  auto h_elements = Kokkos::create_mirror_view(elements);

  Kokkos::parallel_for(
      "specfem::domain::elastic_isotropic::allocate_memory",
      specfem::kokkos::HostRange(0, h_elements.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        h_elements(i).element = (element_type<qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                element_type<qp_type,
                             specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(elements, h_elements);

  Kokkos::parallel_for(
      "specfem::domain::elastic_isotropic::instantialize_element",
      specfem::kokkos::DeviceRange(0, ispec_domain.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        const int ispec = ispec_domain(i);
        auto &element = elements(ispec).element;
        new (element)
            element_type<qp_type, specfem::enums::element::property::isotropic>(
                ispec, partial_derivatives, properties);
      });

  return;
}

template <class qp_type>
void initialize_sources(
    const specfem::compute::properties &properties,
    const specfem::compute::sources &compute_sources,
    specfem::kokkos::DeviceView1d<source_container<source_type<qp_type> > >
        &sources) {

  const auto h_ispec_type = properties.h_ispec_type;
  const auto ispec_type = properties.ispec_type;
  const auto h_ispec_array = compute_sources.h_ispec_array;
  const auto ispec_array = compute_sources.ispec_array;
  int nsources_domain = 0;

  specfem::kokkos::DeviceView1d<int> my_recs;

  // Check how many sources belong to this domain
  for (int ispec = 0; ispec < h_ispec_array.extent(0); ispec++) {
    if (h_ispec_type(h_ispec_array(ispec)) ==
        specfem::enums::element::elastic) {
      nsources_domain++;
    }
  }

  sources =
      specfem::kokkos::DeviceView1d<source_container<source_type<qp_type> > >(
          "specfem::domain::elastic_isotropic::sources", nsources_domain);

  specfem::kokkos::DeviceView1d<int> my_sources(
      "specfem::domain::elastic_isotropic::my_sources", nsources_domain);

  auto h_sources = Kokkos::create_mirror_view(sources);
  auto h_my_sources = Kokkos::create_mirror_view(my_sources);

  // Store which sources belong to this domain
  int index = 0;
  for (int isource = 0; isource < h_ispec_array.extent(0); isource++) {
    if (h_ispec_type(h_ispec_array(isource)) ==
        specfem::enums::element::elastic) {
      h_my_sources(index) = isource;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index == nsources_domain);
#endif

  Kokkos::deep_copy(my_sources, h_my_sources);

  // Allocate memory for the sources on the device
  Kokkos::parallel_for(
      "specfem::domain::elastic_isotropic::allocate_memory",
      specfem::kokkos::HostRange(0, nsources_domain),
      KOKKOS_LAMBDA(const int i) {
        h_sources(i).source = (source_type<qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                source_type<qp_type,
                            specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(sources, h_sources);

  Kokkos::parallel_for(
      "specfem::domain::elastic_isotropic::initialize_source",
      specfem::kokkos::DeviceRange(0, nsources_domain),
      KOKKOS_LAMBDA(const int isource) {
        auto &source = sources(isource).source;
        const int ispec = ispec_array(my_sources(isource));
        auto source_time_function =
            compute_sources.stf_array(my_sources(isource)).T;
        const auto sv_source_array =
            Kokkos::subview(compute_sources.source_array, my_sources(isource),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        new (source)
            source_type<qp_type, specfem::enums::element::property::isotropic>(
                ispec, sv_source_array, source_time_function);
      });

  return;
}

template <class qp_type>
void initialize_receivers(
    const specfem::compute::receivers compute_receivers,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::properties &properties,
    specfem::kokkos::DeviceView1d<receiver_container<receiver_type<qp_type> > >
        &receivers) {

  const auto h_ispec_type = properties.h_ispec_type;
  const auto h_ispec_array = compute_receivers.h_ispec_array;
  const auto ispec_array = compute_receivers.ispec_array;
  const auto seis_types = compute_receivers.seismogram_types;

  int nreceivers_domain = 0;

  // Check how many receivers belong to this domain
  for (int irec = 0; irec < ispec_array.extent(0); irec++) {
    if (h_ispec_type(h_ispec_array(irec)) == specfem::enums::element::elastic) {
      nreceivers_domain++;
    }
  }

  nreceivers_domain = nreceivers_domain * seis_types.extent(0);

  receivers = specfem::kokkos::DeviceView1d<
      receiver_container<receiver_type<qp_type> > >(
      "specfem::domain::elastic_isotropic::receivers", nreceivers_domain);

  specfem::kokkos::DeviceView1d<int> my_receivers(
      "specfem::domain::elastic_isotropic::my_recs", nreceivers_domain);

  auto h_my_receivers = Kokkos::create_mirror_view(my_receivers);
  auto h_receivers = Kokkos::create_mirror_view(receivers);

  // Store which receivers belong to this domain
  int index = 0;
  for (int irec = 0; irec < ispec_array.extent(0); irec++) {
    if (h_ispec_type(h_ispec_array(irec)) == specfem::enums::element::elastic) {
      h_my_receivers(index) = irec;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index * seis_types.extent(0) == nreceivers_domain);
#endif

  Kokkos::parallel_for(
      "specfem::domain::elastic_isotropic::allocate_memory",
      specfem::kokkos::HostRange(0, nreceivers_domain),
      KOKKOS_LAMBDA(const int i) {
        h_receivers(i).receiver = (receiver_type<qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                receiver_type<qp_type,
                              specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(receivers, h_receivers);
  Kokkos::deep_copy(my_receivers, h_my_receivers);

  Kokkos::parallel_for(
      "specfem::domain::elastic_isotropic::initialize_receiver",
      specfem::kokkos::DeviceRange(0, nreceivers_domain),
      KOKKOS_LAMBDA(const int inum) {
        const int irec = my_receivers(inum / seis_types.extent(0));
        const int iseis = inum % seis_types.extent(0);

        const int ispec = ispec_array(irec);
        const auto seis_type = seis_types(iseis);

        const type_real cos_rec = compute_receivers.cos_recs(irec);
        const type_real sin_rec = compute_receivers.sin_recs(irec);

        auto sv_receiver_array =
            Kokkos::subview(compute_receivers.receiver_array, irec, Kokkos::ALL,
                            Kokkos::ALL, Kokkos::ALL);

        // Subview the seismogram array at current receiver and current
        // seismogram value
        auto sv_receiver_seismogram =
            Kokkos::subview(compute_receivers.seismogram, Kokkos::ALL, iseis,
                            irec, Kokkos::ALL);
        auto sv_receiver_field =
            Kokkos::subview(compute_receivers.receiver_field, Kokkos::ALL, irec,
                            iseis, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

        auto &receiver = receivers(inum).receiver;
        new (receiver)
            receiver_type<qp_type,
                          specfem::enums::element::property::isotropic>(
                ispec, sin_rec, cos_rec, seis_type, sv_receiver_array,
                sv_receiver_seismogram, partial_derivatives, properties,
                sv_receiver_field);
      });

  return;
}

template <class qp_type>
void assign_elemental_properties(
    specfem::compute::partial_derivatives partial_derivatives,
    specfem::compute::properties properties,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView1d<element_container<element_type<qp_type> > >
        &elements,
    const int &nspec, const int &ngllz, const int &ngllx, int &nelem_domain) {

  // Assign elemental properties
  // ----------------------------------------------------------------------
  nelem_domain = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) ==
        specfem::enums::element::medium::elastic::value) {
      nelem_domain++;
    }
  }

  specfem::kokkos::DeviceView1d<int> ispec_domain(
      "specfem::domain::elastic::h_ispec_domain", nelem_domain);
  specfem::kokkos::HostMirror1d<int> h_ispec_domain =
      Kokkos::create_mirror_view(ispec_domain);

  elements =
      specfem::kokkos::DeviceView1d<element_container<element_type<qp_type> > >(
          "specfem::domain::elastic::elements", nelem_domain);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) == specfem::enums::element::elastic) {
      h_ispec_domain(index) = ispec;
      index++;
    }
  }

  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  instantialize_element(partial_derivatives, properties, ispec_domain,
                        elements);
};

template <class qp_type>
specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>::
    domain(const int ndim, const int nglob, const qp_type &quadrature_points,
           specfem::compute::compute *compute,
           specfem::compute::properties material_properties,
           specfem::compute::partial_derivatives partial_derivatives,
           specfem::compute::sources compute_sources,
           specfem::compute::receivers compute_receivers,
           specfem::quadrature::quadrature *quadx,
           specfem::quadrature::quadrature *quadz)
    : field(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field", nglob,
          specfem::enums::element::medium::elastic::components)),
      field_dot(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field_dot", nglob,
          specfem::enums::element::medium::elastic::components)),
      field_dot_dot(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::field_dot_dot", nglob,
              specfem::enums::element::medium::elastic::components)),
      rmass_inverse(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::rmass_inverse", nglob,
              specfem::enums::element::medium::elastic::components)),
      quadrature_points(quadrature_points), compute(compute), quadx(quadx),
      quadz(quadz) {

  this->h_field = Kokkos::create_mirror_view(this->field);
  this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  //----------------------------------------------------------------------------
  // Initialize views

  initialize_views<specfem::enums::element::medium::elastic>(
      nglob, this->field, this->field_dot, this->field_dot_dot,
      this->rmass_inverse);

  //----------------------------------------------------------------------------
  // Inverse of mass matrix

  initialize_rmass_inverse<specfem::enums::element::medium::elastic>(
      compute->ibool, material_properties.ispec_type, quadx->get_w(),
      quadz->get_w(), material_properties.rho, partial_derivatives.jacobian,
      this->rmass_inverse);

  // ----------------------------------------------------------------------------
  // Inverse of mass matrix

  assign_elemental_properties(partial_derivatives, material_properties,
                              this->field_dot_dot, this->elements, nspec, ngllz,
                              ngllx, this->nelem_domain);

  // ----------------------------------------------------------------------------
  // Initialize the sources

  initialize_sources(material_properties, compute_sources, this->sources);

  // ----------------------------------------------------------------------------
  // Initialize the receivers

  initialize_receivers(compute_receivers, partial_derivatives,
                       material_properties, this->receivers);

  return;
};

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_field(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_field_dot(specfem::sync::kind
                                                          kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot, field_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot, h_field_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_field_dot_dot(specfem::sync::kind
                                                              kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_rmass_inverse(specfem::sync::kind
                                                              kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_rmass_inverse, rmass_inverse);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(rmass_inverse, h_rmass_inverse);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::divide_mass_matrix() {

  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::divide_mass_matrix",
      specfem::kokkos::DeviceRange(
          0, specfem::enums::element::medium::elastic::components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        this->field_dot_dot(iglob, idim) =
            this->field_dot_dot(iglob, idim) * this->rmass_inverse(iglob, idim);
      });

  // Kokkos::fence();

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::compute_stiffness_interaction() {

  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();
  const auto ibool = this->compute->ibool;

  int scratch_size =
      2 *
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::x,
                                            specfem::enums::axes::x>();

  scratch_size +=
      2 *
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::z,
                                            specfem::enums::axes::z>();

  scratch_size +=
      6 *
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::x,
                                            specfem::enums::axes::z>();

  scratch_size +=
      quadrature_points.template shmem_size<int, specfem::enums::axes::x,
                                            specfem::enums::axes::z>();

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_gradients",
      specfem::kokkos::DeviceTeam(this->nelem_domain, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto element = this->elements(team_member.league_rank());
        const auto ispec = element.get_ispec();

        // Instantiate shared views
        // ---------------------------------------------------------------
        auto s_hprime_xx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprime_zz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));
        auto s_hprimewgll_xx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprimewgll_zz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));

        auto s_fieldx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_fieldz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_1 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_2 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_3 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_4 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_iglob =
            quadrature_points.template ScratchView<int, specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));

        // ---------- Allocate shared views -------------------------------
        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix) = hprime_xx(iz, ix);
              s_hprimewgll_xx(ix, iz) = wxgll(iz) * hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix) = hprime_zz(iz, ix);
              s_hprimewgll_zz(ix, iz) = wzgll(iz) * hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              const int iglob = ibool(ispec, iz, ix);
              s_fieldx(iz, ix) = field(iglob, 0);
              s_fieldz(iz, ix) = field(iglob, 1);
              s_stress_integral_1(iz, ix) = 0.0;
              s_stress_integral_2(iz, ix) = 0.0;
              s_stress_integral_3(iz, ix) = 0.0;
              s_stress_integral_4(iz, ix) = 0.0;
              s_iglob(iz, ix) = iglob;
            });

        // ------------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              type_real duxdxl = 0.0;
              type_real duxdzl = 0.0;
              type_real duzdxl = 0.0;
              type_real duzdzl = 0.0;

              element.compute_gradient(xz, s_hprime_xx, s_hprime_zz, s_fieldx,
                                       s_fieldz, &duxdxl, &duxdzl, &duzdxl,
                                       &duzdzl);

              element.compute_stress(
                  xz, duxdxl, duxdzl, duzdxl, duzdzl,
                  &s_stress_integral_1(iz, ix), &s_stress_integral_2(iz, ix),
                  &s_stress_integral_3(iz, ix), &s_stress_integral_4(iz, ix));
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);

              const int iglob = s_iglob(iz, ix);
              const type_real wxglll = wxgll(ix);
              const type_real wzglll = wzgll(iz);

              auto sv_field_dot_dot =
                  Kokkos::subview(field_dot_dot, iglob, Kokkos::ALL());

              element.update_acceleration(
                  xz, wxglll, wzglll, s_stress_integral_1, s_stress_integral_2,
                  s_stress_integral_3, s_stress_integral_4, s_hprimewgll_xx,
                  s_hprimewgll_zz, sv_field_dot_dot);
            });
      });
  return;
}

template <typename qp_type>
void specfem::domain::domain<
    specfem::enums::element::medium::elastic,
    qp_type>::compute_source_interaction(const type_real timeval) {

  const int nsources = this->sources.extent(0);
  const auto ibool = this->compute->ibool;

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_source_interaction",
      specfem::kokkos::DeviceTeam(nsources, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        int isource = team_member.league_rank();
        const auto source = this->sources(isource);
        const int ispec = source.get_ispec();
        auto sv_ibool = Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);

        type_real stf;

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, 1),
            [=](const int &, type_real &lsum) {
              lsum = source.eval_stf(timeval);
            },
            stf);

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec, iz, ix);

              type_real accelx, accelz;
              auto sv_field_dot_dot =
                  Kokkos::subview(field_dot_dot, iglob, Kokkos::ALL());

              source.compute_interaction(xz, stf, &accelx, &accelz);
              source.update_acceleration(accelx, accelz, sv_field_dot_dot);
            });
      });
  return;
}

template <typename qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::compute_seismogram(const int isig_step) {

  // Allocate scratch views for field, field_dot & field_dot_dot. Incase of
  // acostic domains when calculating displacement, velocity and acceleration
  // seismograms we need to compute the derivatives of the field variables. This
  // requires summing over all lagrange derivatives at all quadrature points
  // within the element. Scratch views speed up this computation by limiting
  // global memory accesses.

  const auto ibool = this->compute->ibool;
  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  // hprime_xx
  int scratch_size =
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::x,
                                            specfem::enums::axes::x>();

  // hprime_zz
  scratch_size +=
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::z,
                                            specfem::enums::axes::z>();

  // field, field_dot, field_dot_dot - x and z components
  scratch_size +=
      6 *
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::z,
                                            specfem::enums::axes::x>();

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_seismogram",
      specfem::kokkos::DeviceTeam(this->receivers.extent(0), Kokkos::AUTO, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const int irec = team_member.league_rank();
        const auto receiver = this->receivers(irec);
        const int ispec = receiver.get_ispec();
        const auto seismogram_type = receiver.get_seismogram_type();

        // Instantiate shared views
        // ----------------------------------------------------------------
        auto s_hprime_xx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_hprime_zz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));

        auto s_fieldx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_fieldx_dot = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_fieldx_dot_dot = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_fieldz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_fieldz_dot = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_fieldz_dot_dot = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        // Allocate shared views
        // ----------------------------------------------------------------
        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix) = hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::z>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix) = hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec, iz, ix);
              s_fieldx(iz, ix) = this->field(iglob, 0);
              s_fieldx_dot(iz, ix) = this->field_dot(iglob, 0);
              s_fieldx_dot_dot(iz, ix) = this->field_dot_dot(iglob, 0);
              s_fieldz(iz, ix) = this->field(iglob, 1);
              s_fieldz_dot(iz, ix) = this->field_dot(iglob, 1);
              s_fieldz_dot_dot(iz, ix) = this->field_dot_dot(iglob, 1);
            });

        // Get seismogram field
        // ----------------------------------------------------------------

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              receiver.get_field(xz, isig_step, s_fieldx, s_fieldz,
                                 s_fieldx_dot, s_fieldz_dot, s_fieldx_dot_dot,
                                 s_fieldz_dot_dot, s_hprime_xx, s_hprime_zz);
            });

        // compute seismograms components
        //-------------------------------------------------------------------
        switch (seismogram_type) {
        case specfem::enums::seismogram::type::displacement:
        case specfem::enums::seismogram::type::velocity:
        case specfem::enums::seismogram::type::acceleration:
          dimension::array_type<type_real> seismogram_components;
          Kokkos::parallel_reduce(
              quadrature_points.template TeamThreadRange<
                  specfem::enums::axes::z, specfem::enums::axes::x>(
                  team_member),
              [=](const int xz,
                  dimension::array_type<type_real> &l_seismogram_components) {
                receiver.compute_seismogram_components(xz, isig_step,
                                                       l_seismogram_components);
              },
              specfem::kokkos::Sum<dimension::array_type<type_real> >(
                  seismogram_components));
          Kokkos::single(Kokkos::PerTeam(team_member), [=] {
            receiver.compute_seismogram(isig_step, seismogram_components);
          });
          break;
        }
      });
}

#endif
