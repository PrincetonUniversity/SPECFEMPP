#pragma once

#include "../impl/receiver_iterator.hpp"
#include "enumerations/interface.hpp"
#include "specfem/receivers.hpp"

namespace specfem::assembly {

template <>
struct receivers<specfem::dimension::type::dim3>
    : public receivers_impl::StationIterator,
      public receivers_impl::SeismogramIterator<
          specfem::dimension::type::dim3> {

public:
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag for this assembly

private:
  using IndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                          ///< elements
                                                          ///< associated with
                                                          ///< the receivers
  using LagrangeInterpolantType =
      Kokkos::View<type_real ****[3], Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                   ///< Lagrange interpolant for
                                                   ///< every receiver (nrec,
                                                   ///< nglly, ngllz, ngllx, 3)

public:
  /**
   * @brief Construct a new receivers object
   *
   */
  receivers() = default;

  /**
   * @brief Construct a new receivers object
   *
   * @param nspec Total Number of spectral elements in the domain
   * @param nglly Total Number of GLL points in the y-direction
   * @param ngllz Total Number of GLL points in the z-direction
   * @param ngllx Total Number of GLL points in the x-direction
   * @param max_sig_step Maximum number seismogram sample points
   * @param dt Time increament
   * @param t0 Initial time
   * @param nsteps_between_samples Number of time steps between samples
   * @param receivers Vector of receivers
   * @param stypes Vector of seismogram types (displacement, velocity,
   * acceleration, pressure, or rotation)
   * @param mesh Mesh object
   * @param tags Tags for every element in the mesh
   * @param properties Properties object
   */
  receivers(
      const int nspec, const int nglly, const int ngllz, const int ngllx,
      const int max_sig_step, const type_real dt, const type_real t0,
      const int nsteps_between_samples,
      const std::vector<
          std::shared_ptr<specfem::receivers::receiver<dimension_tag> > >
          &receivers,
      const std::vector<specfem::wavefield::type> &stypes,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::mesh::tags<dimension_tag> &tags,
      const specfem::assembly::element_types<dimension_tag> &element_types);

  /**
   * @brief Get the spectral element indices in which the receivers are located
   * on the device
   *
   * Returns only the indices of the elements that are associated with a
   * specific medium and property tag
   * @param medium Medium tag
   * @property property Property tag
   * @return Kokkos::View<int *, Kokkos::DefaultExecutionSpace> View of the
   * elements indices associated with the receivers
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
  get_indices_on_device(const specfem::element::medium_tag medium,
                        const specfem::element::property_tag property) const;

  /**
   * @brief Get the spectral element indices in which the receivers are located
   * on the host
   *
   * Returns only the indices of the elements that are associated with a
   * specific medium and property tag
   * @param medium Medium tag
   * @property property Property tag
   * @return Kokkos::View<int *, Kokkos::DefaultExecutionSpace> View of the
   * elements indices associated with the receivers
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
  get_indices_on_host(const specfem::element::medium_tag medium,
                      const specfem::element::property_tag property) const;

  /**
   * @brief Get the seismogram types
   *
   * @return std::vector<specfem::wavefield::type> Vector of seismogram types
   */
  std::vector<specfem::wavefield::type> get_seismogram_types() const {
    return seismogram_types_;
  }

  /**
   * @brief Get the station iterator
   *
   * @return const StationIterator& Iterator over stations
   */
  const receivers_impl::StationIterator &stations() const {
    return static_cast<const receivers_impl::StationIterator &>(*this);
  }

  /**
   * @brief Set rotation matrix for a receiver
   * @param irec Receiver index
   * @param rotation_matrix 3x3 rotation matrix (nrec, 3, 3) layout
   */
  void set_rotation_matrix(
      int irec,
      const std::array<std::array<type_real, 3>, 3> &rotation_matrix) {
    static_cast<receivers_impl::SeismogramIterator<dimension_tag> &>(*this)
        .set_rotation_matrix(irec, rotation_matrix);
  }

private:
  int nspec;              ///< Total number of spectral elements
  IndexViewType elements; ///< View to store the elements associated with the
                          ///< receivers
  IndexViewType::HostMirror h_elements; ///< Host view to store the
                                        ///< elements associated with the
                                        ///< receivers
  LagrangeInterpolantType lagrange_interpolant; ///< Lagrange interpolant for
                                                ///< every receiver
  LagrangeInterpolantType::HostMirror
      h_lagrange_interpolant; ///< Lagrange interpolant for every receiver
                              ///< stored on the host
  specfem::assembly::element_types<dimension_tag> element_types; ///< Element
                                                                 ///< types

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC),
                       PROPERTY_TAG(ISOTROPIC)),
                      DECLARE((IndexViewType, receiver_indices),
                              (IndexViewType::HostMirror, h_receiver_indices),
                              (IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))

  template <typename ChunkIndexType, typename ViewType>
  friend KOKKOS_FUNCTION void load_on_device(const ChunkIndexType &chunk_index,
                                             const receivers &receivers,
                                             ViewType &lagrange_interpolant);

  template <typename ChunkIndexType, typename SiesmogramViewType>
  friend KOKKOS_FUNCTION void
  store_on_device(const ChunkIndexType &chunk_index,
                  const SiesmogramViewType &seismogram_components,
                  const receivers &receivers);
};

/**
 * @defgroup ComputeReceiversDataAccess
 */

/**
 * @brief Load the Lagrange interpolant for receivers associated with the
 * iterator on the device
 *
 * @ingroup ComputeReceiversDataAccess
 *
 * @tparam ChunkIndexType Chunk index type @ref
 * specfem::execution::ChunkElementIndex
 * @tparam ViewType Lagrange interpolant associated with the receivers in the
 * iterator
 *
 * @param team_member Kokkos team member
 * @param chunk_index Chunk index
 * @param receivers Receivers object containing the receiver information
 * @param lagrange_interpolant Lagrange interpolant associated with the
 * receivers in the iterator
 */
template <typename ChunkIndexType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const ChunkIndexType &chunk_index,
               const receivers<specfem::dimension::type::dim3> &receivers,
               ViewType &lagrange_interpolant) {

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        const int irec = index.imap;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        // Load all three components for 3D
        lagrange_interpolant(ielement, index.iy, index.iz, index.ix, 0) =
            receivers.lagrange_interpolant(irec, index.iy, index.iz, index.ix,
                                           0);
        lagrange_interpolant(ielement, index.iy, index.iz, index.ix, 1) =
            receivers.lagrange_interpolant(irec, index.iy, index.iz, index.ix,
                                           1);
        lagrange_interpolant(ielement, index.iy, index.iz, index.ix, 2) =
            receivers.lagrange_interpolant(irec, index.iy, index.iz, index.ix,
                                           2);
      });

  return;
}

/**
 * @brief Store the seismogram components for receivers associated with the
 * iterator on the device
 *
 * Before you store the seismogram components, you need to set the seismogram
 * time step and type. You can do this by calling the following functions:
 * @c receivers.set_seismogram_step(isig_step);
 * @c receivers.set_seismogram_type(iseis);
 *
 * @ingroup ComputeReceiversDataAccess
 * @tparam ChunkIndexType Chunk index type
 * @tparam SeismogramViewType View of the seismogram components
 * @param receivers Receivers object containing the receiver information
 */
template <typename ChunkIndexType, typename SeismogramViewType>
KOKKOS_FUNCTION void
store_on_device(const ChunkIndexType &chunk_index,
                const SeismogramViewType &seismogram_components,
                const receivers<specfem::dimension::type::dim3> &receivers) {

  const int isig_step = receivers.get_seismogram_step();
  const int iseis = receivers.get_seis_type();

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        const int irec = index.imap;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        // Store all three components for 3D
        receivers.seismogram_components(isig_step, iseis, irec, 0) =
            seismogram_components(ielement, 0);
        receivers.seismogram_components(isig_step, iseis, irec, 1) =
            seismogram_components(ielement, 1);
        receivers.seismogram_components(isig_step, iseis, irec, 2) =
            seismogram_components(ielement, 2);
      });
  return;
}

} // namespace specfem::assembly
