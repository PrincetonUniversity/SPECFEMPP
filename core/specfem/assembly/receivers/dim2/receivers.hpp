#pragma once

#include "../impl/receiver_iterator.hpp"
#include "enumerations/interface.hpp"
#include "specfem/receivers.hpp"

namespace specfem::assembly {

/**
 * @brief 2D assembly receiver specialization for seismic simulations
 *
 * Specialized implementation of receivers for 2D spectral element simulations.
 * This class manages seismic receivers located within 2D finite element meshes,
 * handling Lagrange interpolation for accurate field sampling and coordinate
 * transformations using receiver angles for proper seismogram orientation.
 *
 * Key features for 2D:
 * - Supports multiple medium types: elastic_psv, elastic_sh, acoustic,
 * poroelastic
 * - Uses angle-based coordinate rotation (sine/cosine) for 2D transformations
 * - Records 2-component seismograms (typically horizontal and vertical)
 * - Efficient Kokkos-based data structures for GPU computations
 *
 * The class inherits from both StationIterator (for station metadata) and
 * SeismogramIterator (for time-series data access), providing a unified
 * interface for receiver management and seismogram recording.
 */
template <>
struct receivers<specfem::dimension::type::dim2>
    : public receivers_impl::StationIterator,
      public receivers_impl::SeismogramIterator<
          specfem::dimension::type::dim2> {

public:
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag for this assembly

private:
  using IndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                          ///< elements
                                                          ///< associated with
                                                          ///< the receivers
  using LagrangeInterpolantType =
      Kokkos::View<type_real ***[2], Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                   ///< Lagrange interpolant for
                                                   ///< every receiver

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
      const int nspec, const int ngllz, const int ngllx, const int max_sig_step,
      const type_real dt, const type_real t0, const int nsteps_between_samples,
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

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC,
                                    ISOTROPIC_COSSERAT)),
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
 * @defgroup ComputeReceiversDataAccess2D
 * @brief 2D receiver data access functions for device computations
 *
 * These functions provide efficient access to receiver data during GPU kernel
 * execution for 2D spectral element simulations. They handle loading of
 * Lagrange interpolants and storing of seismogram components with proper
 * indexing.
 */

/**
 * @brief Load the Lagrange interpolant for receivers associated with the
 * iterator on the device
 *
 * @ingroup ComputeReceiversDataAccess2D
 *
 * @tparam ChunkIndexType Chunk index type @ref
 * specfem::execution::ChunkElementIndex
 * @tparam ViewType Lagrange interpolant associated with the receivers in the
 * iterator
 *
 * @param chunk_index Chunk index
 * @param receivers Receivers object containing the receiver information
 * @param lagrange_interpolant Lagrange interpolant associated with the
 * receivers in the iterator
 */
template <typename ChunkIndexType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const ChunkIndexType &chunk_index,
               const receivers<specfem::dimension::type::dim2> &receivers,
               ViewType &lagrange_interpolant) {

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        const int irec = index.imap;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        lagrange_interpolant(ielement, index.iz, index.ix, 0) =
            receivers.lagrange_interpolant(irec, index.iz, index.ix, 0);
        lagrange_interpolant(ielement, index.iz, index.ix, 1) =
            receivers.lagrange_interpolant(irec, index.iz, index.ix, 1);
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
 * @ingroup ComputeReceiversDataAccess2D
 * @tparam ChunkIndexType Chunk index type
 * @tparam SeismogramViewType View of the seismogram components
 * @param receivers Receivers object containing the receiver information
 */
template <typename ChunkIndexType, typename SeismogramViewType>
KOKKOS_FUNCTION void
store_on_device(const ChunkIndexType &chunk_index,
                const SeismogramViewType &seismogram_components,
                const receivers<specfem::dimension::type::dim2> &receivers) {

  const int isig_step = receivers.get_seismogram_step();
  const int iseis = receivers.get_seis_type();

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        const int irec = index.imap;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        receivers.seismogram_components(isig_step, iseis, irec, 0) =
            seismogram_components(ielement, 0);
        receivers.seismogram_components(isig_step, iseis, irec, 1) =
            seismogram_components(ielement, 1);
      });
  return;
}

} // namespace specfem::assembly
