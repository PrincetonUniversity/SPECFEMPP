
#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag> class mpi;
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass>
class mpi_buffer;

namespace mpi_impl {

/**
 * @brief Base class storing common interface metadata shared by all MPI
 * communication group types (face, edge, corner).
 *
 * Captures per-connection data that is independent of the connection geometry:
 * - The mesh entity type (orientation) of each shared interface in both the
 *   local and neighboring element
 * - The spectral element indices for the local and neighboring elements
 *
 * Subclasses extend this with geometry-specific transformation metadata:
 * - `face_communication_group`: adds `theta` (discrete 2D rotation index)
 * - `edge_communication_group`: adds `reflect` (1D direction flag)
 * - `corner_communication_group`: no additional data (single GLL point)
 *
 * Memory is organized as 1D `Kokkos::View` arrays indexed by interface ID
 * within the group, enabling efficient GPU-accelerated synchronization kernels.
 *
 * @see face_communication_group, edge_communication_group,
 *      corner_communication_group
 * @see mpi – Collects one group per connection type per neighbor MPI rank
 */
class communication_group {
public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int n; /**< Number of shared interfaces in this communication group
                    (faces, edges, or corners) */
  unsigned int ngll; /**< Number of GLL points per face dimension */

  using FaceNormalViewType = Kokkos::View<specfem::mesh_entity::dim3::type *,
                                          Kokkos::DefaultExecutionSpace>;

  using ElementIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  using MediumTagViewType = Kokkos::View<specfem::element::medium_tag *,
                                         Kokkos::DefaultExecutionSpace>;

  FaceNormalViewType my_orientation; /**< Mesh entity type of each face in the
                                     local element (nfaces) */
  FaceNormalViewType neighbor_orientation; /**< Mesh entity type of each face in
                                           the neighboring element (nfaces) */
  FaceNormalViewType::host_mirror_type h_my_orientation; /**< Host mirror for
                                                   my_orientation */
  FaceNormalViewType::host_mirror_type h_neighbor_orientation; /**< Host mirror
                                                         for
                                                         neighbor_orientation */
  ElementIndexViewType my_element; /**< Spectral element index of the local
                                   element for each face (nfaces) */
  ElementIndexViewType neighbor_element; /**< Spectral element index of the
                                         neighboring element for each face
                                         (nfaces) */
  ElementIndexViewType::host_mirror_type h_my_element; /**< Host mirror for
                                                 my_element */
  ElementIndexViewType::host_mirror_type h_neighbor_element; /**< Host mirror
                                                       for neighbor_element */

  MediumTagViewType connection_medium_tag; /**< Medium tag of the local element
                                              for each interface (n) */
  MediumTagViewType::host_mirror_type h_connection_medium_tag; /**< Host mirror
  for connection_medium_tag */

  communication_group() : n(0), ngll(0) {}

  /**
   * @brief Constructs the common interface metadata for a neighbor pair.
   *
   * @param my_rank          MPI rank of the current subdomain
   * @param neighbor_rank    MPI rank of the neighboring subdomain
   * @param edges            Interface connectivity data from
   *                         adjacency_graph.mpi_connections()
   * @param element_types    Element type mapping for medium-tag lookup
   * @param ngllz, nglly, ngllx  GLL points per element dimension
   */
  communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const specfem::assembly::element_types<dimension_tag> element_types,
      const int ngllz, const int nglly, const int ngllx);
};

/**
 * @brief Communication group for corner-shared MPI interfaces.
 *
 * A corner interface consists of a single shared GLL point. No transformation
 * metadata is required because a point has no orientation.
 *
 * @see communication_group
 */
class corner_communication_group : public communication_group {
public:
  corner_communication_group() = default;
  corner_communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const specfem::assembly::element_types<dimension_tag> element_types,
      const int ngllz, const int nglly, const int ngllx);
};

/**
 * @brief Communication group for edge-shared MPI interfaces.
 *
 * An edge interface is a 1D line of GLL points. The `reflect` flag encodes
 * whether the two partitions traverse the edge in the same direction.
 *
 * @see communication_group
 */
class edge_communication_group : public communication_group {
public:
  using ReflectViewType = Kokkos::View<bool *, Kokkos::DefaultExecutionSpace>;
  ReflectViewType reflect; /**< Per-edge reflection flag: true → reversed
                              traversal direction relative to neighbor */
  ReflectViewType::host_mirror_type h_reflect; /**< Host mirror for reflect */

  edge_communication_group() = default;
  edge_communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const specfem::assembly::element_types<dimension_tag> element_types,
      const int ngllz, const int nglly, const int ngllx);
};

/**
 * @brief Communication group for face-shared MPI interfaces.
 *
 * A face interface is a 2D ngll × ngll grid of GLL points. The `theta` index
 * encodes the discrete 90° rotation between the two partitions' coordinate
 * systems: theta=0 (identity), 1 (90° CW), 2 (180°), 3 (270° CW).
 *
 * @see communication_group
 */
class face_communication_group : public communication_group {
public:
  using ThetaViewType =
      Kokkos::View<unsigned char *, Kokkos::DefaultExecutionSpace>;
  ThetaViewType theta; /**< Discrete rotation index ∈ [0,3] aligning face
                          coordinate systems across the MPI boundary */
  ThetaViewType::host_mirror_type h_theta; /**< Host mirror for theta */

  face_communication_group() = default;
  face_communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const specfem::assembly::element_types<dimension_tag> element_types,
      const int ngllz, const int nglly, const int ngllx);
};

/**
 * @brief Computes the compact mapping from unique MPI-surface GLL points to
 * per-medium global assembled indices (iglob), and sends the transformed index
 * mapping to the neighbor.
 */
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct packer {
public:
  constexpr static auto field_type = FieldType;       ///< Simulation field type
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag
  constexpr static auto medium_tag = MediumTag;       ///< Medium tag
  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int ncorners;      /**< Number of corners in communication group */
  unsigned int nedges;        /**< Number of edges in communication group */
  unsigned int nfaces;        /**< Number of faces in communication group */
  unsigned int ngll;          /**< GLL points per face dimension */

  unsigned int nglob; /**< Number of unique GLL points on the MPI surface */

  using IndexMappingView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  IndexMappingView mapping; /**< Maps local nglob index to global iglob */
  IndexMappingView::host_mirror_type h_mapping; /**< Host mirror for mapping */

  packer() : nfaces(0), nedges(0), ncorners(0), ngll(0), nglob(0) {}

  /**
   * @param corner_group, edge_group, face_group  Communication groups for this
   *        neighbor, classified by interface geometry
   * @param element  Element object for mapping face/edge/corner coordinates to
   *                 element 3D coordinates
   * @param fields   Assembly fields providing the per-medium iglob lookup
   */
  packer(const corner_communication_group &corner_group,
         const edge_communication_group &edge_group,
         const face_communication_group &face_group,
         const specfem::mesh_entity::element<dimension_tag> &element,
         const specfem::assembly::fields<dimension_tag> &fields);

private:
  /**
   * @brief Sends transformed GLL indices for all interface types to the
   * neighboring MPI process using 7 blocking MPI messages (base_tag+0..+6):
   *   +0  Metadata: {nfaces, nedges, ncorners, ngll, nglob}
   *   +1  Rotated face indices [nfaces][ngll][ngll]
   *   +2  Face neighbor element indices [nfaces]
   *   +3  Reflected edge indices [nedges][ngll]
   *   +4  Edge neighbor element indices [nedges]
   *   +5  Corner neighbor element indices [ncorners]
   *   +6  Corner nglob indices [ncorners]
   */
  void send_unpacking_indices(
      Kokkos::View<int ***, Kokkos::HostSpace> face_index_mapping,
      Kokkos::View<unsigned char *, Kokkos::HostSpace> theta,
      Kokkos::View<int *, Kokkos::HostSpace> neighbor_element,
      Kokkos::View<int **, Kokkos::HostSpace> edge_index_mapping,
      Kokkos::View<bool *, Kokkos::HostSpace> edge_reflect,
      Kokkos::View<int *, Kokkos::HostSpace> edge_neighbor_element,
      Kokkos::View<int *, Kokkos::HostSpace> corner_neighbor_element,
      Kokkos::View<int *, Kokkos::HostSpace> corner_index_mapping);
};

/**
 * @brief Receives the neighbor's index mapping and builds the local unpack
 * mapping from neighbor nglob indices to local per-medium iglobs.
 */
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct unpacker {
public:
  constexpr static auto field_type = FieldType;       ///< Simulation field type
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag
  constexpr static auto medium_tag = MediumTag;       ///< Medium tag

  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int nfaces;        /**< Number of faces in communication group */
  unsigned int nedges;        /**< Number of edges in communication group */
  unsigned int ncorners;      /**< Number of corners in communication group */
  unsigned int ngll;          /**< GLL points per face dimension */
  unsigned int nglob;         /**< Number of unique GLL points received from
                                   the neighbor MPI surface */

  using IndexMappingView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  IndexMappingView mapping; /**< Maps neighbor nglob index to local iglob */
  IndexMappingView::host_mirror_type h_mapping;

  using OrientationHostView =
      Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>;
  using ElementHostView = Kokkos::View<int *, Kokkos::HostSpace>;

  OrientationHostView h_face_orientations; /**< Filtered local face orientations
                                              (medium-tag filtered) */
  OrientationHostView h_edge_orientations; /**< Filtered local edge orientations
                                              (medium-tag filtered) */
  OrientationHostView h_corner_orientations; /**< Filtered local corner
                                                orientations (medium-tag
                                                filtered) */
  ElementHostView h_face_elements;           /**< Filtered local face elements
                                                in compute ordering */
  ElementHostView h_edge_elements;           /**< Filtered local edge elements
                                                in compute ordering */
  ElementHostView h_corner_elements;         /**< Filtered local corner elements
                                                in compute ordering */

  unpacker()
      : my_rank(0), neighbor_rank(0), nfaces(0), nedges(0), ncorners(0),
        ngll(0), nglob(0) {}

  unpacker(const corner_communication_group &corner_group,
           const edge_communication_group &edge_group,
           const face_communication_group &face_group,
           const specfem::mesh_entity::element<dimension_tag> &element);

  /**
   * @brief Posts all 7 non-blocking MPI_Irecv calls and returns the buffers.
   *
   * @return Tuple of (requests[7], metadata[5], face_indices, face_elements,
   *         edge_indices, edge_elements, corner_nglob_idx, corner_elements)
   */
  std::tuple<std::array<MPI_Request, 7>,
             Kokkos::View<unsigned int[5], Kokkos::HostSpace>,
             Kokkos::View<int ***, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace>,
             Kokkos::View<int **, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace>>
  receive_unpacking_buffers();

  /**
   * @brief Waits for all receives, validates metadata, and builds h_mapping.
   */
  void assemble_unpacking_mapping(
      const std::array<MPI_Request, 7> &requests,
      const Kokkos::View<unsigned int[5], Kokkos::HostSpace> metadata_buf,
      const Kokkos::View<int ***, Kokkos::HostSpace> recv_face_indices,
      const Kokkos::View<int *, Kokkos::HostSpace> face_elements,
      const Kokkos::View<int **, Kokkos::HostSpace> recv_edge_indices,
      const Kokkos::View<int *, Kokkos::HostSpace> edge_elements,
      const Kokkos::View<int *, Kokkos::HostSpace> corner_nglob_idx,
      const Kokkos::View<int *, Kokkos::HostSpace> corner_elements,
      const Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
          my_face_orientations,
      const Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
          my_edge_orientations,
      const Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
          my_corner_orientations,
      const specfem::mesh_entity::element<dimension_tag> &element,
      const specfem::assembly::fields<dimension_tag> &fields,
      const Kokkos::View<const int *, Kokkos::HostSpace> &h_mesh_to_compute);
};

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct communication_pattern {
public:
  constexpr static auto field_type = FieldType;       ///< Simulation field type
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag
  constexpr static auto medium_tag = MediumTag;       ///< Medium tag
  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unpacker<field_type, dimension_tag, medium_tag> unpack; /**< Unpacker for
                                                 receiving data from neighbor */
  packer<field_type, dimension_tag, medium_tag> pack; /**< Packer for sending
                                                         data to neighbor */

  communication_pattern() = default;

  using PendingReceiveState =
      std::tuple<std::array<MPI_Request, 7>,
                 Kokkos::View<unsigned int[5], Kokkos::HostSpace>,
                 Kokkos::View<int ***, Kokkos::HostSpace>,
                 Kokkos::View<int *, Kokkos::HostSpace>,
                 Kokkos::View<int **, Kokkos::HostSpace>,
                 Kokkos::View<int *, Kokkos::HostSpace>,
                 Kokkos::View<int *, Kokkos::HostSpace>,
                 Kokkos::View<int *, Kokkos::HostSpace>>;

  /**
   * @brief Phase 1: initialises ranks, constructs the unpacker, and posts all
   * MPI_Irecv calls. Returns the pending receive state that must be passed to
   * complete_construction() after ALL neighbors have completed phase 1.
   */
  PendingReceiveState
  post_receives(const corner_communication_group &corner_group,
                const edge_communication_group &edge_group,
                const face_communication_group &face_group,
                const specfem::mesh_entity::element<dimension_tag> &element);

  /**
   * @brief Phase 2+3: performs blocking sends (packer), then waits for the
   * receives posted in post_receives() and assembles the unpack mapping.
   * Must be called only after post_receives() has been called for ALL
   * neighbors.
   */
  void complete_construction(
      const corner_communication_group &corner_group,
      const edge_communication_group &edge_group,
      const face_communication_group &face_group,
      const specfem::mesh_entity::element<dimension_tag> &element,
      const specfem::assembly::fields<dimension_tag> &fields,
      const Kokkos::View<const int *, Kokkos::HostSpace> &h_mesh_to_compute,
      PendingReceiveState pending);
};

} // namespace mpi_impl

/**
 * @brief Manages MPI communication groups and patterns for 3D distributed
 * simulations.
 *
 * @see face_communication_group, edge_communication_group,
 *      corner_communication_group
 */
template <> class mpi<specfem::element::dimension_tag::dim3> {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  /// Media that participate in 3D cross-rank MPI exchange.
  static constexpr auto media = MEDIUM_SET(elastic, acoustic);

  /// Medium combinations for 3D MPI communication patterns
  static constexpr auto medium_combinations =
      specfem::tag_dispatch::dimension_set<dimension_tag>{} * media;

  /// Per-medium map from neighbor rank to communication pattern
  template <typename TagsType>
  using ForwardCommunicationPatternMap =
      std::unordered_map<unsigned int,
                         mpi_impl::communication_pattern<
                             specfem::simulation::field_type::forward,
                             TagsType::dimension_tag, TagsType::medium_tag>>;

  template <typename TagsType>
  using BackwardCommunicationPatternMap =
      std::unordered_map<unsigned int,
                         mpi_impl::communication_pattern<
                             specfem::simulation::field_type::backward,
                             TagsType::dimension_tag, TagsType::medium_tag>>;

  template <typename TagsType>
  using AdjointCommunicationPatternMap =
      std::unordered_map<unsigned int,
                         mpi_impl::communication_pattern<
                             specfem::simulation::field_type::adjoint,
                             TagsType::dimension_tag, TagsType::medium_tag>>;

  specfem::simulation::type simulation; ///< Simulation mode (forward, combined,
                                        ///< etc.)

  std::vector<mpi_impl::corner_communication_group> corner_groups;
  std::vector<mpi_impl::edge_communication_group> edge_groups;
  std::vector<mpi_impl::face_communication_group> face_groups;

  /// Communication patterns keyed by neighbor rank, one map per medium
  specfem::tag_dispatch::TypedStorage<ForwardCommunicationPatternMap,
                                      decltype(medium_combinations)>
      forward_communication_patterns;
  specfem::tag_dispatch::TypedStorage<BackwardCommunicationPatternMap,
                                      decltype(medium_combinations)>
      backward_communication_patterns;
  specfem::tag_dispatch::TypedStorage<AdjointCommunicationPatternMap,
                                      decltype(medium_combinations)>
      adjoint_communication_patterns;

  mpi() = default;

  /**
   * @brief Constructs all MPI communication groups and communication patterns.
   *
   * @param adjacency_graph  Mesh adjacency graph containing MPI connection data
   * @param simulation       Simulation mode controlling which field-type
   *                         communication maps are instantiated
   * @param fields           Assembly fields (any field type; iglob is
   *                         field-type-independent)
   * @param h_mesh_to_compute  Host map from mesh-order element indices to
   *                           compute-order element indices
   * @param ngllz, nglly, ngllx  GLL points per element dimension
   */
  mpi(const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::simulation::type simulation,
      const specfem::assembly::fields<dimension_tag> &fields,
      const Kokkos::View<const int *, Kokkos::HostSpace> &h_mesh_to_compute,
      const int ngllz, const int nglly, const int ngllx);

  template <specfem::simulation::field_type FieldType,
            specfem::element::medium_tag MediumTag,
            specfem::data_access::DataClassType DataClass>
  mpi_buffer<FieldType, dimension_tag, MediumTag, DataClass>
  create_mpi_buffer() const;
};

namespace mpi_impl {
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass>
struct mpi_buffer {
public:
  constexpr static auto field_type = FieldType;       ///< Simulation field type
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto data_class = DataClass; ///< Data class type
  constexpr static unsigned int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  using BufferViewType =
      Kokkos::View<type_real *[components], Kokkos::DefaultExecutionSpace>;

  using MappingViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  unsigned int my_rank;
  unsigned int neighbor_rank;

  MPI_Request send_request;
  MPI_Request recv_request;

  BufferViewType send_buffer; /**< Device buffer for outgoing field data */
  BufferViewType recv_buffer; /**< Device buffer for incoming field data */

  MappingViewType pack_mapping;   /**< Pack iglob mapping (from packer) */
  MappingViewType unpack_mapping; /**< Unpack iglob mapping (from unpacker) */

#ifndef SPECFEM_CUDA_AWARE_MPI
  BufferViewType::host_mirror_type h_send_buffer;
  BufferViewType::host_mirror_type h_recv_buffer;
#endif

  mpi_buffer() = default;

  mpi_buffer(const mpi_impl::communication_pattern<field_type, dimension_tag,
                                                   medium_tag> &comm_pattern) {
    this->my_rank = comm_pattern.my_rank;
    this->neighbor_rank = comm_pattern.neighbor_rank;
    // Copy iglob mappings from communication pattern for use during pack/unpack
    pack_mapping = comm_pattern.pack.mapping;
    unpack_mapping = comm_pattern.unpack.mapping;

    send_buffer =
        BufferViewType("mpi_buffer::send_buffer", comm_pattern.pack.nglob);
    recv_buffer =
        BufferViewType("mpi_buffer::recv_buffer", comm_pattern.unpack.nglob);
#ifndef SPECFEM_CUDA_AWARE_MPI
    h_send_buffer = Kokkos::create_mirror_view(send_buffer);
    h_recv_buffer = Kokkos::create_mirror_view(recv_buffer);
#endif
    // Initialize MPI request handles to prevent undefined behavior
    send_request = MPI_REQUEST_NULL;
    recv_request = MPI_REQUEST_NULL;
  }

  void pack(const specfem::assembly::simulation_field<dimension_tag, field_type>
                &field);

  void send();
  void receive();

  void
  unpack(specfem::assembly::simulation_field<dimension_tag, field_type> &field);
};
} // namespace mpi_impl

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass>
class mpi_buffer {

private:
  int max_connections_per_process =
      0; // This should be set to the maximum number of connections per process
         // across all ranks

public:
  constexpr static auto field_type = FieldType;
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto data_class = DataClass;

  using BufferMap = std::unordered_map<
      unsigned int,
      mpi_impl::mpi_buffer<field_type, dimension_tag, medium_tag, DataClass>>;

  BufferMap buffers;

  mpi_buffer() = default;

  mpi_buffer(const mpi<dimension_tag> &mpi_obj) {
    using TagsType = specfem::tags::Tags<dimension_tag, medium_tag>;
    const auto *patterns_ptr = [&]() {
      if constexpr (field_type == specfem::simulation::field_type::forward) {
        return &mpi_obj.forward_communication_patterns.template get<TagsType>();
      } else if constexpr (field_type ==
                           specfem::simulation::field_type::backward) {
        return &mpi_obj.backward_communication_patterns
                    .template get<TagsType>();
      } else if constexpr (field_type ==
                           specfem::simulation::field_type::adjoint) {
        return &mpi_obj.adjoint_communication_patterns.template get<TagsType>();
      } else {
        static_assert(field_type == specfem::simulation::field_type::forward,
                      "Unsupported field type for mpi_buffer");
        return &mpi_obj.forward_communication_patterns.template get<TagsType>();
      }
    }();
    const auto &patterns = *patterns_ptr;

    buffers.clear();
    buffers.reserve(patterns.size());
    for (const auto &[neighbor_rank, comm_pattern] : patterns) {
      buffers.emplace(
          neighbor_rank,
          mpi_impl::mpi_buffer<field_type, dimension_tag, medium_tag,
                               DataClass>(comm_pattern));
    }

    this->max_connections_per_process = buffers.size();
    SPECFEM_MPI_SAFECALL(
        MPI_Allreduce(MPI_IN_PLACE, &this->max_connections_per_process, 1,
                      MPI_INT, MPI_MAX, specfem::MPI::communicator()));
  }

  void pack(const specfem::assembly::simulation_field<dimension_tag, field_type>
                &field) {
    for (auto &entry : buffers)
      entry.second.pack(field);
  }

  void unpack(
      specfem::assembly::simulation_field<dimension_tag, field_type> &field) {
    for (auto &entry : buffers)
      entry.second.unpack(field);
  }

  void send() {
    for (auto &entry : buffers)
      entry.second.send();
  }

  void receive() {
    for (auto &entry : buffers)
      entry.second.receive();
  }

  void wait();
};

} // namespace specfem::assembly

#include "mpi.tpp"
