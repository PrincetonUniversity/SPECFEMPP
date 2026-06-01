#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <vector>

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
template <specfem::data_access::DataClassType DCT>
void specfem::assembly::mpi_impl::
    mpi_buffer<FieldType, DimensionTag, MediumTag>::pack(
        const specfem::assembly::simulation_field<dimension_tag, field_type>
            &field) {

  const auto field_m = field.template get_field<MediumTag>();

  using data_class_tag =
      std::integral_constant<specfem::data_access::DataClassType, DCT>;

  auto mapping = pack_mapping;
  Kokkos::parallel_for(
      "assembly::mpi_buffer::pack",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                         send_buffer.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int &i) {
        const int iglob = mapping(i);
        for (unsigned int icomp = 0; icomp < components; icomp++)
          this->send_buffer(i, icomp) =
              field_m.template get_value<true>(data_class_tag{}, iglob, icomp);
      });
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
template <specfem::data_access::DataClassType DCT>
void specfem::assembly::mpi_impl::
    mpi_buffer<FieldType, DimensionTag, MediumTag>::unpack(
        specfem::assembly::simulation_field<dimension_tag, field_type> &field) {

  const auto field_m = field.template get_field<MediumTag>();

  using data_class_tag =
      std::integral_constant<specfem::data_access::DataClassType, DCT>;

  auto mapping = unpack_mapping;
  Kokkos::parallel_for(
      "assembly::mpi_buffer::unpack",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                         recv_buffer.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int &i) {
        const int iglob = mapping(i);
        for (unsigned int icomp = 0; icomp < components; icomp++)
          field_m.template get_value<true>(data_class_tag{}, iglob, icomp) +=
              this->recv_buffer(i, icomp);
      });
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::mpi_buffer<FieldType, DimensionTag,
                                             MediumTag>::send() {

  if (send_buffer.extent(0) == 0) {
    this->send_request = MPI_REQUEST_NULL;
    return;
  }

#ifdef SPECFEM_CUDA_AWARE_MPI
  SPECFEM_MPI_SAFECALL(
      MPI_Isend(this->send_buffer.data(), send_buffer.extent(0) * components,
                SPECFEM_MPI_TYPE_REAL, this->neighbor_rank,
                100 * static_cast<int>(MediumTag), specfem::MPI::communicator(),
                &this->send_request));
#else
  // For non-CUDA-aware MPI, copy device data to host mirror first
  Kokkos::deep_copy(this->h_send_buffer, this->send_buffer);
  SPECFEM_MPI_SAFECALL(
      MPI_Isend(this->h_send_buffer.data(), send_buffer.extent(0) * components,
                SPECFEM_MPI_TYPE_REAL, this->neighbor_rank,
                100 * static_cast<int>(MediumTag), specfem::MPI::communicator(),
                &this->send_request));
#endif

  return;
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::mpi_buffer<FieldType, DimensionTag,
                                             MediumTag>::receive() {

  if (recv_buffer.extent(0) == 0) {
    this->recv_request = MPI_REQUEST_NULL;
    return;
  }

  // Receiver uses the sender's tag base (neighbor is sender for recv)
#ifdef SPECFEM_CUDA_AWARE_MPI
  SPECFEM_MPI_SAFECALL(
      MPI_Irecv(this->recv_buffer.data(), recv_buffer.extent(0) * components,
                SPECFEM_MPI_TYPE_REAL, this->neighbor_rank,
                100 * static_cast<int>(MediumTag), specfem::MPI::communicator(),
                &this->recv_request));
#else
  // For non-CUDA-aware MPI, receive into host mirror
  SPECFEM_MPI_SAFECALL(
      MPI_Irecv(this->h_recv_buffer.data(), recv_buffer.extent(0) * components,
                SPECFEM_MPI_TYPE_REAL, this->neighbor_rank,
                100 * static_cast<int>(MediumTag), specfem::MPI::communicator(),
                &this->recv_request));
#endif

  return;
}

template <specfem::simulation::field_type FieldType,
          specfem::element::medium_tag MediumTag>
specfem::assembly::mpi_buffer<FieldType, specfem::element::dimension_tag::dim3,
                              MediumTag>
specfem::assembly::mpi<
    specfem::element::dimension_tag::dim3>::create_mpi_buffer() const {

  if (simulation == specfem::simulation::type::forward) {
    if (FieldType != specfem::simulation::field_type::forward) {
      specfem::Logger::error(
          "mpi::create_mpi_buffer: simulation type mismatch with field type");
    }
  } else if (simulation == specfem::simulation::type::combined) {
    if ((FieldType != specfem::simulation::field_type::backward) &&
        (FieldType != specfem::simulation::field_type::adjoint)) {
      specfem::Logger::error(
          "mpi::create_mpi_buffer: simulation type mismatch with field type");
    }
  } else {
    specfem::Logger::error(
        "mpi::create_mpi_buffer: unsupported simulation type for mpi_buffer");
  }

  return mpi_buffer<FieldType, dimension_tag, MediumTag>(*this);
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_buffer<FieldType, DimensionTag, MediumTag>::wait() {
  // Collect all MPI_Request handles from the buffer map
  std::vector<MPI_Request> requests;
  for (const auto &entry : buffers) {
    const auto &buf = entry.second;
    requests.push_back(buf.send_request);
    requests.push_back(buf.recv_request);
  }

  if (requests.empty()) {
    return; // No requests to wait on
  }

  std::vector<MPI_Status> statuses(requests.size());

  // Cast size_t to int to satisfy MPI_Waitall signature
  int request_count = static_cast<int>(requests.size());
  SPECFEM_MPI_SAFECALL(
      MPI_Waitall(request_count, requests.data(), statuses.data()));

#if !defined(NDEBUG) && defined(SPECFEM_ENABLE_MPI)
  // In debug mode, check the statuses for errors
  for (size_t i = 0; i < requests.size(); ++i) {
    if (statuses[i].MPI_ERROR != MPI_SUCCESS) {
      fprintf(stderr, "MPI error in request %zu: %d\n", i,
              statuses[i].MPI_ERROR);
    }
  }
#endif

#ifndef SPECFEM_CUDA_AWARE_MPI
  // For non-CUDA-aware MPI, copy received data from host mirrors to device
  for (auto &entry : buffers) {
    auto &buf = entry.second;
    Kokkos::deep_copy(buf.recv_buffer, buf.h_recv_buffer);
  }
#endif

  return;
}
