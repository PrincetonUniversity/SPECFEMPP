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
void specfem::assembly::mpi_impl::
    mpi_buffer<FieldType, DimensionTag, MediumTag>::pack(
        const specfem::assembly::simulation_field<dimension_tag, field_type>
            &field) {

  const auto field_m = field.template get_field<MediumTag>();

  using acc_tag =
      std::integral_constant<specfem::data_access::DataClassType,
                             specfem::data_access::DataClassType::acceleration>;

  auto mapping = pack_mapping;
  Kokkos::parallel_for(
      "assembly::mpi_buffer::pack",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                         send_buffer.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int &i) {
        const int iglob = mapping(i);
        for (unsigned int icomp = 0; icomp < components; icomp++)
          this->send_buffer(i, icomp) =
              field_m.template get_value<true>(acc_tag{}, iglob, icomp);
      });
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::
    mpi_buffer<FieldType, DimensionTag, MediumTag>::unpack(
        specfem::assembly::simulation_field<dimension_tag, field_type> &field) {

  const auto field_m = field.template get_field<MediumTag>();

  using acc_tag =
      std::integral_constant<specfem::data_access::DataClassType,
                             specfem::data_access::DataClassType::acceleration>;

  auto mapping = unpack_mapping;
  Kokkos::parallel_for(
      "assembly::mpi_buffer::unpack",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                         recv_buffer.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int &i) {
        const int iglob = mapping(i);
        for (unsigned int icomp = 0; icomp < components; icomp++)
          field_m.template get_value<true>(acc_tag{}, iglob, icomp) +=
              this->recv_buffer(i, icomp);
      });
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::mpi_buffer<FieldType, DimensionTag,
                                             MediumTag>::send() {
  const auto &tag_generator = specfem::MPI::message_tag_generator;

  // message_tag_base is precomputed at construction; +7 is the field-data slot
  SPECFEM_MPI_SAFECALL(
      MPI_Isend(this->send_buffer.data(), send_buffer.extent(0) * components,
                SPECFEM_MPI_TYPE_REAL, this->neighbor_rank,
                tag_generator(this->my_rank, this->neighbor_rank, 0),
                specfem::MPI::communicator(), &this->send_request));

  return;
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::mpi_buffer<FieldType, DimensionTag,
                                             MediumTag>::receive() {
  // Receiver uses the sender's tag base (neighbor is sender for recv)
  const auto &tag_generator = specfem::MPI::message_tag_generator;
  SPECFEM_MPI_SAFECALL(
      MPI_Irecv(this->recv_buffer.data(), recv_buffer.extent(0) * components,
                SPECFEM_MPI_TYPE_REAL, this->neighbor_rank,
                tag_generator(this->neighbor_rank, this->my_rank, 0),
                specfem::MPI::communicator(), &this->recv_request));

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

  std::vector<MPI_Status> statuses(requests.size());

  SPECFEM_MPI_SAFECALL(
      MPI_Waitall(requests.size(), requests.data(), statuses.data()));

#ifndef NDEBUG
  // In debug mode, check the statuses for errors
  for (size_t i = 0; i < requests.size(); ++i) {
    if (statuses[i].MPI_ERROR != MPI_SUCCESS) {
      fprintf(stderr, "MPI error in request %zu: %d\n", i,
              statuses[i].MPI_ERROR);
    }
  }
#endif

  auto &tag_generator = specfem::MPI::message_tag_generator;
  tag_generator.reset();

  return;
}
