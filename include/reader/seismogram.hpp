#ifndef SPECFEM_READER_SEISMOGRAM_HPP
#define SPECFEM_READER_SEISMOGRAM_HPP

#include "enumerations/specfem_enums.hpp"
#include "reader/reader.hpp"

namespace specfem {
namespace forcing_function {
class external;
} // namespace forcing_function
} // namespace specfem

namespace specfem {
namespace reader {

class seismogram : public reader {
public:
  seismogram(const char *filename,
             const specfem::enums::seismogram::format type,
             specfem::kokkos::HostView2d<type_real> source_time_function)
      : filename(filename), type(type),
        source_time_function(source_time_function) {}
  seismogram(const std::string &filename,
             const specfem::enums::seismogram::format type,
             specfem::kokkos::HostView2d<type_real> source_time_function)
      : filename(filename), type(type),
        source_time_function(source_time_function) {}
  void read() override;

private:
  std::string filename;
  type_real dt;
  specfem::enums::seismogram::format type;
  specfem::kokkos::HostView2d<type_real> source_time_function;
};
} // namespace reader
} // namespace specfem

#endif /* SPECFEM_READER_SEISMOGRAM_HPP */
