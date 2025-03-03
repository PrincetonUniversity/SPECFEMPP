#ifndef SPECFEM_READER_SEISMOGRAM_HPP
#define SPECFEM_READER_SEISMOGRAM_HPP

#include "IO/reader.hpp"
#include "enumerations/specfem_enums.hpp"

namespace specfem {
namespace forcing_function {
class external;
} // namespace forcing_function
} // namespace specfem

namespace specfem {
namespace IO {

class seismogram_reader {
public:
  seismogram_reader(){};
  seismogram_reader(const char *filename,
                    const specfem::enums::seismogram::format type,
                    specfem::kokkos::HostView2d<type_real> source_time_function)
      : filename(filename), type(type),
        source_time_function(source_time_function) {}
  seismogram_reader(const std::string &filename,
                    const specfem::enums::seismogram::format type,
                    specfem::kokkos::HostView2d<type_real> source_time_function)
      : filename(filename), type(type),
        source_time_function(source_time_function) {}
  void read();

private:
  std::string filename;
  type_real dt;
  specfem::enums::seismogram::format type;
  specfem::kokkos::HostView2d<type_real> source_time_function;
};
} // namespace IO
} // namespace specfem

#endif /* SPECFEM_READER_SEISMOGRAM_HPP */
