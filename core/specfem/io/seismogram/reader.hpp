#ifndef SPECFEM_READER_SEISMOGRAM_HPP
#define SPECFEM_READER_SEISMOGRAM_HPP

#include "enumerations/specfem_enums.hpp"
#include "specfem/io/reader.hpp"

namespace specfem {
namespace source_time_functions {
class external;
} // namespace source_time_functions
} // namespace specfem

namespace specfem {
namespace io {

/**
 * @brief Reader for loading seismogram data from files
 *
 * Reads recorded seismograms in various formats and stores them in memory
 * for use as external source time functions or data processing.
 */
class seismogram_reader {
public:
  seismogram_reader() {};
  seismogram_reader(const char *filename,
                    const specfem::enums::seismogram::format type,
                    Kokkos::View<type_real **, Kokkos::LayoutRight,
                                 Kokkos::DefaultHostExecutionSpace>
                        source_time_function)
      : filename(filename), type(type),
        source_time_function(source_time_function) {}
  seismogram_reader(const std::string &filename,
                    const specfem::enums::seismogram::format type,
                    Kokkos::View<type_real **, Kokkos::LayoutRight,
                                 Kokkos::DefaultHostExecutionSpace>
                        source_time_function)
      : filename(filename), type(type),
        source_time_function(source_time_function) {}
  void read();

private:
  std::string filename;
  type_real dt;
  specfem::enums::seismogram::format type;
  Kokkos::View<type_real **, Kokkos::LayoutRight,
               Kokkos::DefaultHostExecutionSpace>
      source_time_function;
};
} // namespace io
} // namespace specfem

#endif /* SPECFEM_READER_SEISMOGRAM_HPP */
