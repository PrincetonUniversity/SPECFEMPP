#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "writer.hpp"
#include <boost/filesystem.hpp>
#ifdef NO_VTK
#include <sstream>
#endif

namespace specfem {
namespace writer {
class plot_wavefield : public writer {
public:
  plot_wavefield(const specfem::compute::assembly &assembly,
                 const specfem::display::format &output_format,
                 const specfem::display::wavefield &component,
                 const specfem::wavefield::type &wavefield,
                 const std::string &output_folder)
      : output_format(output_format), component(component),
        output_folder(output_folder), wavefield(wavefield), assembly(assembly) {
#ifdef NO_VTK
    std::ostringstream message;
    message << "Display section is not enabled, since SPECFEM++ was built "
               "without VTK\n"
            << "Please install VTK and rebuild SPECFEM++ with "
               "-DVTK_DIR=/path/to/vtk";
    throw std::runtime_error(message.str());
#endif
  }

  void write() override;

private:
  const specfem::display::format output_format;
  const specfem::display::wavefield component;
  const specfem::wavefield::type wavefield;
  const boost::filesystem::path output_folder;
  specfem::compute::assembly assembly;
};
} // namespace writer
} // namespace specfem
