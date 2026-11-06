#pragma once

#include "specfem/enums.hpp"
#include "specfem/io/xdmf/description.hpp"
#include <boost/filesystem.hpp>
#include <string>
#include <vector>

namespace specfem {
namespace io {
namespace xdmf {

/**
 * @brief Write an XDMF description of per-GLL element-group datasets.
 *
 * The file is Xdmf 2.0, read by VTK's `vtkXdmfReader` (ParaView's "XDMF
 * Reader"). It holds one spatial collection with one uniform grid per element
 * group: a quadrilateral (2D) or hexahedral (3D) topology built from the
 * group's "connectivity" dataset, an `X_Y` or `X_Y_Z` geometry from the
 * coordinate datasets, one node-centered attribute per stored dataset and
 * one `Function` attribute per derived quantity. Heavy data references are
 * `<heavy_file_name>:/<group>/<dataset>`, relative to the `.xmf` file.
 *
 * @tparam DimensionTag Dimension of the mesh
 * @param xdmf_path Path of the `.xmf` file to write
 * @param heavy_file_name HDF5 file name as referenced from the `.xmf`, e.g.
 *        "Properties.h5"
 * @param groups Element groups stored in the heavy file
 * @throws std::runtime_error if the file cannot be opened or a derived
 *         quantity names an operand that is not a stored dataset of its group
 */
template <specfem::element::dimension_tag DimensionTag>
void write_element_groups(
    const boost::filesystem::path &xdmf_path,
    const std::string &heavy_file_name,
    const std::vector<specfem::io::xdmf::ElementGroupDescription<DimensionTag>>
        &groups);

} // namespace xdmf
} // namespace io
} // namespace specfem
