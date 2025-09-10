#pragma once

namespace specfem::interface {

enum class interface_tag {
  elastic_acoustic, ///< Elastic to acoustic interface
  acoustic_elastic  ///< Acoustic to elastic interface
};

} // namespace specfem::interface
