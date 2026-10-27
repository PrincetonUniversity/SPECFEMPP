#pragma once

#include "specfem/program.hpp"

namespace specfem::program {

void program_3d(
    const YAML::Node &parameter_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim3>>>
        tasks,
    const bool globe = false,
    const std::optional<specfem::injection::PluginOptions> &injection =
        std::nullopt,
    const specfem::injection::MethodRegistry &registry =
        specfem::injection::default_method_registry());

} // namespace specfem::program
