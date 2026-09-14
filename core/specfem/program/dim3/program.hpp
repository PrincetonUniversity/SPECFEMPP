#pragma once

#include "specfem/program.hpp"

namespace specfem::program {

void program_3d(
    const YAML::Node &parameter_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim3>>>
        tasks,
    const bool globe = false);

} // namespace specfem::program
