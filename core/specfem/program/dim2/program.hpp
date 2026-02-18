#pragma once

#include "specfem/program.hpp"

namespace specfem::program {

void program_2d(
    const YAML::Node &parameter_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim2> > >
        tasks);

} // namespace specfem::program
