#include "specfem/io/sources/read_sources.tpp"

// Explicit instantiation for dim3
template std::tuple<std::vector<std::shared_ptr<specfem::sources::source<
                        specfem::element::dimension_tag::dim3>>>,
                    type_real, std::optional<specfem::datetime::type>>
specfem::io::read_sources<specfem::element::dimension_tag::dim3>(
    const std::vector<specfem::enums::source_file_entry> &entries,
    const int nsteps, const type_real user_t0, const type_real dt,
    const specfem::simulation::type simulation_type);
