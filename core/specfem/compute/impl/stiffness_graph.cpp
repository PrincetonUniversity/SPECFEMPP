// The only translation unit of specfem_compute that includes TensorOperations
// headers. It is excluded from unity batches in CMakeLists.txt.

#include "stiffness_graph.hpp"

#ifdef SPECFEM_ENABLE_TENSOROPS
#include <TensorOperations/LevelGraph.hpp>
#endif

#include <stdexcept>

template <int NGLL, typename Tags>
void specfem::compute::impl::compute_stiffness_graph(
    const specfem::assembly::assembly<Tags::dimension_tag> & /* assembly */,
    const specfem::datatype::ElementIndexRange & /* elements */) {
#ifdef SPECFEM_ENABLE_TENSOROPS
  throw std::runtime_error(
      "specfem::compute: the stiffness graph is not implemented yet.");
#else
  throw std::runtime_error(
      "specfem::compute: the stiffness graph requires SPECFEM++ built with "
      "SPECFEM_ENABLE_TENSOROPS=ON (and SPECFEM_TENSOROPS_ROOT pointing at a "
      "TensorOperations checkout).");
#endif
}

template void specfem::compute::impl::compute_stiffness_graph<
    5, specfem::compute::impl::stiffness_graph_tags>(
    const specfem::assembly::assembly<
        specfem::compute::impl::stiffness_graph_tags::dimension_tag> &,
    const specfem::datatype::ElementIndexRange &);
