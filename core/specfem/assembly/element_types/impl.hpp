#pragma once

#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace specfem::assembly::element_types_impl {

// ---------------------------------------------------------------------------
// Helper: allocate a device view + host mirror, fill indices where pred
// holds, then deep-copy to device.
// ---------------------------------------------------------------------------
template <typename DevView, typename HostView, typename Pred>
void fill_index_views(DevView &dev_view, HostView &host_view,
                      const std::string &label, const int nspec, Pred &&pred) {
  int count = 0;
  for (int ispec = 0; ispec < nspec; ispec++)
    if (pred(ispec))
      count++;

  dev_view = DevView(label, count);
  host_view = Kokkos::create_mirror_view(dev_view);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++)
    if (pred(ispec))
      host_view(index++) = ispec;

  Kokkos::deep_copy(dev_view, host_view);
}

} // namespace specfem::assembly::element_types_impl
