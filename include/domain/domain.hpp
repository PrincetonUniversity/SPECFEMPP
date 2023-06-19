#ifndef _DOMAIN_HPP
#define _DOMAIN_HPP

#include "compute/interface.hpp"
#include "impl/elements/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {

/**
 * @brief domain class
 *
 * @tparam medium class defining the domain medium. Separate implementations
 * exist for elastic, acoustic or poroelastic media
 * @tparam element type element type defines the base element type to be used in
 * the medium. Having a base element lets us define specialized kernels for
 * elemental operations.
 *
 * Check specfem::domain::impl::elements for more details on specialized
 * elemental kernels
 *
 * Domain implementation details:
 *  - field -> stores a 2 dimensional field along different components. The
 * components may vary based of medium type. For example Acoustic domain have
 * only 1 component i.e. potential, Elastic domain have 2 components i.e. X,Z
 */
template <class medium, class quadrature_points> class domain {};
} // namespace domain

} // namespace specfem

#endif
