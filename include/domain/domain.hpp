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
 * Domain class serves as the driver used to compute the elemental kernels. For
 * example, method @c compute_stiffness_interaction is used to
 * implement Kokkos parallelization and loading memory to scratch spaces, which
 * are then used by the elemental implementation to update acceleration. The
 * goal the domain class is to provide a general Kokkos parallelization
 * framework which can be used by specialized elemental implementations. This
 * allows us to hide the Kokkos parallelization details from the end developer
 * when implementing new physics (i.e. specialized elements).
 *
 *
 * @tparam medium class defining the domain medium. Separate implementations
 * exist for elastic, acoustic or poroelastic media
 * @tparam quadrature_points class used to define the quadrature points either
 * at compile time or run time
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
