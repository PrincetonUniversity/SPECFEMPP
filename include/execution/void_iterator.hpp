#pragma once

#include "policy.hpp"

namespace specfem {
namespace execution {

/**
 * @brief Base level iterator when there is no valid index to iterate over.
 *
 */
class VoidIterator : public VoidPolicy {
public:
  using base_policy_type = VoidPolicy; ///< Base policy type
  using policy_index_type = void;      ///< Index type for the policy
  using index_type = void;             ///< Index type for the iterator

  /**
   * @brief Returns an empty index type.
   *
   */
  KOKKOS_INLINE_FUNCTION
  constexpr index_type operator()() const { return; }
};

} // namespace execution
} // namespace specfem
