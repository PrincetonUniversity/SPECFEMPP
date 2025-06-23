#pragma once

#include "policy.hpp"

namespace specfem {
namespace execution {

/**
 * @brief Base level iterator when there is no valid index to iterate over.
 *
 * @tparam ExecutionSpace The execution space type where iterator closure will
 * be executed.
 */
template <typename ExecutionSpace>
class VoidIterator : public VoidPolicy<ExecutionSpace> {
public:
  using base_policy_type = VoidPolicy<ExecutionSpace>; ///< Base policy type
  using policy_index_type = void; ///< Index type for the policy
  using index_type = void;        ///< Index type for the iterator
  using execution_space_type =
      typename base_policy_type::execution_space; ///< Execution space type

  /**
   * @brief Returns an empty index type.
   *
   */
  KOKKOS_INLINE_FUNCTION
  constexpr index_type operator()() const { return; }
};

} // namespace execution
} // namespace specfem
