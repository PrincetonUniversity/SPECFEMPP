#pragma once

#include "policy.hpp"

namespace specfem {
namespace execution {

class VoidIterator : public VoidPolicy {
public:
  using base_policy_type = VoidPolicy;
  using policy_index_type = void;
  using index_type = void;

  KOKKOS_INLINE_FUNCTION
  constexpr index_type operator()() const { return; }
};

} // namespace execution
} // namespace specfem
