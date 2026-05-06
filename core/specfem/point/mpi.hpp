#pragma once

namespace specfem::point {

class mpi {
public:
  unsigned int iglob; /**< Global index of the GLL point being communicated */
  unsigned int entity_iglob; /***< Global index on the entity to which the GLL
                                point belongs */

  mpi() = default;

  mpi(const unsigned int iglob, const unsigned int entity_iglob)
      : iglob(iglob), entity_iglob(entity_iglob) {}
};

} // namespace specfem::point
