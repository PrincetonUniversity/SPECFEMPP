#ifndef ENUMS_H
#define ENUMS_H

namespace specfem {

namespace elements {
enum type { elastic, acoustic, poroelastic };
} // namespace elements

namespace wave {
enum type { p_sv, sh };
} // namespace wave

namespace seismogram {
enum type { displacement, velocity, acceleration };
} // namespace seismogram

} // namespace specfem

#endif
