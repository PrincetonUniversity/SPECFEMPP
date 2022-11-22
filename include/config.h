#ifndef CONFIG_H
#define CONFIG_H

using type_real = float;
const static int ndim{ 2 };
const static int ngll{ 5 };
const static int fint{ 4 }, fdouble{ 8 }, fbool{ 4 }, fchar{ 512 };
enum element_type { elastic, acoustic, poroelastic };
const static bool use_best_location{ true };
enum wave_type { p_sv, sh };

#endif
