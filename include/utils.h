#ifndef UTILS_H
#define UTILS_H

#include "config.h"

namespace utilities {
struct input_holder {
  // Struct to hold temporary variables read from database file
  type_real val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
      val11, val12;
  int n, indic;
};

struct return_holder {
  type_real rho, mu, kappa, qmu, qkappa;
};
} // namespace utilities

#endif
