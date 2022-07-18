#include "../include/gll.h"
#include <iostream>
#include <stdexcept>

gll::gll::gll() : alpha(0.0), beta(0.0), ngllx(5), ngllz(5){};

gll::gll::gll(const double alpha, const double beta)
    : alpha(alpha), beta(beta), ngllx(5), ngllz(5) {
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
}

gll::gll::gll(const double alpha, const double beta, const int ngll)
    : alpha(alpha), beta(beta), ngllx(ngll), ngllz(ngll) {
  if (ngll <= 1)
    throw std::invalid_argument("Minimum number of Gauss-Labatto points is 2");
  if (ngll <= 2)
    throw std::invalid_argument(
        "Minimum number of Gauss-Lobatto points for the SEM is 3");
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
}

gll::gll::gll(const double alpha, const double beta, const int ngllx,
              const int ngllz)
    : alpha(alpha), beta(beta), ngllx(ngllx), ngllz(ngllz) {
  if (ngllx <= 1)
    throw std::invalid_argument("Minimum number of Gauss-Labatto points is 2");
  if (ngllx <= 2)
    throw std::invalid_argument(
        "Minimum number of Gauss-Lobatto points for the SEM is 3");
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
  if (ngllx != ngllz) {
    throw std::invalid_argument(
        "ngllz != ngllx: Cannot handle unstructured meshes due to mismatch in "
        "polynomials at boundaries");
  }
}
