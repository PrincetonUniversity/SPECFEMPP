#ifndef MATERIAL_H
#define MATERIAL_H

#include "../include/config.h"
#include "../include/utils.h"
#include <ostream>

namespace specfem {

class material {
public:
  material();
  virtual void assign(utilities::value_holder &holder){};
};

class elastic_material : public material {
public:
  elastic_material();
  void assign(utilities::value_holder &holder);
  friend std::ostream &operator<<(std::ostream &out, const elastic_material &h);

private:
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
};

class acoustic_material : public material {
public:
  acoustic_material();
  void assign(utilities::value_holder &holder);
  friend std::ostream &operator<<(std::ostream &out,
                                  const acoustic_material &h);

private:
  type_real density = 0.0, cs = 0.0, cp = 0.0, Qkappa = 9999.0, Qmu = 9999.0,
            compaction_grad = 0.0, lambdaplus2mu = 0.0, mu = 0.0, lambda = 0.0,
            kappa = 0.0, young = 0.0, poisson = 0.0;
};

std::ostream &operator<<(std::ostream &out, const specfem::elastic_material &h);

std::ostream &operator<<(std::ostream &out,
                         const specfem::acoustic_material &h);

} // namespace specfem

#endif
