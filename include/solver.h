

namespace specfem {
namespace solver {
class solver {
public:
  virtual void run(){};
}

class time_marching : public solver {

public:
  time_marching(const specfem::Domain::Domain *domain,
                const specfem::TimeScheme::TimeScheme *it)
      : domain(domain), it(it){};
  void run() override;

private:
  specfem::Domain::Domain *domain;
  specfem::TimeScheme::TimeScheme *it;
}
} // namespace solver
} // namespace specfem
