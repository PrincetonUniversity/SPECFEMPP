
namespace specfem {
namespace point {

struct index {
  int ispec;
  int igllz;
  int igllx;

  index() = default;

  index(const int &ispec, const int &igllz, const int &igllx)
      : ispec(ispec), igllz(igllz), igllx(igllx) {}
}

} // namespace point
} // namespace specfem
