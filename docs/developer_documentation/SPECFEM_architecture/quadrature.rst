.. _quadrature_dev_guide:

Quadrature developer guide:
===========================

Quadrature class defines quadrature rules for numerical integration. Defining new quadrature rules requires building a parent class that inherits from the Quadrature class and implements the following methods:

1. Constructor:
   - A constructor must:
        b. Compute the quadrature points ``specfem::kokkos::DeviceView1d<type_real> xi`` and weights ``specfem::kokkos::DeviceView1d<type_real> w`` and store them within the class as Kokkos views. Where ``xi(p)`` defines the p`th quadrature point and ``w(p)`` defines the weight of the `p`th quadrature point.
        c. Compute derivatives of polynomial ``specfem::kokkos::DeviceView2d<type_real> hprime`` at quadrature points and store them within Kokkos View. Where ``hprime(p, q)`` defines the derivative of the ``q``th polynomial at the ``p``th quadrature point.
        d. Define host mirrors of the above Kokkos views. i.e ``specfem::kokkos::HostMirror1d<type_real> h_xi``, ``specfem::kokkos::HostMirror1d<type_real> h_w`` and ``specfem::kokkos::HostMirror2d<type_real> h_hprime``.
   - A typical implementation computes these values on the host and them deep copies them to the device using ``Kokkos::deep_copy``.

2. ``int get_N() const override`` : Returns the number of quadrature points.
3. ``specfem::kokkos::DeviceView1d<type_real> get_xi() const override`` and ``specfem::kokkos::HostMirror1d<type_real> get_hxi() const override`` : Returns the quadrature points on the device and host respectively.
4. ``specfem::kokkos::DeviceView1d<type_real> get_w() const override`` and ``specfem::kokkos::HostMirror1d<type_real> get_hw() const override`` : Returns the quadrature weights on the device and host respectively.
5. ``specfem::kokkos::DeviceView2d<type_real> get_hprime() const override`` and ``specfem::kokkos::HostMirror2d<type_real> get_hhprime() const override`` : Returns the derivatives of the polynomials at the quadrature points on the device and host respectively.
6. ``void print(const std::ostream& os) const override`` : Prints the quadrature information to the output stream `os`.
