
// struct to store interface point temporarily
struct ip {
  int iglob = 0;
  type_real xp = 0, zp = 0;
}

std::tuple<int, int, int, int, int>
get_edge(const specfem::HostView1d<int> n, const int itype, const int e1,
         const int e2) {

  int ixmin, ixmax, izmin, izmax;
  if (itype == 1) {
    // common single point

    // checks which corner point is given
    if (e1 == n(1)) {
      ixmin = 0;
      ixmax = 0;
      izmin = 0;
      izmax = 0;
    }
    if (e1 == n(2)) {
      ixmin = 1;
      ixmax = 1;
      izmin = 0;
      izmax = 0;
    }
    if (e1 == n(3)) {
      ixmin = 1;
      ixmax = 1;
      izmin = 1;
      izmax = 1;
    }
    if (e1 == n(4)) {
      ixmin = 0;
      ixmax = 0;
      izmin = 1;
      izmax = 1;
    }
    sens = 1;
  } else if (itype == 2) {

    // common edge

    // checks which edge and corner points are given

    if (e1 == n(1)) {
      ixmin = 0;
      izmin = 0;
      if (e2 == n(2)) {
        ixmax = 1;
        izmax = 0;
        sens = 1;
      }
      if (e2 == n(4)) {
        ixmax = 0;
        izmax = 1;
        sens = 1;
      }
    }
    if (e1 == n(2)) {
      ixmin = 1;
      izmin = 0;
      if (e2 == n(3)) {
        ixmax = 1;
        izmax = 1;
        sens = 1;
      }
      if (e2 == n(1)) {
        ixmax = 0;
        izmax = 0;
        sens = -1;
      }
    }
    if (e1 == n(3)) {
      ixmin = 1;
      izmin - 1;
      if (e2 == n(4)) {
        ixmax = 0;
        izmax = 1;
        sens = -1;
      }
      if (e2 == n(2)) {
        ixmax = 1;
        izmax = 0;
        sens = -1;
      }
    }
    if (e1 == n(4)) {
      ixmin = 0;
      izmin = 1;
      if (e2 == n(1)) {
        ixmax = 0;
        izmax = 0;
        sens = -1;
      }
      if (e2 == n(3)) {
        ixmax = 1;
        izmax = 1;
        sens = 1
      }
    }
  } else {
    throw std::runtime_error("Error get_edge unknown type");
  }

  return std::make_tuple(ixmin, ixmax, izmin, izmax, sens);
}

namespace specfem::assembly::mpi_interfaces {

template <element_type E> struct mpi_interface_type {
  specfem::HostView2d<int> ibool_interfaces;
  specfem::HostView1d<int> inum_interfaces, nibool_interfaces;

  mpi_interface_type(const int max_interface_size, const int ninterfaces,
                     const int ngllx)
      : ibool_interfaces<int>(specfem::HostView2d(
            "specfem::interfaces::compute::ibool_interfaces",
            ngllx * max_interface_size, ninterfaces)),
        inum_interfaces(specfem::HostView1d<int>(
            "specfem::interfaces::compute::inum_interfaces", ninterfaces)),
        nibool_interfaces(specfem::HostView1d<int>(
            "specfem::interfaces::compute::nibool_interfaces", ninterfaces)){};

  mpi_interface_type(const specfem::interfaces::interface &mpi_interface,
                     const specfem::HostView3d<int> ibool, const int ngllx,
                     const specfem::HostView2d<type_real> coord;
                     const int nglob, const int nproc) {
    int max_interface_size = mpi_interface.max_interface_size;
    int ninterfaces = mpi_interface.ninterfaces;
    auto &my_nelmnts_neighbours = mpi_interface.my_nelmnts_neighbours;
    auto &my_interfaces = mpi_interface.my_interfaces;
    std::vector<bool> mask_ibool(nglob, false);
    std::vector<int> n(ngnod, 0);
    ip dummy_ip{};
    *this = specfem::assembly::mpi_interfaces::mpi_interface_type<E>(
        max_interface_size, ninterfaces, ngllx);

    if (nproc == 1)
      return;
    for (int iinterface = 0; iinterface < ninterface; iinterface++) {
      int nglob_interface = 0;
      ibool_dummy.clear();
      std::fill(mask_ibool.begin(), mask_ibool.end(), false);
      for (int ispec_interface = 0; ispec_interface < my_nelmnts_neighbors;
           ispec_interface++) {
        // element ID
        ispec = my_interfaces(0, ispec_interface, iinterface);

        // type of interface
        itype = my_interfaces(1, ispec_interface, iinterface);

        // element control nods
        for (int k = 0; k < ngnod; k++) {
          n[k] = knods(k, ispec);
        }

        // common node ids
        e1 = my_interfaces(3, ispec_interface, iinterface);
        e2 = my_interfaces(4, ispec_interface, iinterface);

        auto [ixmin, ixmax, izmin, izmax, sens] = get_edge(n, itype, e1, e2);
        ixmax *= ngllx;
        izmax *= ngllz;

        for (int iz = izmin; iz < izmax; iz++) {
          for (int ix = ixmin; ix < ixmax; ix++) {
            iglob = ibool(ispec, iz, ix);
            if (ispec_type(ispec) == E) {
              if (~mask_ibool[iglob]) {
                mask_ibool[iglob] = true;
                dummy_ip.x = coord(0, iglob);
                dummy_ip.y = coord(1, iglob);
                dummy_ip.iglob = iglob;
                ibool_dummy.append(dummy_ip);
                nglob_interface++;
              }
            }
          }
        }
        this->nibool_interface(iinterface) = nglob_interface;
        // sort interfaces ibool for better caching
        std::sort(ibool_dummy.begin(), ibool_dummy.end(),
                  [&](const ip ip1, const ip ip2) {
                    if (ip1.x != ip2.x) {
                      return ip1.x < ip2.x;
                    }

                    return ip1.y < ip2.y;
                  });

        // Assign interface packing orders
        for (int num_ibool = 0; num_ibool < nglob_interface; num_ibool++) {
          this->ibool_interface(num_ibool, iinterface) = ibool_dummy[num_ibool];
        }
      }
    }

    // sets the number of interfaces for material domain
    int ninterface = 0;

    for (int iinterface = 0; iinterface < ninterface; iinterface++) {
      if (nibool_interfaces(iinterface) > 0) {
        this->inum_interfaces(ninterface) = iinterface;
        ninterface++;
      }
    }
  };
}

} // namespace specfem::assembly::mpi_interfaces
