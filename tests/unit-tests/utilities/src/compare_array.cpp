
#include "../include/interface.hpp"
#include "fortranio/interface.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

bool specfem::testing::equate(int computed_value, int ref_value) {
  return (computed_value == ref_value);
};

bool specfem::testing::equate(type_real computed_value, type_real ref_value,
                              type_real tol) {
  return (fabs(computed_value - ref_value) < tol);
};

// void equate_norm(type_real error_norm, type_real computed_norm,
//                  type_real tolerance) {
//   type_real percent_norm = (error_norm / computed_norm);

//   // check nan value
//   if (percent_norm != percent_norm) {
//     std::ostringstream ss;
//     ss << "Normalized error is NaN value";

//     throw std::runtime_error(ss.str());
//   }

//   if (percent_norm > tolerance) {
//     std::ostringstream ss;
//     ss << "Normalized error is = " << percent_norm
//        << " which is greater than specified tolerance = " << tolerance
//        << " computed norm = " << computed_norm
//        << " error norm = " << error_norm;

//     throw std::runtime_error(ss.str());
//   }

//   return;
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView1d<int, Kokkos::LayoutRight> computed_array,
//     std::string ref_file, int n1) {
//   assert(computed_array.extent(0) == n1);

//   int ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//     try {
//       equate(computed_array(i1), ref_value);
//     } catch (std::runtime_error &e) {
//       stream.close();
//       std::ostringstream ss;
//       ss << e.what() << ", at i1 = " << i1;
//       throw std::runtime_error(ss.str());
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView2d<int, Kokkos::LayoutRight> computed_array,
//     std::string ref_file, int n1, int n2) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);

//   int ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//       try {
//         equate(computed_array(i1, i2), ref_value);
//       } catch (std::runtime_error &e) {
//         stream.close();
//         std::ostringstream ss;
//         ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2;
//         throw std::runtime_error(ss.str());
//       }
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView3d<int, Kokkos::LayoutRight> computed_array,
//     std::string ref_file, int n1, int n2, int n3) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);
//   assert(computed_array.extent(2) == n3);

//   int ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//         try {
//           equate(computed_array(i1, i2, i3), ref_value);
//         } catch (std::runtime_error &e) {
//           stream.close();
//           std::ostringstream ss;
//           ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2
//              << ", n3 = " << i3;
//           throw std::runtime_error(ss.str());
//         }
//       }
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1) {
//   assert(computed_array.extent(0) == n1);

//   type_real max_val = std::numeric_limits<type_real>::min();
//   type_real min_val = std::numeric_limits<type_real>::max();

//   for (int i1 = 0; i1 < n1; i1++) {
//     if (max_val < computed_array(i1))
//       max_val = computed_array(i1);
//     if (min_val > computed_array(i1))
//       min_val = computed_array(i1);
//   }

//   type_real tol = 1e-2 * fabs(max_val + min_val) / 2;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//     try {
//       equate(computed_array(i1), ref_value, tol);
//     } catch (std::runtime_error &e) {
//       stream.close();
//       std::ostringstream ss;
//       ss << e.what() << ", at i1 = " << i1;
//       throw std::runtime_error(ss.str());
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);

//   type_real max_val = std::numeric_limits<type_real>::min();
//   type_real min_val = std::numeric_limits<type_real>::max();

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       if (max_val < computed_array(i1, i2))
//         max_val = fabs(computed_array(i1, i2));
//       if (min_val > computed_array(i1, i2))
//         min_val = fabs(computed_array(i1, i2));
//     }
//   }

//   type_real tol = 10 * fabs(max_val + min_val) / 2;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//       try {
//         equate(computed_array(i1, i2), ref_value, tol);
//       } catch (std::runtime_error &e) {
//         stream.close();
//         std::ostringstream ss;
//         ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2;
//         throw std::runtime_error(ss.str());
//       }
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2, int n3) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);
//   assert(computed_array.extent(2) == n3);

//   type_real max_val = std::numeric_limits<type_real>::min();
//   type_real min_val = std::numeric_limits<type_real>::max();

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         if (max_val < computed_array(i1, i2, i3))
//           max_val = computed_array(i1, i2, i3);
//         if (min_val > computed_array(i1, i2, i3))
//           min_val = computed_array(i1, i2, i3);
//       }
//     }
//   }

//   type_real tol = 1e-2 * fabs(max_val + min_val) / 2;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//         try {
//           equate(computed_array(i1, i2, i3), ref_value, tol);
//         } catch (std::runtime_error &e) {
//           stream.close();
//           std::ostringstream ss;
//           ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2
//              << ", n3 = " << i3;
//           throw std::runtime_error(ss.str());
//         }
//       }
//     }
//   }
// }

// void specfem::testing::compare_norm(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, type_real tolerance) {
//   assert(computed_array.extent(0) == n1);

//   type_real error_norm = 0.0;
//   type_real computed_norm = 0.0;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     specfem::fortran_IO::fortran_read_line(stream, &ref_value);

//     error_norm += std::sqrt((computed_array(i1) - ref_value) *
//                             (computed_array(i1) - ref_value));
//     computed_norm += std::sqrt((computed_array(i1) * computed_array(i1)));
//   }

//   std::cout << error_norm << std::endl;

//   std::cout << computed_norm << std::endl;

//   stream.close();

//   equate_norm(error_norm, computed_norm, tolerance);
// }

// void specfem::testing::compare_norm(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2, type_real
//     tolerance) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);

//   type_real error_norm = 0.0;
//   type_real computed_norm = 0.0;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       specfem::fortran_IO::fortran_read_line(stream, &ref_value);

//       error_norm += std::sqrt((computed_array(i1, i2) - ref_value) *
//                               (computed_array(i1, i2) - ref_value));
//       computed_norm +=
//           std::sqrt((computed_array(i1, i2) * computed_array(i1, i2)));
//     }
//   }

//   stream.close();

//   equate_norm(error_norm, computed_norm, tolerance);
// }

// void specfem::testing::compare_norm(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2, int n3, type_real
//     tolerance) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);
//   assert(computed_array.extent(2) == n3);

//   type_real error_norm = 0.0;
//   type_real computed_norm = 0.0;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         specfem::fortran_IO::fortran_read_line(stream, &ref_value);

//         error_norm += std::sqrt((computed_array(i1, i2, i3) - ref_value) *
//                                 (computed_array(i1, i2, i3) - ref_value));
//         computed_norm += std::sqrt(
//             (computed_array(i1, i2, i3) * computed_array(i1, i2, i3)));
//       }
//     }
//   }

//   std::cout << error_norm << std::endl;

//   std::cout << computed_norm << std::endl;

//   stream.close();

//   equate_norm(error_norm, computed_norm, tolerance);
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView1d<int, Kokkos::LayoutLeft> computed_array,
//     std::string ref_file, int n1) {
//   assert(computed_array.extent(0) == n1);

//   int ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//     try {
//       equate(computed_array(i1), ref_value);
//     } catch (std::runtime_error &e) {
//       stream.close();
//       std::ostringstream ss;
//       ss << e.what() << ", at i1 = " << i1;
//       throw std::runtime_error(ss.str());
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView2d<int, Kokkos::LayoutLeft> computed_array,
//     std::string ref_file, int n1, int n2) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);

//   int ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//       try {
//         equate(computed_array(i1, i2), ref_value);
//       } catch (std::runtime_error &e) {
//         stream.close();
//         std::ostringstream ss;
//         ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2;
//         throw std::runtime_error(ss.str());
//       }
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView3d<int, Kokkos::LayoutLeft> computed_array,
//     std::string ref_file, int n1, int n2, int n3) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);
//   assert(computed_array.extent(2) == n3);

//   int ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//         try {
//           equate(computed_array(i1, i2, i3), ref_value);
//         } catch (std::runtime_error &e) {
//           stream.close();
//           std::ostringstream ss;
//           ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2
//              << ", n3 = " << i3;
//           throw std::runtime_error(ss.str());
//         }
//       }
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1) {
//   assert(computed_array.extent(0) == n1);

//   type_real max_val = std::numeric_limits<type_real>::min();
//   type_real min_val = std::numeric_limits<type_real>::max();

//   for (int i1 = 0; i1 < n1; i1++) {
//     if (max_val < computed_array(i1))
//       max_val = computed_array(i1);
//     if (min_val > computed_array(i1))
//       min_val = computed_array(i1);
//   }

//   type_real tol = 1e-2 * fabs(max_val + min_val) / 2;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//     try {
//       equate(computed_array(i1), ref_value, tol);
//     } catch (std::runtime_error &e) {
//       stream.close();
//       std::ostringstream ss;
//       ss << e.what() << ", at i1 = " << i1;
//       throw std::runtime_error(ss.str());
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);

//   type_real max_val = std::numeric_limits<type_real>::min();
//   type_real min_val = std::numeric_limits<type_real>::max();

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       if (max_val < computed_array(i1, i2))
//         max_val = fabs(computed_array(i1, i2));
//       if (min_val > computed_array(i1, i2))
//         min_val = fabs(computed_array(i1, i2));
//     }
//   }

//   type_real tol = 1e-2 * fabs(max_val + min_val) / 2;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//       try {
//         equate(computed_array(i1, i2), ref_value, tol);
//       } catch (std::runtime_error &e) {
//         stream.close();
//         std::ostringstream ss;
//         ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2;
//         throw std::runtime_error(ss.str());
//       }
//     }
//   }
// }

// void specfem::testing::test_array(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2, int n3) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);
//   assert(computed_array.extent(2) == n3);

//   type_real max_val = std::numeric_limits<type_real>::min();
//   type_real min_val = std::numeric_limits<type_real>::max();

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         if (max_val < computed_array(i1, i2, i3))
//           max_val = computed_array(i1, i2, i3);
//         if (min_val > computed_array(i1, i2, i3))
//           min_val = computed_array(i1, i2, i3);
//       }
//     }
//   }

//   type_real tol = 1e-2 * fabs(max_val + min_val) / 2;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//         try {
//           equate(computed_array(i1, i2, i3), ref_value, tol);
//         } catch (std::runtime_error &e) {
//           stream.close();
//           std::ostringstream ss;
//           ss << e.what() << ", at n1 = " << i1 << ", n2 = " << i2
//              << ", n3 = " << i3;
//           throw std::runtime_error(ss.str());
//         }
//       }
//     }
//   }
// }

// void specfem::testing::compare_norm(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, type_real tolerance) {
//   assert(computed_array.extent(0) == n1);

//   type_real error_norm = 0.0;
//   type_real computed_norm = 0.0;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     specfem::fortran_IO::fortran_read_line(stream, &ref_value);

//     error_norm += std::sqrt((computed_array(i1) - ref_value) *
//                             (computed_array(i1) - ref_value));
//     computed_norm += std::sqrt((computed_array(i1) * computed_array(i1)));
//   }

//   stream.close();

//   equate_norm(error_norm, computed_norm, tolerance);
// }

// void specfem::testing::compare_norm(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2, type_real
//     tolerance) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);

//   type_real error_norm = 0.0;
//   type_real computed_norm = 0.0;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       specfem::fortran_IO::fortran_read_line(stream, &ref_value);
//       type_real computed_value = computed_array(i1, i2);

//       error_norm += std::sqrt((computed_value - ref_value) *
//                               (computed_value - ref_value));
//       computed_norm += std::sqrt((computed_value * computed_value));
//     }
//   }

//   stream.close();

//   equate_norm(error_norm, computed_norm, tolerance);
// }

// void specfem::testing::compare_norm(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2, int n3, type_real
//     tolerance) {
//   assert(computed_array.extent(0) == n1);
//   assert(computed_array.extent(1) == n2);
//   assert(computed_array.extent(2) == n3);

//   type_real error_norm = 0.0;
//   type_real computed_norm = 0.0;

//   type_real ref_value;
//   std::ifstream stream;
//   stream.open(ref_file);

//   for (int i1 = 0; i1 < n1; i1++) {
//     for (int i2 = 0; i2 < n2; i2++) {
//       for (int i3 = 0; i3 < n3; i3++) {
//         specfem::fortran_IO::fortran_read_line(stream, &ref_value);

//         error_norm += std::sqrt((computed_array(i1, i2, i3) - ref_value) *
//                                 (computed_array(i1, i2, i3) - ref_value));
//         computed_norm += std::sqrt(
//             (computed_array(i1, i2, i3) * computed_array(i1, i2, i3)));
//       }
//     }
//   }

//   std::cout << error_norm << std::endl;

//   std::cout << computed_norm << std::endl;

//   stream.close();

//   equate_norm(error_norm, computed_norm, tolerance);
// }
