#include "IO/fortranio/interface.hpp"
#include "specfem_setup.hpp"
#include <boost/algorithm/string/trim.hpp>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <vector>

TEST(iotests, fortran_io) {

  std::string filename = "fortran_io/input.bin";
  std::ifstream stream;

  int ival;
  bool bval;
  std::string sval;
  double dval;
  std::vector<int> vval(100, 0);

  stream.open(filename);

  specfem::IO::fortran_read_line(stream, &ival);
  EXPECT_EQ(ival, 100);
  specfem::IO::fortran_read_line(stream, &dval);
  EXPECT_FLOAT_EQ(dval, 100.0);
  specfem::IO::fortran_read_line(stream, &bval);
  EXPECT_TRUE(bval);
  specfem::IO::fortran_read_line(stream, &sval);
  EXPECT_THAT(sval.c_str(), testing::StartsWith("Test case"));
  specfem::IO::fortran_read_line(stream, &ival, &dval);
  EXPECT_EQ(ival, 100);
  EXPECT_FLOAT_EQ(dval, 100.0);
  specfem::IO::fortran_read_line(stream, &bval, &sval);
  EXPECT_TRUE(bval);
  EXPECT_THAT(sval.c_str(), testing::StartsWith("Test case"));
  specfem::IO::fortran_read_line(stream, &vval);

  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(vval[i], 10);
  }

  stream.close();
}
