#include "../../../include/config.h"
#include "../../../include/fortran_IO.h"
#include <boost/algorithm/string/trim.hpp>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

TEST(iotests, fortran_io) {

  std::string filename = "../../../tests/unittests/fortran_io/input.bin";
  std::ifstream stream;

  int ival;
  bool bval;
  std::string sval;
  type_real dval;

  stream.open(filename);

  std::cout << stream.is_open() << std::endl;
  IO::fortran_IO::fortran_read_line(stream, &ival);
  EXPECT_EQ(ival, 100);
  IO::fortran_IO::fortran_read_line(stream, &dval);
  EXPECT_FLOAT_EQ(dval, 100.0);
  IO::fortran_IO::fortran_read_line(stream, &bval);
  EXPECT_FALSE(bval);
  IO::fortran_IO::fortran_read_line(stream, &sval);
  EXPECT_THAT(sval.c_str(), testing::StartsWith("Test case"));
  IO::fortran_IO::fortran_read_line(stream, &ival, &dval);
  EXPECT_EQ(ival, 100);
  EXPECT_FLOAT_EQ(dval, 100.0);
  IO::fortran_IO::fortran_read_line(stream, &bval, &sval);
  EXPECT_FALSE(bval);
  EXPECT_THAT(sval.c_str(), testing::StartsWith("Test case"));

  stream.close();
}
