#include "reader/seismogram.hpp"
#include <fstream>
#include <tuple>
#include <vector>

void specfem::reader::seismogram::read() {

  if (type != specfem::enums::seismogram::format::ascii) {
    throw std::runtime_error("Only ASCII format is supported");
  }

  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("File " + filename + " not found");
  }

  std::string line;
  int nsteps = 0;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    type_real time, value;
    if (!(iss >> time >> value)) {
      throw std::runtime_error("Seismogram file " + filename +
                               " is not formatted correctly");
    }

    source_time_function(nsteps, 0) = time;
    source_time_function(nsteps, 1) = value;
  }

  file.close();

  if (nsteps != source_time_function.extent(0)) {
    throw std::runtime_error("Error in reading seismogram file : " + filename +
                             " traces dont match with nsteps");
  }

  return;
}
