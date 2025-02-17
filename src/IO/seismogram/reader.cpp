#include "IO/seismogram/reader.hpp"
#include <fstream>
#include <tuple>
#include <vector>

void specfem::IO::seismogram_reader::read() {

  if (type != specfem::enums::seismogram::format::ascii) {
    throw std::runtime_error("Only ASCII format is supported");
  }

  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("File " + filename + " not found");
  }

  // Get the total number of lines in the file
  int nlines = 0;
  std::string line;

  while (std::getline(file, line)) {
    nlines++;
  }

  file.clear();
  file.seekg(0, std::ios::beg);

  if (nlines != source_time_function.extent(0)) {
    file.close();
    throw std::runtime_error("Error in reading seismogram file : " + filename +
                             " traces dont match with nsteps");
  }

  int nsteps = 0;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    double time, value;
    if (!(iss >> time >> value)) {
      throw std::runtime_error("Seismogram file " + filename +
                               " is not formatted correctly");
    }

    source_time_function(nsteps, 0) = time;
    source_time_function(nsteps, 1) = value;
    nsteps++;
  }

  file.close();

  return;
}
