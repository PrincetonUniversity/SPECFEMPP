#include "../include/config.h"
#include "../include/params.h"
#include "../include/sources.h"
#include "../include/specfem_mpi.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

namespace IO {
class param_file {

public:
  param_file(std::string parameters_file);
  void open();
  void close();
  std::string param_read(std::string name, bool start_from_beginning);
  void read(int &value, std::string name, bool start_from_beginning);
  void read(type_real &value, std::string name, bool start_from_beginning);
  void read(bool &value, std::string name, bool start_from_beginning);
  void read(std::string &value, std::string name, bool start_from_beginning);

private:
  std::string filename;
  std::ifstream stream;
};

void read_parameters_file(std::string param_file, specfem::parameters &param);
void read_sources_file(std::string source_file,
                       std::vector<specfem::sources::source> &sources,
                       int nsources);
} // namespace IO
