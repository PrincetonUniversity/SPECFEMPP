#include <fstream>
#include <iostream>
#include <regex>

namespace IO {
class param_file {

public:
  param_file(std::string param_file);
  void open();
  void close();
  std::string param_read(std::string name);
  void read(int value, std::string name);
  void read(type_real value, std::string name);
  void read(bool value, std::string name);
  void read(std : string value, std::string name);

private:
  std::string filename;
  std::ifstream stream;
}

void read_parameters_file(std::string param_file, specfem::parameters::parameters &param);
void read_sources_file(std::string source_file,
                       std::vector<specfem::sources::source> &sources,
                       int nsources);

} // namespace IO
