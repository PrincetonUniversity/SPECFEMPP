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

  /**
   * @brief Routines to read parameters(Par) file
   *
   * @note Modern method of doing this is through yaml.
   *
   */

public:
  /**
   * @brief Construct a new param file object
   *
   * @param parameters_file std::string parameters filename
   */
  param_file(std::string parameters_file);
  /**
   * @brief Open the file
   *
   */
  void open();
  /**
   * @brief Close the file
   *
   */
  void close();
  std::string param_read(std::string name, bool start_from_beginning);

  /**
   * @brief Read interger value of a variable
   *
   * @param value Address of int variable to store the value
   * @param name name of the variable as described in parameters file
   * @param start_from_beginning set to true if you want to search from
   * beginning. If there are multiple instances of same variable as in the case
   * of sources file then set to false to sequentially search the file. Default:
   * true
   */
  void read(int &value, std::string name, bool start_from_beginning);
  /**
   * @brief Read real value of a variable
   *
   * @param value Address of real variable to store the value
   * @param name name of the variable as described in parameters file
   * @param start_from_beginning set to true if you want to search from
   * beginning. If there are multiple instances of same variable as in the case
   * of sources file then set to false to sequentially search the file. Default:
   * true
   */
  void read(type_real &value, std::string name, bool start_from_beginning);
  /**
   * @brief Read bool value of a variable
   *
   * @param value Address of bool variable to store the value
   * @param name name of the variable as described in parameters file
   * @param start_from_beginning set to true if you want to search from
   * beginning. If there are multiple instances of same variable as in the case
   * of sources file then set to false to sequentially search the file. Default:
   * true
   */
  void read(bool &value, std::string name, bool start_from_beginning);
  /**
   * @brief Read string value of a variable
   *
   * @param value Address of string variable to store the value
   * @param name name of the variable as described in parameters file
   * @param start_from_beginning set to true if you want to search from
   * beginning. If there are multiple instances of same variable as in the case
   * of sources file then set to false to sequentially search the file. Default:
   * true
   */
  void read(std::string &value, std::string name, bool start_from_beginning);

private:
  std::string filename;
  std::ifstream stream;
};

/**
 * @brief Read parameters file. The variables to be read are hard coded (This
 * wouldn't be the case if we were to use a yaml)
 *
 * @param param_file Parameters filename
 * @param param parameters struct to store read values
 */
void read_parameters_file(std::string param_file, specfem::parameters &param);
/**
 * @brief Read sources file. Again we should be using yaml here
 *
 * @param source_file Name of the sources file
 * @param sources std::vector of sources struct
 * @param nsources Total number of sources present
 */
void read_sources_file(std::string source_file,
                       std::vector<specfem::sources::source> &sources,
                       int nsources);
} // namespace IO
