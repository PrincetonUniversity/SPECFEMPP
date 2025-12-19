#include "specfem_mpi/interface.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <vector>

specfem::MPI::MPI::MPI(int *argc, char ***argv) {
#ifdef MPI_PARALLEL
  MPI_Initialized(&this->extern_init);
  if (!this->extern_init) {
    MPI_Init(argc, argv);
  }
  this->comm = MPI_COMM_WORLD;
  MPI_Comm_size(this->comm, &this->world_size);
  MPI_Comm_rank(this->comm, &this->my_rank);
#else
  this->world_size = 1;
  this->my_rank = 0;
  this->extern_init = 0;
#endif
}

void specfem::MPI::MPI::sync_all() const {
#ifdef MPI_PARALLEL
  MPI_Barrier(this->comm);
#endif
}

specfem::MPI::MPI::~MPI() {
#ifdef MPI_PARALLEL
  if (!this->extern_init) {
    MPI_Finalize();
  }
#endif
}

int specfem::MPI::MPI::get_size() const { return this->world_size; }

int specfem::MPI::MPI::get_rank() const { return this->my_rank; }

void specfem::MPI::MPI::exit() {
#ifdef MPI_PARALLEL
  int ierr = MPI_Abort(this->comm, 30);
#else
  std::exit(30);
#endif
}

int specfem::MPI::MPI::reduce(int lvalue,
                              specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  int svalue;

  MPI_Reduce(&lvalue, &svalue, 1, MPI_INT, reducer, this->get_main(),
             this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

int specfem::MPI::MPI::all_reduce(int lvalue,
                                  specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  int svalue;

  MPI_Allreduce(&lvalue, &svalue, 1, MPI_INT, reducer, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

float specfem::MPI::MPI::reduce(float lvalue,
                                specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  float svalue;

  MPI_Reduce(&lvalue, &svalue, 1, MPI_FLOAT, reducer, this->get_main(),
             this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

float specfem::MPI::MPI::all_reduce(float lvalue,
                                    specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  float svalue;

  MPI_Allreduce(&lvalue, &svalue, 1, MPI_FLOAT, reducer, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

double specfem::MPI::MPI::reduce(double lvalue,
                                 specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  double svalue;

  MPI_Reduce(&lvalue, &svalue, 1, MPI_DOUBLE, reducer, this->get_main(),
             this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

double specfem::MPI::MPI::all_reduce(double lvalue,
                                     specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  double svalue;

  MPI_Allreduce(&lvalue, &svalue, 1, MPI_DOUBLE, reducer, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

std::vector<int> specfem::MPI::MPI::gather(int lelement) const {

  std::vector<int> gelement(this->world_size, 0);

#ifdef MPI_PARALLEL
  auto data_type = MPI_INT;
  MPI_Gather(&lelement, 1, data_type, &gelement[0], this->world_size, data_type,
             this->get_main(), this->comm);
  return gelement;
#else
  gelement[0] = lelement;
  return gelement;
#endif
}

std::vector<float> specfem::MPI::MPI::gather(float lelement) const {

  std::vector<float> gelement(this->world_size, 0);

#ifdef MPI_PARALLEL
  auto data_type = MPI_FLOAT;
  MPI_Gather(&lelement, 1, data_type, &gelement[0], this->world_size, data_type,
             this->get_main(), this->comm);
  return gelement;
#else
  gelement[0] = lelement;
  return gelement;
#endif
}

std::vector<double> specfem::MPI::MPI::gather(double lelement) const {

  std::vector<double> gelement(this->world_size, 0);

#ifdef MPI_PARALLEL
  auto data_type = MPI_DOUBLE;
  MPI_Gather(&lelement, 1, data_type, &gelement[0], this->world_size, data_type,
             this->get_main(), this->comm);
  return gelement;
#else
  gelement[0] = lelement;
  return gelement;
#endif
}

int specfem::MPI::MPI::scatter(std::vector<int> gelement) const {

  int lelement;
  assert(gelement.size() == this->world_size);

#ifdef MPI_PARALLEL
  auto data_type = MPI_INT;
  MPI_Scatter(&gelement[0], this->world_size, data_type, &lelement, 1,
              data_type, this->get_main(), this->comm);
  return lelement;
#else
  lelement = gelement[0];
  return lelement;
#endif
}

float specfem::MPI::MPI::scatter(std::vector<float> gelement) const {

  float lelement;
  assert(gelement.size() == this->world_size);

#ifdef MPI_PARALLEL
  auto data_type = MPI_FLOAT;
  MPI_Scatter(&gelement[0], this->world_size, data_type, &lelement, 1,
              data_type, this->get_main(), this->comm);
  return lelement;
#else
  lelement = gelement[0];
  return lelement;
#endif
}

double specfem::MPI::MPI::scatter(std::vector<double> gelement) const {

  double lelement;
  assert(gelement.size() == this->world_size);

#ifdef MPI_PARALLEL
  auto data_type = MPI_DOUBLE;
  MPI_Scatter(&gelement[0], this->world_size, data_type, &lelement, 1,
              data_type, this->get_main(), this->comm);
  return lelement;
#else
  lelement = gelement[0];
  return lelement;
#endif
}

void specfem::MPI::MPI::bcast(int &val) const {
#ifdef MPI_PARALLEL
  MPI_Bcast(&val, 1, MPI_INT, this->get_main(), this->comm);
#endif
}

void specfem::MPI::MPI::bcast(float &val) const {
#ifdef MPI_PARALLEL
  MPI_Bcast(&val, 1, MPI_FLOAT, this->get_main(), this->comm);
#endif
}

void specfem::MPI::MPI::bcast(double &val) const {
#ifdef MPI_PARALLEL
  MPI_Bcast(&val, 1, MPI_DOUBLE, this->get_main(), this->comm);
#endif
}

void specfem::MPI::MPI::bcast(int &val, int root) const {
#ifdef MPI_PARALLEL
  MPI_Bcast(&val, 1, MPI_INT, root, this->comm);
#endif
}

void specfem::MPI::MPI::bcast(float &val, int root) const {
#ifdef MPI_PARALLEL
  MPI_Bcast(&val, 1, MPI_FLOAT, root, this->comm);
#endif
}

void specfem::MPI::MPI::bcast(double &val, int root) const {
#ifdef MPI_PARALLEL
  MPI_Bcast(&val, 1, MPI_DOUBLE, root, this->comm);
#endif
}

/**
 * @brief Print a title string with surrounding decoration
 *
 * @param s Title string to print
 */
void specfem::MPI::MPI::print_title(const std::string &s) const {
  const int total_width = 60; // Fixed width for the separator lines

  // Calculate padding for centering
  int text_length = s.length();
  int padding = (total_width - text_length) / 2;

  // Build the centered title line
  std::string title_line(padding, '-');
  title_line += s;
  title_line += std::string(total_width - padding - text_length, '-');

  this->cout(title_line);
}

/**
 * @brief Print a formatted header with SPECFEM++ branding
 *
 * @param s Header string to print (default: "SPECFEM++")
 */
void specfem::MPI::MPI::print_header(const std::string &s) const {
  const int total_width = 60;
  const std::string deco_line(total_width, '=');
  // Add spaces around the string
  const std::string title = " " + s + " ";

  // Calculate padding for centering
  int text_length = title.length();
  int padding = (total_width - text_length) / 2;

  // Build the centered title line
  std::string title_line(padding, '-');
  title_line += title;
  title_line += std::string(total_width - padding - text_length, '-');

  this->cout(deco_line);
  this->cout(title_line);
  this->cout(deco_line);
}
