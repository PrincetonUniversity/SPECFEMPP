#ifndef _SPECFEM_IO_ASCII_IMPL_NATIVE_TYPE_TPP
#define _SPECFEM_IO_ASCII_IMPL_NATIVE_TYPE_TPP

#include <fstream>
#include <iomanip>
#include <string>

template <> struct specfem::IO::impl::ASCII::native_type<bool> {
  static void write(std::ostream &os, const bool &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, bool &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoi(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<unsigned short> {
  static void write(std::ostream &os, const unsigned short &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned short &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoul(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<short> {
  static void write(std::ostream &os, const short &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, short &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoi(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<int> {
  static void write(std::ofstream &os, const int &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, int &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoi(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<long> {
  static void write(std::ostream &os, const long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, long &value) {
    std::string line;
    std::getline(is, line);
    value = std::stol(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<long long> {
  static void write(std::ostream &os, const long long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, long long &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoll(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<unsigned int> {
  static void write(std::ostream &os, const unsigned int &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned int &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoul(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<unsigned long> {
  static void write(std::ostream &os, const unsigned long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned long &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoul(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<unsigned long long> {
  static void write(std::ostream &os, const unsigned long long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned long long &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoull(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<unsigned char> {
  static void write(std::ostream &os, const unsigned char &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned char &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoul(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<float> {
  static void write(std::ostream &os, const float &value) {
    os << std::setprecision(10) << std::scientific << value << "\n";
  }
  static void read(std::ifstream &is, float &value) {
    std::string line;
    std::getline(is, line);
    value = std::stof(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<double> {
  static void write(std::ostream &os, const double &value) {
    os << std::setprecision(10) << std::scientific << value << "\n";
  }
  static void read(std::ifstream &is, double &value) {
    std::string line;
    std::getline(is, line);
    value = std::stod(line);
  }
};

template <> struct specfem::IO::impl::ASCII::native_type<std::string> {
  static void write(std::ostream &os, const std::string &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, std::string &value) {
    std::getline(is, value);
  }
};

#endif /* _SPECFEM_IO_ASCII_IMPL_NATIVE_TYPE_TPP */
