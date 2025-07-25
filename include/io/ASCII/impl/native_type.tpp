#ifndef _SPECFEM_IO_ASCII_IMPL_NATIVE_TYPE_TPP
#define _SPECFEM_IO_ASCII_IMPL_NATIVE_TYPE_TPP

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace specfem::io::impl::ASCII {
template <typename T, typename SafeCall>
T safe_parser(const std::string &value, SafeCall &&call) {
  try {
    return call(value);
  } catch (const std::out_of_range &e) {
    std::cerr << "Error parsing value: " << value << ". Exception: " << e.what()
              << std::endl;
    throw;
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument for value: " << value
              << ". Exception: " << e.what() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "Unknown error while parsing value: " << value << std::endl;
    throw;
  }
}

// Check if value is too small for floating-point types
template <typename T> T safe_real_writer(const T &value) {
  if (std::is_floating_point<T>::value &&
      std::abs(value) < std::numeric_limits<T>::min()) {
    return 0.0; // Set to zero if too small
  }
  if (std::is_floating_point<T>::value &&
      std::abs(value) > std::numeric_limits<T>::max()) {
    throw std::overflow_error(
        "Value exceeds maximum limit for floating-point type");
  }

  return value;
}
} // namespace specfem::io::impl::ASCII

template <> struct specfem::io::impl::ASCII::native_type<bool> {
  static void write(std::ostream &os, const bool &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, bool &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<bool>(
        line,
        [](const std::string &str) { return str == "1" || str == "true"; });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned short> {
  static void write(std::ostream &os, const unsigned short &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned short &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<unsigned short>(
        line, [](const std::string &str) { return std::stoul(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<short> {
  static void write(std::ostream &os, const short &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, short &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<short>(
        line, [](const std::string &str) { return std::stoi(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<int> {
  static void write(std::ofstream &os, const int &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, int &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<int>(
        line, [](const std::string &str) { return std::stoi(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<long> {
  static void write(std::ostream &os, const long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, long &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<long>(
        line, [](const std::string &str) { return std::stol(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<long long> {
  static void write(std::ostream &os, const long long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, long long &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<long long>(
        line, [](const std::string &str) { return std::stoll(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned int> {
  static void write(std::ostream &os, const unsigned int &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned int &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<unsigned int>(
        line, [](const std::string &str) { return std::stoul(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned long> {
  static void write(std::ostream &os, const unsigned long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned long &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<unsigned long>(
        line, [](const std::string &str) { return std::stoul(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned long long> {
  static void write(std::ostream &os, const unsigned long long &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned long long &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<unsigned long long>(
        line, [](const std::string &str) { return std::stoull(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned char> {
  static void write(std::ostream &os, const unsigned char &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, unsigned char &value) {
    std::string line;
    std::getline(is, line);
    value = std::stoul(line);
  }
};

template <> struct specfem::io::impl::ASCII::native_type<float> {
  static void write(std::ostream &os, const float &value) {
    os << std::setprecision(10) << std::scientific
       << specfem::io::impl::ASCII::safe_real_writer(value) << "\n";
  }
  static void read(std::ifstream &is, float &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<float>(
        line, [](const std::string &str) { return std::stof(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<double> {
  static void write(std::ostream &os, const double &value) {
    os << std::setprecision(10) << std::scientific
       << specfem::io::impl::ASCII::safe_real_writer(value) << "\n";
  }
  static void read(std::ifstream &is, double &value) {
    std::string line;
    std::getline(is, line);
    value = specfem::io::impl::ASCII::safe_parser<double>(
        line, [](const std::string &str) { return std::stod(str); });
  }
};

template <> struct specfem::io::impl::ASCII::native_type<std::string> {
  static void write(std::ostream &os, const std::string &value) {
    os << value << "\n";
  }
  static void read(std::ifstream &is, std::string &value) {
    std::getline(is, value);
  }
};

#endif /* _SPECFEM_IO_ASCII_IMPL_NATIVE_TYPE_TPP */
