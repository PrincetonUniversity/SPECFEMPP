#pragma once

#include "file.hpp"

#ifndef NO_NPZ
template <typename value_type>
void specfem::io::impl::NPZ::File<specfem::io::write>::write(
    const value_type *data, const std::vector<size_t> &dims,
    const std::string &path) {
  if (!stream.is_open()) {
    std::ostringstream oss;
    oss << "ERROR : File is closed for writing " << path;
    throw std::runtime_error(oss.str());
  }

  // Count total elements
  size_t nels = 1;
  for (int i = 0; i < dims.size(); ++i) {
    nels *= dims[i];
  }

  stream.seekp(global_header_offset, std::ios::beg);

  NPYString npy_header =
      specfem::io::impl::NPY::create_npy_header<value_type>(dims);
  size_t nbytes = nels * sizeof(value_type) + npy_header.size();

  // get the CRC of the data to be added
  uint32_t crc = crc32(0L, (uint8_t *)&npy_header[0], npy_header.size());
  crc = crc32(crc, (uint8_t *)data, nels * sizeof(value_type));

  // build the local header
  NPYString local_header;
  local_header += "PK";                  // first part of sig
  local_header += (uint16_t)0x0403;      // second part of sig
  local_header += (uint16_t)20;          // min version to extract
  local_header += (uint16_t)0;           // general purpose bit flag
  local_header += (uint16_t)0;           // compression method
  local_header += (uint16_t)0;           // file last mod time
  local_header += (uint16_t)0;           // file last mod date
  local_header += (uint32_t)crc;         // crc
  local_header += (uint32_t)nbytes;      // compressed size
  local_header += (uint32_t)nbytes;      // uncompressed size
  local_header += (uint16_t)path.size(); // path length
  local_header += (uint16_t)0;           // extra field length
  local_header += path;

  // build global header
  global_header += "PK";             // first part of sig
  global_header += (uint16_t)0x0201; // second part of sig
  global_header += (uint16_t)20;     // version made by
  global_header.insert(global_header.end(), local_header.begin() + 4,
                       local_header.begin() + 30);
  global_header += (uint16_t)0; // file comment length
  global_header += (uint16_t)0; // disk number where file starts
  global_header += (uint16_t)0; // internal file attributes
  global_header += (uint32_t)0; // external file attributes
  global_header +=
      (uint32_t)global_header_offset; // relative offset of local file header,
                                      // since it begins where the global
                                      // header used to begin
  global_header += path;

  // update meta data
  nrecs += 1;
  global_header_offset += nbytes + local_header.size();

  // build footer
  NPYString footer;
  footer += "PK";                           // first part of sig
  footer += (uint16_t)0x0605;               // second part of sig
  footer += (uint16_t)0;                    // number of this disk
  footer += (uint16_t)0;                    // disk where footer starts
  footer += (uint16_t)nrecs;                // number of records on this disk
  footer += (uint16_t)nrecs;                // total number of records
  footer += (uint32_t)global_header.size(); // nbytes of global headers
  footer += (uint32_t)global_header_offset; // offset of start of global
                                            // headers, since global header now
                                            // starts after newly written array
  footer += (uint16_t)0; // zip file comment length

  // write everything
  stream.write(&local_header[0], local_header.size());
  stream.write(&npy_header[0], npy_header.size());
  stream.write(reinterpret_cast<const char *>(data), sizeof(value_type) * nels);
  stream.write(&global_header[0], global_header.size());
  stream.write(&footer[0], footer.size());
}

template <typename value_type>
void specfem::io::impl::NPZ::File<specfem::io::read>::read(
    value_type *data, const std::vector<size_t> &dims,
    const std::string &path) {
  if (!stream.is_open()) {
    std::ostringstream oss;
    oss << "ERROR : File is closed for writing " << path;
    throw std::runtime_error(oss.str());
  }

  // Count total elements
  int total_elements = 1;
  int rank = dims.size();
  for (int i = 0; i < rank; ++i) {
    total_elements *= dims[i];
  }

  // whether cursor has reached the bottom of the file
  bool reached_bottom = false;

  while (1) {
    NPYString local_header(30);
    stream.read(&local_header[0], 30);
    std::streamsize header_res = stream.gcount();
    if (header_res != 30)
      throw std::runtime_error("npz_load: failed fread");

    // if we've reached the global header, stop reading
    if (local_header[2] != 0x03 || local_header[3] != 0x04) {
      if (!reached_bottom) {
        reached_bottom = true;
        stream.seekg(0, std::ios::beg);
        continue;
      } else {
        throw std::runtime_error("npz_load: variable " + path +
                                 " not found in file");
      }
    }

    // read in the variable name
    uint16_t name_len = *(uint16_t *)&local_header[26];
    std::string vname(name_len, ' ');
    // size_t vname_res = fread(&vname[0], sizeof(char), name_len, fp);
    stream.read(&vname[0], name_len);
    std::streamsize vname_res = stream.gcount();
    if (vname_res != name_len)
      throw std::runtime_error("npz_load: failed fread");

    // read in the extra field
    uint16_t extra_field_len = *(uint16_t *)&local_header[28];
    stream.seekg(extra_field_len, std::ios::cur); // skip past the extra field

    uint16_t compr_method = *reinterpret_cast<uint16_t *>(&local_header[0] + 8);

    if (compr_method != 0) {
      throw std::runtime_error(
          "npz_load: only uncompressed arrays are supported");
    }

    if (vname == path) {
      std::vector<size_t> shape =
          specfem::io::impl::NPY::parse_npy_header<value_type>(stream);
      if (rank != shape.size()) {
        std::ostringstream oss;
        oss << "ERROR : Rank mismatch between dataset and file";
        throw std::runtime_error(oss.str());
      }

      for (int i = 0; i < rank; ++i) {
        if (dims[i] != shape[i]) {
          std::ostringstream oss;
          oss << "ERROR : Dimension mismatch between dataset and file";
          throw std::runtime_error(oss.str());
        }
      }

      stream.read(reinterpret_cast<char *>(data),
                  total_elements * sizeof(value_type));
      break;
    } else {
      // skip past the data
      uint32_t size = *(uint32_t *)&local_header[22];
      stream.seekg(size, std::ios::cur);
    }
  }
}
#endif
