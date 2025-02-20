#include "mesh/dim3/element_types/element_types.hpp"
#include "enumerations/dimension.hpp"
#include <iostream>

void specfem::mesh::element_types<
    specfem::dimension::type::dim3>::set_elements() {

  // Create the vectors
  std::vector<int> ispec_elastic;
  std::vector<int> ispec_acoustic;
  std::vector<int> ispec_poroelastic;

  // Initialize the number of elements
  this->nelastic = 0;
  this->nacoustic = 0;
  this->nporoelastic = 0;

  // Initialize the vectors
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (ispec_is_elastic(ispec) == true) {
      ispec_elastic.push_back(ispec);
      this->nelastic++;
    } else if (ispec_is_acoustic(ispec) == true) {
      ispec_acoustic.push_back(ispec);
      this->nacoustic++;
    } else if (ispec_is_poroelastic(ispec) == true) {
      ispec_poroelastic.push_back(ispec);
      this->nporoelastic++;
    } else {
      std::ostringstream message;
      message << "Error: Spectral element " << ispec
              << " is not assigned to any material";
      throw std::runtime_error(message.str());
    }

    // Assign the vectors to the view
    this->ispec_elastic = View1D<int>("ispec_elastic", ispec_elastic.size());
    this->ispec_acoustic = View1D<int>("ispec_acoustic", ispec_acoustic.size());
    this->ispec_poroelastic =
        View1D<int>("ispec_poroelastic", ispec_poroelastic.size());
  };
}

void specfem::mesh::element_types<specfem::dimension::type::dim3>::print()
    const {
  std::cout << "Number of elastic elements: " << nelastic << std::endl;
  std::cout << "Number of acoustic elements: " << nacoustic << std::endl;
  std::cout << "Number of poroelastic elements: " << nporoelastic << std::endl;
}

void specfem::mesh::element_types<specfem::dimension::type::dim3>::print(
    const int ispec) const {
  std::cout << "Element " << ispec << " is elastic: " << ispec_is_elastic(ispec)
            << std::endl;
  std::cout << "Element " << ispec
            << " is acoustic: " << ispec_is_acoustic(ispec) << std::endl;
  std::cout << "Element " << ispec
            << " is poroelastic: " << ispec_is_poroelastic(ispec) << std::endl;
}

template <specfem::element::medium_tag MediumTag>
void specfem::mesh::element_types<specfem::dimension::type::dim3>::print(
    const int i) const {
  if (MediumTag == specfem::element::medium_tag::elastic) {
    if (i >= nelastic) {
      std::ostringstream message;
      message << "Error: Elastic element " << i
              << " does not exist.\nNumber "
                 "of elastic elements is :"
              << nelastic << ". "
              << "(" << __FILE__ << ":" << __LINE__ << ")\n";
      throw std::runtime_error(message.str());
    }
    std::cout << "Elastic element " << ispec_elastic(i) << std::endl;
  } else if (MediumTag == specfem::element::medium_tag::acoustic) {

    if (i >= nacoustic) {
      std::ostringstream message;
      message << "Error: Acoustic element " << i
              << " does not exist.\nNumber "
                 " of acoustic elements: "
              << nacoustic << ". "
              << "(" << __FILE__ << ":" << __LINE__ << ")\n";
      throw std::runtime_error(message.str());
    }

    std::cout << "Acoustic element " << ispec_acoustic(i) << std::endl;

  } else if (MediumTag == specfem::element::medium_tag::poroelastic) {

    if (i >= nporoelastic) {
      std::ostringstream message;
      message << "Error: Poroelastic element " << i
              << " does not exist.\nNumber "
                 " of poroelastic elements: "
              << nporoelastic << ". "
              << "(" << __FILE__ << ":" << __LINE__ << ")\n";
      throw std::runtime_error(message.str());
    }

    std::cout << "Poroelastic element " << ispec_poroelastic(i) << std::endl;
  }
}

// Explicit instantiations
template void
specfem::mesh::element_types<specfem::dimension::type::dim3>::print<
    specfem::element::medium_tag::elastic>(const int) const;
template void
specfem::mesh::element_types<specfem::dimension::type::dim3>::print<
    specfem::element::medium_tag::acoustic>(const int) const;
template void
specfem::mesh::element_types<specfem::dimension::type::dim3>::print<
    specfem::element::medium_tag::poroelastic>(const int) const;
