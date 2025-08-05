# Install HDF5 as a dependency if not found
find_package(HDF5 COMPONENTS CXX)

if (NOT ${HDF5_FOUND})
    message(STATUS "HDF5 not found. Building without HDF5.")
    set(HDF5_CXX_BUILD OFF)
else()
    message(STATUS "HDF5 libs/ and incs/:.")
    message(STATUS "    LIB:   ${HDF5_LIBRARIES}")
    message(STATUS "    INC:   ${HDF5_INCLUDE_DIRS}")
    message(STATUS "    LIBSO: ${HDF5_CXX_LIBRARIES}")
endif()
