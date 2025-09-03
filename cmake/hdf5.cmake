

if (SPECFEM_ENABLE_HDF5)

    message(STATUS "HDF5 support is enabled. Proceeding with HDF5 configuration.")

    # Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
    list(APPEND CMAKE_MESSAGE_INDENT "  HDF5: ")

    # First, try to find HDF5 system-wide (unless force install is enabled)
    if(NOT SPECFEM_ENABLE_HDF5_FORCE_INSTALL)
        find_package(HDF5 QUIET COMPONENTS CXX)
    endif()

    # If not found or force install is enabled, fetch and build HDF5 from source
    if((NOT HDF5_FOUND) OR (SPECFEM_ENABLE_HDF5_FORCE_INSTALL))

        # Disable unity build for here
        set(SAVE_UNITY_BUILD ${CMAKE_UNITY_BUILD})
        set(CMAKE_UNITY_BUILD OFF)

        message(STATUS "Installing HDF5 from GitHub release...")

        include(FetchContent)

        if (CMAKE_VERSION VERSION_GREATER "3.30.0")
            # For CMake versions > 3.30, we need to use Set the policy)
            if (CMAKE_VERSION VERSION_LESS "3.28.0")
                cmake_policy(SET CMP0169 OLD)
            else()
                cmake_policy(SET CMP0169 NEW)
            endif()
        endif()

        # Set the specific version you want
        set(SPECFEM_HDF5_VERSION "1.14.6" CACHE STRING "HDF5 version to use for source install")
        set(HDF5_SOURCE_URL "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${SPECFEM_HDF5_VERSION}/hdf5-${SPECFEM_HDF5_VERSION}.tar.gz")

        # Configure HDF5 options before fetching
        set(HDF5_BUILD_STATIC ON CACHE BOOL "Build HDF5 static library")
        set(HDF5_BUILD_SHARED OFF CACHE BOOL "Build HDF5 shared library")
        set(HDF5_BUILD_C_LIB OFF CACHE BOOL "Build HDF5 C library")
        # set(HDF5_BUILD_FORTRAN OFF CACHE BOOL "Build HDF5 Fortran library")
        set(HDF5_BUILD_CPP_LIB ON CACHE BOOL "Build HDF5 C++ library")
        set(HDF5_BUILD_HL_LIB OFF CACHE BOOL "Build HDF5 high-level library")
        set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "Build HDF5 examples")
        set(HDF5_BUILD_TESTS OFF CACHE BOOL "Build HDF5 tests")
        set(HDF5_BUILD_TOOLS OFF CACHE BOOL "Build HDF5 tools")

        # Disable HDF5 installation
        set(HDF5_INSTALL OFF CACHE BOOL "Don't install HDF5" FORCE)
        set(SKIP_HDF5_FORTRAN_SHARED ON CACHE BOOL "Skip HDF5 Fortran shared install" FORCE)
        set(HDF5_SKIP_INSTALL_RULES ON CACHE BOOL "Skip HDF5 install rules" FORCE)

        if (CMAKE_VERSION VERSION_LESS "3.28.0")
            # For CMake versions < 3.28, EXCLUDE_FROM_ALL is not supported in FetchContent_Declare
            FetchContent_Declare(
                hdf5
                URL ${HDF5_SOURCE_URL}
                DOWNLOAD_EXTRACT_TIMESTAMP FALSE
                # URL_HASH SHA256=# Add the SHA256 hash here for verification (optional but recommended)
            )

            FetchContent_GetProperties(hdf5)
            if(NOT hdf5_POPULATED)
                FetchContent_Populate(hdf5)
                add_subdirectory(${hdf5_SOURCE_DIR} ${hdf5_BINARY_DIR} EXCLUDE_FROM_ALL)
            endif()
        else()
            # For CMake versions >= 3.28, EXCLUDE_FROM_ALL is supported in FetchContent_Declare
            FetchContent_Declare(
                hdf5
                URL ${HDF5_SOURCE_URL}
                DOWNLOAD_EXTRACT_TIMESTAMP FALSE
                EXCLUDE_FROM_ALL
                # URL_HASH SHA256=# Add the SHA256 hash here for verification (optional but recommended)
            )

            FetchContent_MakeAvailable(hdf5)
        endif()

        # Set variables that find_package would normally set
        set(HDF5_FOUND TRUE)
        set(HDF5_LIBRARIES hdf5-static hdf5_cpp-static)
        set(HDF5_INCLUDE_DIRS ${hdf5_SOURCE_DIR}/src ${hdf5_SOURCE_DIR}/c++/src ${hdf5_BINARY_DIR})

        # Create alias target for modern CMake usage
        if(NOT TARGET hdf5)
            add_library(hdf5 INTERFACE)
            target_link_libraries(hdf5 INTERFACE hdf5-static hdf5_cpp-static)
            target_include_directories(hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
        endif()

        message(STATUS "HDF5 configured from source")

        # Restore the original unity build setting
        set(CMAKE_UNITY_BUILD ${SAVE_UNITY_BUILD})

    elseif(HDF5_FOUND)
        message(STATUS "Using system-wide HDF5")
        message(STATUS "HDF5 libs/ and incs/:.")
        message(STATUS "    LIB:   ${HDF5_LIBRARIES}")
        message(STATUS "    INC:   ${HDF5_INCLUDE_DIRS}")
        message(STATUS "    LIBSO: ${HDF5_CXX_LIBRARIES}")

        # Create alias target for modern CMake usage with system HDF5
        if(NOT TARGET hdf5)
            add_library(hdf5 INTERFACE)
            target_link_libraries(hdf5 INTERFACE ${HDF5_LIBRARIES})
            target_include_directories(hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
        endif()
    else ()
        message(STATUS "HDF5 not found.")
        set(SPECFEM_ENABLE_HDF5 OFF CACHE BOOL "Disable HDF5 support" FORCE)
    endif()

    # Pop the indentation for HDF5 messages
    list(POP_BACK CMAKE_MESSAGE_INDENT)
else()
    message(STATUS "HDF5 support is disabled. Set SPECFEM_ENABLE_HDF5 to ON to enable it.")
    set(SPECFEM_ENABLE_HDF5 OFF CACHE BOOL "Disable HDF5 support" FORCE)
endif()
