# Set policy for new find boost
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 NEW)
endif()

message(STATUS "Configuring Boost library...")

# Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
list(APPEND CMAKE_MESSAGE_INDENT "  Boost: ")

set(SAVE_UNITY_BUILD ${CMAKE_UNITY_BUILD})
set(CMAKE_UNITY_BUILD OFF)

# Try finding boost and if not found install.
find_package(Boost 1.85.0 QUIET COMPONENTS program_options filesystem system graph)

if (NOT ${Boost_FOUND})
    # Add boost lib sources
    set(BOOST_INCLUDE_LIBRARIES program_options filesystem system algorithm tokenizer preprocessor vmd graph)
    set(BOOST_ENABLE_CMAKE ON)
    set(BOOST_ENABLE_MPI OFF CACHE INTERNAL "Boost MPI Switch")
    set(BOOST_ENABLE_PYTHON OFF CACHE INTERNAL "Boost Python Switch")
    set(BOOST_BUILD_TESTING OFF CACHE INTERNAL "Boost Test Switch")
    # The test flag is not really working... added it for completeness

    # Download and extract the boost library from GitHub
    set(BOOST_VERSION 1.87.0)
    message(STATUS "Downloading and extracting boost (${BOOST_VERSION}) library sources. This will take <1 min.")
    include(FetchContent)

    # Fetch boost from the Github release zip file to reduce download time
    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}-cmake.tar.gz # downloading a zip release speeds up the download
        USES_TERMINAL_DOWNLOAD True
        GIT_PROGRESS TRUE
        DOWNLOAD_NO_EXTRACT FALSE
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )

    # Disable Boost installation
    set(BOOST_INSTALL OFF CACHE BOOL "Don't install Boost" FORCE)
    set(BOOST_INSTALL_LIBRARIES OFF CACHE BOOL "Don't install Boost libraries" FORCE)
    set(BOOST_SKIP_INSTALL_RULES ON CACHE BOOL "Skip Boost install rules" FORCE)

    FetchContent_MakeAvailable(Boost)

    # Set BOOST_LIBS after FetchContent_MakeAvailable to ensure targets exist
    set(BOOST_LIBS Boost::program_options Boost::filesystem Boost::system
                   Boost::algorithm Boost::tokenizer Boost::preprocessor Boost::vmd Boost::graph)

else()
    # Create Boost::system target manually since it's header-only in newer versions
    if(NOT TARGET Boost::system)
        add_library(Boost::system INTERFACE IMPORTED)
        set_target_properties(Boost::system PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
    endif()

    # Check which boost LIBRARY_DIRS to use
    set(BOOST_LIBS Boost::boost Boost::program_options Boost::filesystem Boost::system Boost::graph)
    message(STATUS "Boost libs/ and incs/:")
    message(STATUS "    LIB:   ${Boost_LIBRARY_DIRS}")
    message(STATUS "    INC:   ${Boost_INCLUDE_DIRS}")
    message(STATUS "    LIBSO: ${Boost_LIBRARIES}")
endif()

# Create unified boost target for modern CMake usage
if(NOT TARGET boost)
    add_library(boost INTERFACE)
    target_link_libraries(boost INTERFACE ${BOOST_LIBS})
endif()

# Pop the indentation for Boost messages
list(POP_BACK CMAKE_MESSAGE_INDENT)

set(CMAKE_UNITY_BUILD ${SAVE_UNITY_BUILD})
unset(SAVE_UNITY_BUILD)
