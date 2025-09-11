if (SPECFEM_ENABLE_NPZ)
    # Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
    list(APPEND CMAKE_MESSAGE_INDENT "  NPZ: ")

    # First, try to find ZLIB system-wide (unless force install is enabled)
    if(NOT SPECFEM_ENABLE_ZLIB_FORCE_INSTALL)
        find_package(ZLIB QUIET)
    endif()

    # If not found or force install is enabled, fetch and build ZLIB from source
    if((NOT ZLIB_FOUND) OR (SPECFEM_ENABLE_ZLIB_FORCE_INSTALL))

        # Disable unity build for here
        set(SAVE_UNITY_BUILD ${CMAKE_UNITY_BUILD})
        set(CMAKE_UNITY_BUILD OFF)

        message(STATUS "Installing ZLIB from GitHub release...")

        include(FetchContent)

        # Set the specific version you want
        set(SPECFEM_ZLIB_VERSION "1.3" CACHE STRING "ZLIB version to use for source install")
        set(ZLIB_SOURCE_URL "https://github.com/madler/zlib/releases/download/v${SPECFEM_ZLIB_VERSION}/zlib-${SPECFEM_ZLIB_VERSION}.tar.gz")

        if (CMAKE_VERSION VERSION_LESS "3.28.0")
            FetchContent_Declare(
                zlib
                URL ${ZLIB_SOURCE_URL}
                DOWNLOAD_EXTRACT_TIMESTAMP FALSE
                # URL_HASH SHA256=# Add the SHA256 hash here for verification (optional but recommended)
            )

            FetchContent_GetProperties(zlib)
            if(NOT zlib_POPULATED)
                FetchContent_Populate(zlib)
                add_subdirectory(${zlib_SOURCE_DIR} ${zlib_BINARY_DIR} EXCLUDE_FROM_ALL)
            endif()
        else()
            FetchContent_Declare(
                zlib
                URL ${ZLIB_SOURCE_URL}
                DOWNLOAD_EXTRACT_TIMESTAMP FALSE
                EXCLUDE_FROM_ALL
                # URL_HASH SHA256=# Add the SHA256 hash here for verification (optional but recommended)
            )

            FetchContent_MakeAvailable(zlib)
        endif()

        # Set variables that find_package would normally set
        set(ZLIB_FOUND TRUE)
        set(ZLIB_LIBRARIES zlibstatic)
        set(ZLIB_INCLUDE_DIRS ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

        # Create alias target for modern CMake usage
        if(NOT TARGET zlib)
            add_library(zlib INTERFACE)
            target_link_libraries(zlib INTERFACE zlibstatic)
            target_include_directories(zlib INTERFACE ${ZLIB_INCLUDE_DIRS})
        endif()

        message(STATUS "ZLIB configured from source")

        # Restore the original unity build setting
        set(CMAKE_UNITY_BUILD ${SAVE_UNITY_BUILD})

    elseif(ZLIB_FOUND)
        message(STATUS "Using system-wide ZLIB")
        message(STATUS "ZLIB libs/ and incs/:.")
        message(STATUS "    LIB:   ${ZLIB_LIBRARIES}")
        message(STATUS "    INC:   ${ZLIB_INCLUDE_DIRS}")

        # Create alias target for modern CMake usage with system ZLIB
        if(NOT TARGET zlib)
            add_library(zlib INTERFACE)
            target_link_libraries(zlib INTERFACE ${ZLIB_LIBRARIES})
            target_include_directories(zlib INTERFACE ${ZLIB_INCLUDE_DIRS})
        endif()
    else ()
        message(STATUS "ZLIB not found.")
        set(SPECFEM_ENABLE_ZLIB OFF CACHE BOOL "Disable ZLIB support" FORCE)
    endif()

    # Pop the indentation for ZLIB messages
    list(POP_BACK CMAKE_MESSAGE_INDENT)
else()
    message(STATUS "NPZ support is disabled. Set SPECFEM_ENABLE_NPZ to ON to enable it.")
    set(SPECFEM_ENABLE_NPZ OFF CACHE BOOL "Disable NPZ support" FORCE)
endif()
