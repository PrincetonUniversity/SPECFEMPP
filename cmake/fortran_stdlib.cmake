message(STATUS "Configuring fortran-lang/stdlib library...")

# Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
list(APPEND CMAKE_MESSAGE_INDENT "  fortran_stdlib: ")

include(FetchContent)

set(FORTRAN_STDLIB_VERSION "0.7.0" CACHE STRING "fortran-lang/stdlib version")

set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)
set(STDLIB_BUILD_TESTING OFF CACHE BOOL "Disable stdlib tests" FORCE)
set(STDLIB_BUILD_BENCHMARKS OFF CACHE BOOL "Disable stdlib benchmarks" FORCE)
set(STDLIB_INSTALL_EXPORTNAME "" CACHE STRING "Do not install stdlib" FORCE)

message(STATUS "Fetching fortran-lang/stdlib v${FORTRAN_STDLIB_VERSION} from GitHub...")

FetchContent_Declare(
    fortran_stdlib
    GIT_REPOSITORY https://github.com/fortran-lang/stdlib
    GIT_TAG v${FORTRAN_STDLIB_VERSION}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    EXCLUDE_FROM_ALL TRUE
)

FetchContent_MakeAvailable(fortran_stdlib)

message(STATUS "fortran-lang/stdlib v${FORTRAN_STDLIB_VERSION} configured")

list(POP_BACK CMAKE_MESSAGE_INDENT)
