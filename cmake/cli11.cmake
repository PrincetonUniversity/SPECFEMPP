message(STATUS "Configuring CLI11...")
list(APPEND CMAKE_MESSAGE_INDENT "  CLI11: ")
include(FetchContent)
FetchContent_Declare(
    CLI11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
set(CLI11_TESTING OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(CLI11)
list(POP_BACK CMAKE_MESSAGE_INDENT)
