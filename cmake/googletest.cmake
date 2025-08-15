message(STATUS "Configuring GoogleTest...")

list(APPEND CMAKE_MESSAGE_INDENT "  GoogleTest: ")

# Make sure that GTEST is not installed later in the process
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Pop the indentation for GoogleTest messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
