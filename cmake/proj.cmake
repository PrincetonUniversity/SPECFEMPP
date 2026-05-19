message(STATUS "Configuring PROJ library...")

list(APPEND CMAKE_MESSAGE_INDENT "  PROJ: ")

find_package(PROJ REQUIRED CONFIG)

message(STATUS "Found PROJ ${PROJ_VERSION}")

list(POP_BACK CMAKE_MESSAGE_INDENT)
