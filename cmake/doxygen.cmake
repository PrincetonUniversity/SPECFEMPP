message(STATUS "Configuring Doxygen documentation...")

list(APPEND CMAKE_MESSAGE_INDENT "  Doxygen: ")

# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message(STATUS "Doxygen build started")
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        ${DOXYGEN_OUT}
    )

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    # add_custom_target(docs
    #     COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    #     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #     COMMENT "Generating API documentation with Doxygen"
    #     VERBATIM )
else (DOXYGEN_FOUND)
  message(STATUS "Doxygen need to be installed to generate the doxygen documentation.")
endif (DOXYGEN_FOUND)

# Pop the indentation for Doxygen messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
