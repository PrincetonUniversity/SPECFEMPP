# Helpers for building SPECFEM++ injection plugins.
#
# Included by the in-tree SDK and by the installed SpecfemPluginSDKConfig.cmake,
# so plugin projects get the same function in both cases.

include_guard(GLOBAL)

# specfem_add_injection_plugin(<target>
#   SOURCES   <file>...       # required
#   LIBRARIES <lib>...        # linked PRIVATE
# )
#
# Builds <target> as a loadable MODULE that exports exactly one symbol,
# specfem_injection_plugin. Everything else -- including statically linked
# dependencies such as yaml-cpp or a private Kokkos -- is hidden, so it cannot
# clash with the copies inside the SPECFEM++ executable. A post-build step
# verifies the exported symbol set.
function(specfem_add_injection_plugin target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;LIBRARIES")
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "specfem_add_injection_plugin(${target}): unrecognized arguments: "
            "${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR
            "specfem_add_injection_plugin(${target}): SOURCES is required")
    endif()
    if(NOT TARGET specfem::plugin_sdk)
        message(FATAL_ERROR
            "specfem_add_injection_plugin(${target}): specfem::plugin_sdk not "
            "found; call find_package(SpecfemPluginSDK) first")
    endif()

    add_library(${target} MODULE ${ARG_SOURCES})
    target_link_libraries(${target} PRIVATE specfem::plugin_sdk ${ARG_LIBRARIES})
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        UNITY_BUILD OFF
    )

    # Export control is enforced by the linker, not only by visibility presets,
    # so symbols from static dependencies built with default visibility stay
    # hidden too.
    if(APPLE)
        target_link_options(${target} PRIVATE
            "LINKER:-exported_symbol,_specfem_injection_plugin")
        set(_format macho)
    elseif(UNIX)
        set(_map "${CMAKE_CURRENT_BINARY_DIR}/${target}_exports.map")
        file(WRITE "${_map}"
            "{\n  global: specfem_injection_plugin;\n  local: *;\n};\n")
        target_link_options(${target} PRIVATE "LINKER:--version-script=${_map}")
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${_map}")
        set(_format elf)
    else()
        message(FATAL_ERROR
            "specfem_add_injection_plugin(${target}): unsupported platform")
    endif()

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            "-DPLUGIN_FILE=$<TARGET_FILE:${target}>"
            "-DFORMAT=${_format}"
            "-DNM=${CMAKE_NM}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/check_plugin_exports.cmake"
        COMMENT "Checking exported symbols of injection plugin ${target}"
        VERBATIM
    )
endfunction()
