# Verify that an injection plugin exports exactly specfempp_injection_plugin.
#
# Usage:
#   cmake -DPLUGIN_FILE=<path> -DFORMAT=<elf|macho> [-DNM=<nm>] \
#         -P check_plugin_exports.cmake

if(NOT PLUGIN_FILE OR NOT EXISTS "${PLUGIN_FILE}")
    message(FATAL_ERROR "check_plugin_exports: PLUGIN_FILE '${PLUGIN_FILE}' not found")
endif()

if(NOT NM OR NOT EXISTS "${NM}")
    find_program(NM NAMES nm llvm-nm)
    if(NOT NM)
        message(FATAL_ERROR "check_plugin_exports: no nm tool found")
    endif()
endif()

if(FORMAT STREQUAL "macho")
    set(_flags -g -U)
elseif(FORMAT STREQUAL "elf")
    set(_flags -D --defined-only)
else()
    message(FATAL_ERROR "check_plugin_exports: FORMAT must be elf or macho")
endif()

execute_process(
    COMMAND "${NM}" ${_flags} "${PLUGIN_FILE}"
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error
    RESULT_VARIABLE _result
)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "check_plugin_exports: ${NM} failed on ${PLUGIN_FILE}: ${_error}")
endif()

# Symbols the linker defines in every ELF shared object.
set(_ignored _init _fini _edata _end __bss_start)

string(REPLACE "\n" ";" _lines "${_output}")
set(_exported "")
foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _line)
    if(_line STREQUAL "")
        continue()
    endif()
    string(REGEX REPLACE "^.*[ \t]" "" _symbol "${_line}")
    if(FORMAT STREQUAL "macho")
        string(REGEX REPLACE "^_" "" _symbol "${_symbol}")
    endif()
    if(_symbol IN_LIST _ignored)
        continue()
    endif()
    list(APPEND _exported "${_symbol}")
endforeach()

if(NOT _exported STREQUAL "specfempp_injection_plugin")
    list(JOIN _exported ", " _listed)
    message(FATAL_ERROR
        "check_plugin_exports: ${PLUGIN_FILE} must export exactly "
        "specfempp_injection_plugin, but exports: ${_listed}")
endif()
