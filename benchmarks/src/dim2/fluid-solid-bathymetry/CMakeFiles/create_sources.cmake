cmake_minimum_required(VERSION 3.12)
project(SourceGeneration)

# config.cmake
set(NUMBER_OF_SOURCES_X 197)
set(NUMBER_OF_SOURCES_Z 37)

# Calculate the total number of sources
math(EXPR TOTAL_SOURCES "${NUMBER_OF_SOURCES_X} + ${NUMBER_OF_SOURCES_Z}")

# Generate source entries
set(SOURCE_ENTRIES "")

# Generate X sources
math(EXPR UPPER_X "${NUMBER_OF_SOURCES_X} - 1")

foreach(i RANGE 0 ${UPPER_X})
  math(EXPR X_POS "200 + ${i} * 100")
  math(EXPR Z_POS "720")
  math(EXPR TSHIFT_X "0 + ${i} * 5309")

  string(APPEND SOURCE_ENTRIES "    - moment-tensor:\n")
  string(APPEND SOURCE_ENTRIES "        x: ${X_POS}\n")
  string(APPEND SOURCE_ENTRIES "        z: ${Z_POS}\n")
  string(APPEND SOURCE_ENTRIES "        Mxx: 1.0\n")
  string(APPEND SOURCE_ENTRIES "        Mzz: 1.0\n")
  string(APPEND SOURCE_ENTRIES "        Mxz: 0.0\n")
  string(APPEND SOURCE_ENTRIES "        angle: 0.0\n")
  string(APPEND SOURCE_ENTRIES "        Ricker:\n")
  string(APPEND SOURCE_ENTRIES "            factor: 9.836e-10\n")
  string(APPEND SOURCE_ENTRIES "            tshift: ${TSHIFT_X}e-6\n")
  string(APPEND SOURCE_ENTRIES "            f0: 1.0\n")
endforeach()

math(EXPR UPPER_Z "${NUMBER_OF_SOURCES_Z} - 1")

# Generate Z sources
foreach(i RANGE 0 ${UPPER_Z})
  math(EXPR Z_POS "820 + ${i} * 100")
  math(EXPR TSHIFT_Z "(${i} + 1) * 2893")
  string(APPEND SOURCE_ENTRIES "    - moment-tensor:\n")
  string(APPEND SOURCE_ENTRIES "        x: 200 \n")
  string(APPEND SOURCE_ENTRIES "        z: ${Z_POS}\n")
  string(APPEND SOURCE_ENTRIES "        Mxx: 1.0\n")
  string(APPEND SOURCE_ENTRIES "        Mzz: 1.0\n")
  string(APPEND SOURCE_ENTRIES "        Mxz: 0.0\n")
  string(APPEND SOURCE_ENTRIES "        angle: 0.0\n")
  string(APPEND SOURCE_ENTRIES "        Ricker:\n")
  string(APPEND SOURCE_ENTRIES "            factor: 1.805e-10\n")
  string(APPEND SOURCE_ENTRIES "            tshift: ${TSHIFT_Z}e-05\n")
  string(APPEND SOURCE_ENTRIES "            f0: 1.0\n")
endforeach()
