############################################################
##                                                        ##
##   CMakeLists.txt for libscotch (Scotch 5.1.12b)       ##
##                                                        ##
##   Builds: libscotch, libscotcherr, libscotcherrexit   ##
##   Generates: scotch.h, scotchf.h via dummysizes        ##
##   Handles: Bison/Flex parser with last_resort fallback ##
##                                                        ##
############################################################

################
#  dummysizes  #
################

add_executable(dummysizes dummysizes.c)
set_target_properties(dummysizes PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(dummysizes PRIVATE
  SCOTCH_VERSION=${SCOTCH_VERSION}
  SCOTCH_RELEASE=${SCOTCH_RELEASE}
  SCOTCH_PATCHLEVEL=${SCOTCH_PATCHLEVEL})
target_include_directories(dummysizes PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT WIN32)
  target_link_libraries(dummysizes PRIVATE m)
endif()
if(ZLIB_FOUND)
  target_link_libraries(dummysizes PRIVATE ZLIB::ZLIB)
endif()
if(Threads_FOUND)
  target_link_libraries(dummysizes PRIVATE Threads::Threads)
endif()

##############
#  scotch.h  #
##############

add_custom_command(
  OUTPUT ${GENERATED_INCLUDE_DIR}/scotch.h
  COMMAND $<TARGET_FILE:dummysizes>
    ${CMAKE_CURRENT_SOURCE_DIR}/library.h
    ${GENERATED_INCLUDE_DIR}/scotch.h
  DEPENDS dummysizes library.h
  COMMENT "Generating scotch.h from library.h")
add_custom_target(scotch_h DEPENDS ${GENERATED_INCLUDE_DIR}/scotch.h)

###############
#  scotchf.h  #
###############

add_custom_command(
  OUTPUT ${GENERATED_INCLUDE_DIR}/scotchf.h
  COMMAND $<TARGET_FILE:dummysizes>
    ${CMAKE_CURRENT_SOURCE_DIR}/libraryf.h
    ${GENERATED_INCLUDE_DIR}/scotchf.h
  DEPENDS dummysizes libraryf.h
  COMMENT "Generating scotchf.h from libraryf.h")
add_custom_target(scotchf_h DEPENDS ${GENERATED_INCLUDE_DIR}/scotchf.h)

#################
#  parser_yy.c  #
#################

if(BISON_FOUND)
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/parser_yy.c
      ${CMAKE_CURRENT_BINARY_DIR}/parser_ly.h
    COMMAND ${BISON_EXECUTABLE}
      -pscotchyy -y -d -v
      -o ${CMAKE_CURRENT_BINARY_DIR}/parser_yy.c
      ${CMAKE_CURRENT_SOURCE_DIR}/parser_yy.y
    COMMAND ${CMAKE_COMMAND} -E rename
      ${CMAKE_CURRENT_BINARY_DIR}/parser_yy.h
      ${CMAKE_CURRENT_BINARY_DIR}/parser_ly.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/parser_yy.y
    COMMENT "Generating parser_yy.c and parser_ly.h using Bison"
    VERBATIM)
else()
  message(STATUS "Scotch: Bison not found — using pre-generated parser_yy.c")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/parser_yy.c
      ${CMAKE_CURRENT_BINARY_DIR}/parser_ly.h
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_SOURCE_DIR}/last_resort/parser_yy.c
      ${CMAKE_CURRENT_BINARY_DIR}/parser_yy.c
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_SOURCE_DIR}/last_resort/parser_ly.h
      ${CMAKE_CURRENT_BINARY_DIR}/parser_ly.h
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/last_resort/parser_yy.c
      ${CMAKE_CURRENT_SOURCE_DIR}/last_resort/parser_ly.h
    COMMENT "Copying pre-generated parser_yy.c and parser_ly.h")
endif()
add_custom_target(parser_yy_c DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/parser_yy.c
  ${CMAKE_CURRENT_BINARY_DIR}/parser_ly.h)

#################
#  parser_ll.c  #
#################

if(FLEX_FOUND)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/parser_ll.c
    COMMAND ${FLEX_EXECUTABLE}
      -Pscotchyy
      -o${CMAKE_CURRENT_BINARY_DIR}/parser_ll.c
      ${CMAKE_CURRENT_SOURCE_DIR}/parser_ll.l
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/parser_ll.l
      ${CMAKE_CURRENT_BINARY_DIR}/parser_ly.h
    COMMENT "Generating parser_ll.c using Flex"
    VERBATIM)
else()
  message(STATUS "Scotch: Flex not found — using pre-generated parser_ll.c")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/parser_ll.c
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_SOURCE_DIR}/last_resort/parser_ll.c
      ${CMAKE_CURRENT_BINARY_DIR}/parser_ll.c
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/last_resort/parser_ll.c
    COMMENT "Copying pre-generated parser_ll.c")
endif()
add_custom_target(parser_ll_c DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/parser_ll.c)

###############
#  libscotch  #
###############

set(SCOTCH_C_SOURCES
  # Architecture modules
  arch.c
  arch_build.c
  arch_cmplt.c
  arch_cmpltw.c
  arch_deco.c
  arch_hcub.c
  arch_mesh.c
  arch_tleaf.c
  arch_torus.c
  arch_vcmplt.c
  arch_vhcub.c

  # Bipartition graph
  bgraph.c
  bgraph_bipart_bd.c
  bgraph_bipart_df.c
  bgraph_bipart_ex.c
  bgraph_bipart_fm.c
  bgraph_bipart_gg.c
  bgraph_bipart_gp.c
  bgraph_bipart_ml.c
  bgraph_bipart_st.c
  bgraph_bipart_zr.c
  bgraph_check.c
  bgraph_store.c

  # Common utilities (these need -DSCOTCH_COMMON_RENAME)
  common.c
  common_file.c
  common_file_compress.c
  common_file_uncompress.c
  common_integer.c
  common_memory.c
  common_string.c
  common_stub.c

  # Data structures
  gain.c
  geom.c

  # Graph core
  graph.c
  graph_base.c
  graph_check.c
  graph_coarsen.c
  graph_induce.c
  graph_io.c
  graph_io_chac.c
  graph_io_habo.c
  graph_io_mmkt.c
  graph_io_scot.c
  graph_list.c

  # Hall ordering
  hall_order_hd.c
  hall_order_hf.c
  hall_order_hx.c

  # Hierarchical graph
  hgraph.c
  hgraph_check.c
  hgraph_induce.c
  hgraph_order_bl.c
  hgraph_order_cp.c
  hgraph_order_gp.c
  hgraph_order_hd.c
  hgraph_order_hf.c
  hgraph_order_hx.c
  hgraph_order_nd.c
  hgraph_order_si.c
  hgraph_order_st.c

  # Hierarchical mesh
  hmesh.c
  hmesh_check.c
  hmesh_hgraph.c
  hmesh_induce.c
  hmesh_mesh.c
  hmesh_order_bl.c
  hmesh_order_cp.c
  hmesh_order_gp.c
  hmesh_order_gr.c
  hmesh_order_hd.c
  hmesh_order_hf.c
  hmesh_order_hx.c
  hmesh_order_nd.c
  hmesh_order_si.c
  hmesh_order_st.c

  # K-way graph mapping
  kgraph.c
  kgraph_check.c
  kgraph_map_ml.c
  kgraph_map_rb.c
  kgraph_map_rb_map.c
  kgraph_map_rb_part.c
  kgraph_map_st.c

  # Library API wrappers
  library_arch.c
  library_arch_build.c
  library_arch_build_f.c
  library_arch_f.c
  library_geom.c
  library_geom_f.c
  library_graph.c
  library_graph_base.c
  library_graph_base_f.c
  library_graph_check.c
  library_graph_check_f.c
  library_graph_coarsen.c
  library_graph_coarsen_f.c
  library_graph_f.c
  library_graph_io_chac.c
  library_graph_io_chac_f.c
  library_graph_io_habo.c
  library_graph_io_habo_f.c
  library_graph_io_mmkt.c
  library_graph_io_mmkt_f.c
  library_graph_io_scot.c
  library_graph_io_scot_f.c
  library_graph_map.c
  library_graph_map_f.c
  library_graph_map_view.c
  library_graph_map_view_f.c
  library_graph_order.c
  library_graph_order_f.c
  library_mapping.c
  library_mesh.c
  library_mesh_f.c
  library_mesh_graph.c
  library_mesh_graph_f.c
  library_mesh_io_habo.c
  library_mesh_io_habo_f.c
  library_mesh_io_scot.c
  library_mesh_io_scot_f.c
  library_mesh_order.c
  library_mesh_order_f.c
  library_order.c
  library_parser.c
  library_parser_f.c
  library_random.c
  library_random_f.c
  library_strat.c
  library_version.c
  library_version_f.c

  # Mapping
  mapping.c
  mapping_io.c

  # Mesh
  mesh.c
  mesh_check.c
  mesh_coarsen.c
  mesh_graph.c
  mesh_induce_sepa.c
  mesh_io.c
  mesh_io_habo.c
  mesh_io_scot.c

  # Ordering
  order.c
  order_check.c
  order_io.c

  # Parser (static source + generated files)
  parser.c
  ${CMAKE_CURRENT_BINARY_DIR}/parser_ll.c
  ${CMAKE_CURRENT_BINARY_DIR}/parser_yy.c

  # Vertex separator graph
  vgraph.c
  vgraph_check.c
  vgraph_separate_bd.c
  vgraph_separate_es.c
  vgraph_separate_fm.c
  vgraph_separate_gg.c
  vgraph_separate_gp.c
  vgraph_separate_ml.c
  vgraph_separate_st.c
  vgraph_separate_th.c
  vgraph_separate_vw.c
  vgraph_separate_zr.c
  vgraph_store.c

  # Variable mesh separator
  vmesh.c
  vmesh_check.c
  vmesh_separate_fm.c
  vmesh_separate_gg.c
  vmesh_separate_gr.c
  vmesh_separate_ml.c
  vmesh_separate_st.c
  vmesh_separate_zr.c
  vmesh_store.c
)

# common_*.c files need the -DSCOTCH_COMMON_RENAME flag
set(COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/common.c
  ${CMAKE_CURRENT_SOURCE_DIR}/common_file.c
  ${CMAKE_CURRENT_SOURCE_DIR}/common_file_compress.c
  ${CMAKE_CURRENT_SOURCE_DIR}/common_file_uncompress.c
  ${CMAKE_CURRENT_SOURCE_DIR}/common_integer.c
  ${CMAKE_CURRENT_SOURCE_DIR}/common_memory.c
  ${CMAKE_CURRENT_SOURCE_DIR}/common_string.c
  ${CMAKE_CURRENT_SOURCE_DIR}/common_stub.c
)
set_source_files_properties(${COMMON_SOURCES}
  PROPERTIES COMPILE_DEFINITIONS "SCOTCH_COMMON_RENAME")

add_library(scotch STATIC ${SCOTCH_C_SOURCES})
add_library(SCOTCH::scotch ALIAS scotch)

add_dependencies(scotch scotch_h scotchf_h parser_yy_c parser_ll_c)

target_include_directories(scotch
  PUBLIC
    $<BUILD_INTERFACE:${GENERATED_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

if(NOT WIN32)
  target_link_libraries(scotch PUBLIC m)
endif()

if(ZLIB_FOUND)
  target_link_libraries(scotch PRIVATE ZLIB::ZLIB)
endif()

if(Threads_FOUND)
  target_link_libraries(scotch PRIVATE Threads::Threads)
endif()

####################
#  libscotcherr    #
####################

add_library(scotcherr STATIC library_error.c)
add_library(SCOTCH::scotcherr ALIAS scotcherr)
add_dependencies(scotcherr scotch_h)
target_include_directories(scotcherr
  PUBLIC
    $<BUILD_INTERFACE:${GENERATED_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

########################
#  libscotcherrexit    #
########################

add_library(scotcherrexit STATIC library_error_exit.c)
add_library(SCOTCH::scotcherrexit ALIAS scotcherrexit)
add_dependencies(scotcherrexit scotch_h)
target_include_directories(scotcherrexit
  PUBLIC
    $<BUILD_INTERFACE:${GENERATED_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
