file(GLOB POINT_SOURCE_FILES "./*.cpp")

add_library(
        periodic_tasks
        ${POINT_SOURCE_FILES}
)
add_library(specfem::periodic_tasks ALIAS periodic_tasks)

target_link_libraries(
        periodic_tasks
        reader
        writer
        io
)

target_compile_definitions(
        periodic_tasks
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_VTK}>>:-DNO_VTK>
        $<$<BOOL:${__APPLE__}>:__APPLE__>
)

target_link_libraries(
        periodic_tasks
        assembly
        $<$<BOOL:${SPECFEM_ENABLE_VTK}>:${VTK_LIBRARIES}>
)
