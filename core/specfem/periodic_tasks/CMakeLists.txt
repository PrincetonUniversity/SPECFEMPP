file(GLOB_RECURSE POINT_SOURCE_FILES "./*.cpp")

add_library(
        periodic_tasks
        ${POINT_SOURCE_FILES}
)
add_library(specfem::periodic_tasks ALIAS periodic_tasks)

target_link_libraries(
        periodic_tasks
        PUBLIC
        $<$<BOOL:${SPECFEM_ENABLE_VTK}>:${VTK_LIBRARIES}>
        specfem::element
        specfem::enums
        PRIVATE
        specfem::io
        assembly
        specfem::program
)

target_compile_definitions(
        periodic_tasks
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_VTK}>>:-DNO_VTK>
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_HDF5}>>:-DNO_HDF5>
        $<$<BOOL:${SPECFEM_HDF5_IS_PARALLEL}>:-DSPECFEM_HDF5_IS_PARALLEL>
        $<$<BOOL:${__APPLE__}>:__APPLE__>
)

if (SPECFEM_ENABLE_VTK)
    # Registers VTK object factories; without this vtkRenderWindow::New()
    # returns a non-rendering base class and image capture segfaults
    vtk_module_autoinit(TARGETS periodic_tasks MODULES ${VTK_LIBRARIES})
endif()
