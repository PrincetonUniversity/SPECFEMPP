import numpy as np
import pytest
from gmsh2meshfem.dim3.model.faces import FaceType
from gmsh2meshfem.dim3.model.gmshmodel import GmshModel3D
from gmsh2meshfem.dim3.model.model import Model
from gmsh2meshfem.gmsh_dep import GmshContext


def validate_on_grid(model: Model, grid_coordinates):
    """Verifies the model has elements that take up the given grid.

    Parameters
    ----------
    model : Model
        model to check
    grid_coordinates : np.ndarray
        shape-(2nx+1,2ny+1,2nz+1,3) array of the grid and its midpoints

    Returns
    -------
    np.ndarray
        an array of element indices corresponding to each grid cell.
    """
    ngridpoints_x, ngridpoints_y, ngridpoints_z, _ = grid_coordinates.shape
    nx = ngridpoints_x // 2
    ny = ngridpoints_y // 2
    nz = ngridpoints_z // 2
    assert grid_coordinates.shape == (2 * nx + 1, 2 * ny + 1, 2 * nz + 1, 3), (
        "validate_on_grid needs a shape (2nx+1,2ny+1,2nz+1,3) of points"
    )

    nelem = model.num_elements
    assert model.elements.shape == (nelem, 27), (
        f"elements should be of shape ({nelem},27), Got {model.elements.shape}"
    )
    assert model.materials.shape == (nelem,), (
        f"materials should be of shape ({nelem},), Got {model.elements.shape}"
    )

    def coords_index_closest_to_point(loc, grid=grid_coordinates):
        closest_on_grid = np.unravel_index(
            np.argmin(np.linalg.norm(grid - loc, axis=-1)),
            grid.shape[:-1],
        )
        return (
            closest_on_grid[0],
            closest_on_grid[1],
            closest_on_grid[2],
            np.linalg.norm(loc - grid[*closest_on_grid, :]),
        )

    element_coords = model.nodes[model.elements, :]
    # min distance from any node to the center
    element_length_scale = np.min(
        np.linalg.norm(
            element_coords[:, :26, :] - element_coords[:, 26, None, :], axis=-1
        )
    )
    coords_eps = element_length_scale * 1e-8

    element_hits = np.full((nx, ny, nz), -1, dtype=int)

    # middle index (magic number):
    central_node = 26
    for ielem in range(nelem):
        central_loc = element_coords[ielem, central_node, :]
        px, py, pz, dist = coords_index_closest_to_point(central_loc)
        if dist > element_length_scale / 2:
            continue  # we are outside the grid

        # find which ix,iy,iz this element should be in from central coordinate.
        assert px % 2 == 1, "x-index should be odd (midpoint)"
        assert py % 2 == 1, "y-index should be odd (midpoint)"
        assert pz % 2 == 1, "z-index should be odd (midpoint)"
        assert dist < coords_eps, "Central point is off!"
        ix = px // 2
        iy = py // 2
        iz = pz // 2
        element_hits[ix, iy, iz] = ielem

        # store coordinates of what should be the nodes that make up the element
        surrounding_27 = grid_coordinates[
            ix * 2 : ix * 2 + 3, iy * 2 : iy * 2 + 3, iz * 2 : iz * 2 + 3, :
        ]
        surrounding_hits = np.full((3, 3, 3), -1, dtype=int)

        for inode_elem in range(27):
            node_loc = element_coords[ielem, inode_elem, :]
            jx, jy, jz, dist = coords_index_closest_to_point(node_loc, surrounding_27)
            assert dist < coords_eps, f"Point {inode_elem} is off!"
            surrounding_hits[jx, jy, jz] = inode_elem

        assert np.all(surrounding_hits != -1), (
            f"Not all nodes of element {ielem} @ ({ix},{iy},{iz}) hit!"
        )

    assert np.all(element_hits != -1), "Not all elements on grid have been hit!"
    return element_hits


def validate_boundary(
    model: Model,
    grid_boundary: np.ndarray,
    elems_bdry: np.ndarray,
    context_string: str = "",
):
    """Tests if the model has boundary faces corresponding to the given grid and element ids.

    Parameters
    ----------
    model : Model
        model whose BoundarySpec to validate
    grid_boundary : np.ndarray
        shape - (2m+1, 2n+1, 3) array of grid points for the boundary faces and the midpoints
    elems_bdry : np.ndarray
        shape - (m,n) array of element indices to test against
    """
    m, n = elems_bdry.shape

    assert grid_boundary.shape == (2 * m + 1, 2 * n + 1, 3), (
        context_string
        + "Test was configured incorrectly: grid_boundary has wrong shape. "
        f"Expected {(2 * m + 1, 2 * n + 1, 3)}, got {grid_boundary.shape}"
    )
    # min distance along an axis
    element_length_scale = min(
        np.min(
            np.linalg.norm(
                grid_boundary[2::2, :, :] - grid_boundary[:-2:2, :, :], axis=-1
            )
        ),
        np.min(
            np.linalg.norm(
                grid_boundary[:, 2::2, :] - grid_boundary[:, :-2:2, :], axis=-1
            )
        ),
    )

    for i in range(m):
        for j in range(n):
            ielem = elems_bdry[i, j]
            facepoints_to_check = grid_boundary[
                2 * i : 2 * i + 3, 2 * j : 2 * j + 3, :
            ].reshape(-1, 3)

            bdhit = False

            faces_with_ielem = np.where(model.boundary_faces.element_inds == ielem)
            assert len(faces_with_ielem) == 1  # elem_inds has 1 index.
            for iface in faces_with_ielem[0]:
                facetype: int = model.boundary_faces.element_faces[iface]  # type: ignore
                facepoint_inds = model.elements[
                    ielem, FaceType.HEX_27_node_indices_on_type(facetype)
                ]
                facepoints = model.nodes[facepoint_inds, :]

                was_check_hit = np.any(
                    np.linalg.norm(
                        facepoints_to_check[:, None, :] - facepoints[None, :, :],
                        axis=-1,
                    )
                    < element_length_scale * 1e-8,
                    axis=1,
                )
                if np.all(was_check_hit):
                    bdhit = True
                    break

            assert bdhit, (
                context_string
                + f"Boundary on grid index ({i}, {j}) not hit. Expected element {ielem}."
            )


@pytest.mark.parametrize("nx,ny,nz", [(1, 1, 1), (3, 3, 3), (2, 3, 4)])
def test_grid(box_builder, nx, ny, nz):
    material_id = 1
    # second order, nodes should be spaced like this:
    xgrid = np.linspace(-1, 1, 2 * nx + 1)
    ygrid = np.linspace(-1, 1, 2 * ny + 1)
    zgrid = np.linspace(-1, 1, 2 * nz + 1)

    with GmshContext() as gmsh:
        vol = box_builder.build(gmsh, nx, ny, nz)
        gmsh.model.geo.synchronize()
        box_builder.post_geo_sync()

        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate()

        gmshmodel = GmshModel3D(gmsh, vol)
        model_exportable = gmshmodel.to_model({vol: material_id})

    nelem = nx * ny * nz
    assert model_exportable.num_elements == nelem, (
        "Incorrect number of elements found. Should be "
        f"{nx} ⨉ {ny} ⨉ {nz} = {nelem}. Got {model_exportable.num_elements}"
    )

    num_bd_faces = 2 * (nx * ny) + 2 * (nx * nz) + 2 * (ny * nz)
    assert model_exportable.boundary_faces.num_faces == num_bd_faces

    fullgrid = np.stack(np.meshgrid(xgrid, ygrid, zgrid, indexing="ij"), axis=-1)

    hit_elems = validate_on_grid(
        model_exportable,
        fullgrid,
    )

    validate_boundary(
        model_exportable, fullgrid[0, :, :, :], hit_elems[0, :, :], "[LEFT] "
    )
    validate_boundary(
        model_exportable, fullgrid[-1, :, :, :], hit_elems[-1, :, :], "[RIGHT] "
    )
    validate_boundary(
        model_exportable, fullgrid[:, 0, :, :], hit_elems[:, 0, :], "[FRONT] "
    )
    validate_boundary(
        model_exportable, fullgrid[:, -1, :, :], hit_elems[:, -1, :], "[BACK] "
    )
    validate_boundary(
        model_exportable, fullgrid[:, :, 0, :], hit_elems[:, :, 0], "[BOTTOM] "
    )
    validate_boundary(
        model_exportable, fullgrid[:, :, -1, :], hit_elems[:, :, -1], "[TOP] "
    )


def test_grid_from_two_volumes(box_builder):
    nx = 3
    ny = 5
    nz = 4
    nz_lower = 2
    material_id1 = 1
    material_id2 = 2
    # second order, nodes should be spaced like this:
    xgrid = np.linspace(-1, 1, 2 * nx + 1)
    ygrid = np.linspace(-1, 1, 2 * ny + 1)
    zgrid = np.linspace(-1, 1, 2 * nz + 1)
    nz_upper = nz - nz_lower

    z_interface = zgrid[2 * nz_lower]

    with GmshContext() as gmsh:
        vol1 = box_builder.build(
            gmsh, nx, ny, nz_lower, bbox_min=(-1, -1, -1), bbox_max=(1, 1, z_interface)
        )
        vol2 = box_builder.build(
            gmsh, nx, ny, nz_upper, bbox_min=(-1, -1, z_interface), bbox_max=(1, 1, 1)
        )
        gmsh.model.geo.synchronize()
        box_builder.post_geo_sync()
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate()

        gmshmodel = GmshModel3D(gmsh, [vol1, vol2])
        model_exportable = gmshmodel.to_model({vol1: material_id1, vol2: material_id2})

    nelem = nx * ny * nz
    assert model_exportable.num_elements == nelem, (
        "Incorrect number of elements found. Should be "
        f"{nx} ⨉ {ny} ⨉ {nz} = {nelem}. Got {model_exportable.num_elements}"
    )

    grid = np.stack(np.meshgrid(xgrid, ygrid, zgrid, indexing="ij"), axis=-1)
    validate_on_grid(
        model_exportable,
        grid[:, :, : 2 * nz_lower + 1, :],
    )
    validate_on_grid(
        model_exportable,
        grid[:, :, 2 * nz_lower :, :],
    )

    # nonconforming interface should only be along z_interface
    assert model_exportable.nonconforming_interfaces.elements_a.shape == (nx * ny,)


@pytest.mark.parametrize(
    "nx1,ny1,nz1,nx2,ny2,nz2",
    [(1, 1, 1, 3, 3, 3), (5, 5, 5, 4, 4, 4), (10, 15, 5, 25, 36, 15)],
)
def test_nonconforming(box_builder, nx1, ny1, nz1, nx2, ny2, nz2):
    material_id1 = 1
    material_id2 = 2
    with GmshContext() as gmsh:
        vol1 = box_builder.build(
            gmsh, nx1, ny1, nz1, bbox_min=(-1, -1, -2), bbox_max=(1, 1, 0)
        )
        vol2 = box_builder.build(
            gmsh, nx2, ny2, nz2, bbox_min=(-1, -1, 0), bbox_max=(1, 1, 2)
        )
        gmsh.model.geo.synchronize()
        box_builder.post_geo_sync()
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate()

        gmshmodel = GmshModel3D(gmsh, [vol1, vol2])
        model_exportable = gmshmodel.to_model({vol1: material_id1, vol2: material_id2})

    nsurf1 = nx1 * ny1
    nsurf2 = nx2 * ny2

    uniques = np.unique(
        np.concatenate(
            [
                model_exportable.nonconforming_interfaces.elements_a,
                model_exportable.nonconforming_interfaces.elements_b,
            ]
        )
    )
    n_uniques_surf1 = np.count_nonzero(
        model_exportable.materials[uniques] == material_id1
    )
    n_uniques_surf2 = np.count_nonzero(
        model_exportable.materials[uniques] == material_id2
    )
    assert n_uniques_surf1 == nsurf1, (
        f"Expected {nsurf1} elements from volume 1 contributing to nonconforming interfaces."
        f" Got {n_uniques_surf1}"
    )
    assert n_uniques_surf2 == nsurf2, (
        f"Expected {nsurf2} elements from volume 2 contributing to nonconforming interfaces."
        f" Got {n_uniques_surf2}"
    )
