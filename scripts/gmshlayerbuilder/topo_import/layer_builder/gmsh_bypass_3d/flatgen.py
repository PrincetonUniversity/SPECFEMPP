import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..layer3d import Layer3D, LayerBoundary3D
from ..layer_boundary_type.grid3d import Gridded3D

SPECFEM_NODE_LAYOUT = (
    np.array(
        [
            [[1, 13, 5], [12, 25, 20], [4, 16, 8]],
            [[9, 22, 17], [21, 27, 26], [11, 24, 19]],
            [[2, 14, 6], [10, 23, 18], [3, 15, 7]],
        ],
        dtype=int,
    )
    - 1
)


class SpecfemFaceType(IntEnum):
    BOTTOM = 1
    RIGHT = 2
    TOP = 3
    LEFT = 4
    FRONT = 5
    BACK = 6


@dataclass(frozen=True)
class FlatLayer:
    """A regular grid of elements spanning [0,1]^3, with the given resolutions in each dimension."""

    nx: int
    ny: int
    nz: int

    nodes: np.ndarray = field(init=False)
    elements: np.ndarray = field(init=False)

    element_inds_grid: np.ndarray = field(init=False)

    @property
    def nelem(self):
        return self.elements.shape[0]

    @property
    def nnodes(self):
        return self.nodes.shape[0]

    def __post_init__(self):
        nodelocs = np.stack(
            np.meshgrid(
                np.linspace(0, 1, 2 * self.nx + 1),
                np.linspace(0, 1, 2 * self.ny + 1),
                np.linspace(0, 1, 2 * self.nz + 1),
                indexing="ij",
            ),
            axis=-1,
        )
        node_inds = np.arange(nodelocs.size // 3).reshape(nodelocs.shape[:-1])
        elements = np.stack(
            [
                # corners (1-8)
                node_inds[:-1:2, :-1:2, :-1:2],
                node_inds[2::2, :-1:2, :-1:2],
                node_inds[2::2, 2::2, :-1:2],
                node_inds[:-1:2, 2::2, :-1:2],
                node_inds[:-1:2, :-1:2, 2::2],
                node_inds[2::2, :-1:2, 2::2],
                node_inds[2::2, 2::2, 2::2],
                node_inds[:-1:2, 2::2, 2::2],
                # edges (9-20)
                node_inds[1::2, :-1:2, :-1:2],
                node_inds[2::2, 1::2, :-1:2],
                node_inds[1::2, 2::2, :-1:2],
                node_inds[:-1:2, 1::2, :-1:2],
                node_inds[:-1:2, :-1:2, 1::2],
                node_inds[2::2, :-1:2, 1::2],
                node_inds[2::2, 2::2, 1::2],
                node_inds[:-1:2, 2::2, 1::2],
                node_inds[1::2, :-1:2, 2::2],
                node_inds[2::2, 1::2, 2::2],
                node_inds[1::2, 2::2, 2::2],
                node_inds[:-1:2, 1::2, 2::2],
                # sides (21-26)
                node_inds[1::2, 1::2, :-1:2],
                node_inds[1::2, :-1:2, 1::2],
                node_inds[2::2, 1::2, 1::2],
                node_inds[1::2, 2::2, 1::2],
                node_inds[:-1:2, 1::2, 1::2],
                node_inds[1::2, 1::2, 2::2],
                # center (27)
                node_inds[1::2, 1::2, 1::2],
            ],
            axis=-1,
        )

        object.__setattr__(self, "nodes", nodelocs.reshape((-1, 3)))
        object.__setattr__(self, "elements", elements.reshape((-1, 27)))
        object.__setattr__(
            self,
            "element_inds_grid",
            np.arange(self.nelem).reshape((self.nx, self.ny, self.nz)),
        )


def compute_intersection_inds(n1: int, n2: int) -> Iterable[tuple[int, int]]:
    """Computes a list of index pairs [(i,j),...] for intersecting elements

    (i,j) is in this list iff the intervals
           (i/n1, (i+1)/n1) and (j/n2, (j+1)/n2)
    intersect.

    Parameters
    ----------
    n1 : int
        denominator of the first subdivision
    n2 : int
        denominator of the second subdivision

    Returns
    -------
    Iterable[tuple[int, int]]
        list of intersecting elements
    """
    intersecting_inds = []

    for i in range(n1):
        # range from n2*i - n1 < n1*j < n2*(i+1)

        n1jlow = n2 * i - n1
        j = (n1jlow // n1) + 1
        while n1 * j < n2 * (i + 1):
            intersecting_inds.append((i, j))
            j += 1

    return intersecting_inds


def layer_boundary_to_interpolator(layer_boundary: LayerBoundary3D):
    assert isinstance(layer_boundary, LayerBoundary3D)
    if isinstance(layer_boundary, Gridded3D):
        return RegularGridInterpolator(
            (
                layer_boundary.xmin
                + np.arange(layer_boundary.nx) * layer_boundary.x_spacing,
                layer_boundary.ymin
                + np.arange(layer_boundary.ny) * layer_boundary.y_spacing,
            ),
            layer_boundary.elevations,
            method="linear",
            bounds_error=False,
            fill_value=None,  # type: ignore
        )
    else:
        e = NotImplementedError(
            f'Cannot handle LayerBoundary3D "{type(layer_boundary)}".'
        )
        raise e


@dataclass
class Mesher:
    mesh_min_x: float
    mesh_max_x: float
    mesh_min_y: float
    mesh_max_y: float

    boundaries: list[LayerBoundary3D]
    layers: list[Layer3D]

    export_bdry_top: bool
    export_bdry_bottom: bool
    export_bdry_left: bool
    export_bdry_right: bool
    export_bdry_front: bool
    export_bdry_back: bool

    @property
    def mesh_size_x(self):
        return self.mesh_max_x - self.mesh_min_x

    @property
    def mesh_size_y(self):
        return self.mesh_max_y - self.mesh_min_y

    def export_mesh(self, outfol: Path | str):
        outfol = Path(outfol)
        discretization = [
            FlatLayer(nx=layer.nx, ny=layer.ny, nz=layer.nz) for layer in self.layers
        ]
        nodes = np.empty((sum(layer.nodes.shape[0] for layer in discretization), 3))
        elements = np.empty(
            (sum(layer.elements.shape[0] for layer in discretization), 27), dtype=int
        )
        absorbing_bottom = np.empty(
            (discretization[0].nx * discretization[0].ny, 10), dtype=int
        )
        absorbing_top = np.empty(
            (discretization[-1].nx * discretization[-1].ny, 10), dtype=int
        )
        absorbing_xmin = np.empty(
            (sum(layer.nz * layer.ny for layer in discretization), 10), dtype=int
        )
        absorbing_xmax = np.empty(
            (sum(layer.nz * layer.ny for layer in discretization), 10), dtype=int
        )
        absorbing_ymin = np.empty(
            (sum(layer.nz * layer.nx for layer in discretization), 10), dtype=int
        )
        absorbing_ymax = np.empty(
            (sum(layer.nz * layer.nx for layer in discretization), 10), dtype=int
        )
        materials = np.empty(
            (sum(layer.elements.shape[0] for layer in discretization),), dtype=int
        )
        nc_adjacency_lists = []

        elem_offset = 0
        node_offset = 0
        absorbing_x_offset = 0
        absorbing_y_offset = 0

        def flat_interpolator(height):
            return RegularGridInterpolator(
                (
                    [self.mesh_min_x, self.mesh_max_x],
                    [self.mesh_min_y, self.mesh_max_y],
                ),
                np.full((2, 2), height),
                method="linear",
                bounds_error=False,
                fill_value=None,  # type: ignore
            )

        interpolators = list(
            itertools.pairwise(
                layer_boundary_to_interpolator(lb) for lb in self.boundaries
            )
        )
        for ilayer, layer in enumerate(discretization):
            nnodes = layer.nnodes
            nelem = layer.nelem

            elements[elem_offset : elem_offset + nelem, :] = (
                layer.elements + node_offset
            )
            materials[elem_offset : elem_offset + nelem] = ilayer

            # transform
            nodeslice = slice(node_offset, node_offset + nnodes)
            nodes[nodeslice, 0] = self.mesh_min_x + layer.nodes[:, 0] * self.mesh_size_x
            nodes[nodeslice, 1] = self.mesh_min_y + layer.nodes[:, 1] * self.mesh_size_y
            layerbot, layertop = interpolators[ilayer]
            nodes[nodeslice, 2] = (1 - layer.nodes[:, 2]) * layerbot(
                (nodes[nodeslice, 0], nodes[nodeslice, 0])
            ) + layer.nodes[:, 2] * layertop((nodes[nodeslice, 0], nodes[nodeslice, 0]))

            if ilayer == 0:
                absorbing_bottom[:, 0] = layer.element_inds_grid[:, :, 0].reshape((-1,))
                absorbing_bottom[:, 1:] = (
                    layer.elements[
                        absorbing_bottom[:, 0, None],
                        SPECFEM_NODE_LAYOUT[:, :, 0].reshape((-1,)),
                    ]
                    + node_offset
                )
                absorbing_bottom[:, 0] += elem_offset

            if ilayer == len(discretization) - 1:
                absorbing_top[:, 0] = layer.element_inds_grid[:, :, -1].reshape((-1,))
                absorbing_top[:, 1:] = (
                    layer.elements[
                        absorbing_top[:, 0, None],
                        SPECFEM_NODE_LAYOUT[:, :, -1].reshape((-1,)),
                    ]
                    + node_offset
                )
                absorbing_top[:, 0] += elem_offset

            absorbing_x_slice = slice(
                absorbing_x_offset, absorbing_x_offset + layer.ny * layer.nz
            )
            absorbing_y_slice = slice(
                absorbing_y_offset, absorbing_y_offset + layer.nx * layer.nz
            )
            absorbing_xmin[absorbing_x_slice, 0] = layer.element_inds_grid[
                0, :, :
            ].reshape((-1,))
            absorbing_xmax[absorbing_x_slice, 0] = layer.element_inds_grid[
                -1, :, :
            ].reshape((-1,))
            absorbing_ymin[absorbing_y_slice, 0] = layer.element_inds_grid[
                :, 0, :
            ].reshape((-1,))
            absorbing_ymax[absorbing_y_slice, 0] = layer.element_inds_grid[
                :, -1, :
            ].reshape((-1,))
            absorbing_xmin[absorbing_x_slice, 1:] = (
                layer.elements[
                    absorbing_xmin[absorbing_x_slice, 0, None],
                    SPECFEM_NODE_LAYOUT[0, :, :].reshape((-1,)),
                ]
                + node_offset
            )
            absorbing_xmax[absorbing_x_slice, 1:] = (
                layer.elements[
                    absorbing_xmax[absorbing_x_slice, 0, None],
                    SPECFEM_NODE_LAYOUT[-1, :, :].reshape((-1,)),
                ]
                + node_offset
            )
            absorbing_ymin[absorbing_y_slice, 1:] = (
                layer.elements[
                    absorbing_ymin[absorbing_y_slice, 0, None],
                    SPECFEM_NODE_LAYOUT[:, 0, :].reshape((-1,)),
                ]
                + node_offset
            )
            absorbing_ymax[absorbing_y_slice, 1:] = (
                layer.elements[
                    absorbing_ymax[absorbing_y_slice, 0, None],
                    SPECFEM_NODE_LAYOUT[:, -1, :].reshape((-1,)),
                ]
                + node_offset
            )
            absorbing_xmin[absorbing_x_slice, 0] += elem_offset
            absorbing_xmax[absorbing_x_slice, 0] += elem_offset
            absorbing_ymin[absorbing_y_slice, 0] += elem_offset
            absorbing_ymax[absorbing_y_slice, 0] += elem_offset

            if ilayer > 0:
                # adj with below
                layer_below = discretization[ilayer - 1]
                elem_offset_below = elem_offset - layer_below.elements.shape[0]

                intersecting_x = np.array(
                    compute_intersection_inds(layer_below.nx, layer.nx), dtype=int
                )[:, None, :]
                intersecting_y = np.array(
                    compute_intersection_inds(layer_below.ny, layer.ny), dtype=int
                )[None, :, :]
                elem_low = (
                    layer_below.element_inds_grid[
                        intersecting_x[..., 0], intersecting_y[..., 0], -1
                    ].reshape((-1,))
                    + elem_offset_below
                )
                elem_high = (
                    layer.element_inds_grid[
                        intersecting_x[..., 1], intersecting_y[..., 1], 0
                    ].reshape((-1,))
                    + elem_offset
                )

                num_intersections = elem_low.size
                intersections_arr = np.empty((num_intersections * 2, 4), dtype=int)
                intersections_arr[::2, 0] = elem_low
                intersections_arr[1::2, 1] = elem_low
                intersections_arr[::2, 1] = elem_high
                intersections_arr[1::2, 0] = elem_high

                intersections_arr[:, 2] = 3
                intersections_arr[::2, 3] = SpecfemFaceType.TOP  # top of lower
                intersections_arr[1::2, 3] = SpecfemFaceType.BOTTOM  # bottom of upper

                nc_adjacency_lists.append(intersections_arr)

            node_offset += nnodes
            elem_offset += nelem
            absorbing_x_offset += layer.ny * layer.nz
            absorbing_y_offset += layer.nx * layer.nz

        with (outfol / "node_coords").open("w") as f:
            f.write(f"{nodes.shape[0]:d}\n")
            for inod in range(nodes.shape[0]):
                f.write(f"{inod + 1:d}")
                for idim in range(nodes.shape[1]):
                    f.write(f" {nodes[inod, idim]:.8f}")
                f.write("\n")

        with (outfol / "mesh").open("w") as f:
            f.write(f"{elements.shape[0]:d}\n")
            for ielem in range(elements.shape[0]):
                f.write(f"{ielem + 1:d}")
                for inod in range(elements.shape[1]):
                    f.write(f" {elements[ielem, inod] + 1:d}")
                f.write("\n")

        for absfile, absdat, should_export in [
            ("absorbing_surface_file_xmin", absorbing_xmin, self.export_bdry_left),
            ("absorbing_surface_file_xmax", absorbing_xmax, self.export_bdry_right),
            ("absorbing_surface_file_ymin", absorbing_ymin, self.export_bdry_front),
            ("absorbing_surface_file_ymax", absorbing_ymax, self.export_bdry_back),
            (
                "absorbing_surface_file_bottom",
                absorbing_bottom,
                self.export_bdry_bottom,
            ),
            (
                "free_or_absorbing_surface_file_zmax",
                absorbing_top,
                self.export_bdry_top,
            ),
        ]:
            if not should_export:
                continue
            with (outfol / absfile).open("w") as f:
                f.write(f"{absdat.shape[0]:d}\n")
                for ielem in range(absdat.shape[0]):
                    for inod in range(absdat.shape[1]):
                        f.write(f" {absdat[ielem, inod] + 1:d}")
                    f.write("\n")

        with (outfol / "materials").open("w") as f:
            for ielem in range(elements.shape[0]):
                f.write(f"{ielem + 1:d} {materials[ielem] + 1:d}\n")

        with (outfol / "nc_adjacencies").open("w") as f:
            nc_adjacency_list = np.concatenate(nc_adjacency_lists, axis=0)
            f.write(f"{nc_adjacency_list.shape[0]:d}\n")
            for iadj in range(nc_adjacency_list.shape[0]):
                f.write(
                    f"{nc_adjacency_list[iadj, 0] + 1:d} "
                    f"{nc_adjacency_list[iadj, 1] + 1:d} "
                    f"{nc_adjacency_list[iadj, 2]:d} "
                    f"{nc_adjacency_list[iadj, 3]:d}\n"
                )
