from pathlib import Path, PurePath
import os

import numpy as np

from _gmshlayerbuilder.dim2.layer_builder.model import Model
from _gmshlayerbuilder.dim2.layer_builder.model.edges import EdgeType


NONCONFORMING_CONNECTION_TYPE = 3


def model_edge_to_meshfem_edge(value: EdgeType):
    if value == EdgeType.BOTTOM:
        return 1
    elif value == EdgeType.RIGHT:
        return 2
    elif value == EdgeType.TOP:
        return 3
    elif value == EdgeType.LEFT:
        return 4
    err = ValueError(f"model_edge_to_meshfem_edge(): Cannot process value: {value}")
    err.add_note(
        f"`value` must be one of EdgeType.TOP ({EdgeType.TOP}),"
        f" EdgeType.BOTTOM ({EdgeType.BOTTOM}), EdgeType.LEFT"
        f" ({EdgeType.LEFT}), or EdgeType.RIGHT ({EdgeType.RIGHT})."
    )
    raise err


class Exporter2D:
    destination_folder: Path
    mesh_file: PurePath
    node_coords_file: PurePath
    materials_file: PurePath
    free_surface_file: PurePath
    axial_elements_file: PurePath | None
    absorbing_surface_file: PurePath | None
    acoustic_forcing_surface_file: PurePath | None
    absorbing_cpml_file: PurePath | None
    tangential_detection_curve_file: PurePath | None
    nonconforming_adjacencies_file: PurePath | None

    model: Model

    def __init__(
        self,
        model: Model,
        destination_folder: os.PathLike | str,
        mesh_file: str = "mesh",
        node_coords_file: str = "node_coords",
        materials_file: str = "materials",
        free_surface_file: str = "free_surface",
        axial_elements_file: str | None = None,
        absorbing_surface_file: str | None = None,
        acoustic_forcing_surface_file: str | None = None,
        absorbing_cpml_file: str | None = None,
        tangential_detection_curve_file: str | None = None,
        nonconforming_adjacencies_file: str | None = None,
    ):
        """Initialize an Exporter2D object to write `model` to files for meshfem.


        Args:
            model (Model): The model to export
            destination_folder (os.PathLike | str): Base directory of the output files
            mesh_file (str, optional): name of the file (path
                relative to `destination_folder`). Defaults to "mesh".
            node_coords_file (str, optional): name of the file
                (path relative to `destination_folder`). Defaults
                to "node_coords".
            materials_file (str, optional): name of the file (path
                relative to `destination_folder`). Defaults to
                "materials".
            free_surface_file (str, optional): name of the file
                (path relative to `destination_folder`). Defaults
                to "free_surface".
            axial_elements_file (str | None, optional): name of the file
                (path relative to `destination_folder`), or None
                for no export. Defaults to None.
            absorbing_surface_file (str | None, optional): name of the file
                (path relative to `destination_folder`), or None
                for no export. Defaults to None.
            acoustic_forcing_surface_file (str | None, optional): name of
                the file (path relative to `destination_folder`),
                or None for no export. Defaults to None.
            absorbing_cpml_file (str | None, optional): name of the file
                (path relative to `destination_folder`), or None
                for no export. Defaults to None.
            tangential_detection_curve_file (str | None, optional): name
                of the file (path relative to `destination_folder`),
                or None for no export. Defaults to None.
            nonconforming_adjacencies_file (str | None, optional): name of
                the file (path relative to `destination_folder`),
                or None for no export. Defaults to None.
        """
        self.model = model
        self.destination_folder = Path(destination_folder)
        self.mesh_file = PurePath(mesh_file)
        self.node_coords_file = PurePath(node_coords_file)
        self.materials_file = PurePath(materials_file)
        self.free_surface_file = PurePath(free_surface_file)
        self.axial_elements_file = (
            None if axial_elements_file is None else PurePath(axial_elements_file)
        )
        self.absorbing_surface_file = (
            None if absorbing_surface_file is None else PurePath(absorbing_surface_file)
        )
        self.acoustic_forcing_surface_file = (
            None
            if acoustic_forcing_surface_file is None
            else PurePath(acoustic_forcing_surface_file)
        )
        self.absorbing_cpml_file = (
            None if absorbing_cpml_file is None else PurePath(absorbing_cpml_file)
        )
        self.tangential_detection_curve_file = (
            None
            if tangential_detection_curve_file is None
            else PurePath(tangential_detection_curve_file)
        )
        self.nonconforming_adjacencies_file = (
            None
            if nonconforming_adjacencies_file is None
            else PurePath(nonconforming_adjacencies_file)
        )

    def export_mesh(self):

        if not self.destination_folder.exists():
            self.destination_folder.mkdir()

        # =========================
        # node coords
        # =========================
        with (self.destination_folder / self.node_coords_file).open("w") as f:
            nodes_arr = self.model.nodes

            # header is number of lines (1 line per node)
            nnodes = nodes_arr.shape[0]
            f.write(str(nnodes))

            # always write in 3 values. Technically,
            # nodes_arr should be 3 as well, but we do this
            # just for sanity
            nodes_dim = nodes_arr.shape[1]
            pts = np.zeros((3,))
            for inod in range(nnodes):
                pts[:nodes_dim] = nodes_arr[inod, :]
                f.write(f"\n {pts[0]:.10f} {pts[1]:.10f} {pts[2]:.10f}")

        nelem = self.model.elements.shape[0]

        # =========================
        # elements
        # =========================
        with (self.destination_folder / self.mesh_file).open("w") as f:
            elem_arr = self.model.elements

            f.write(str(nelem))
            for ielem in range(nelem):
                f.write("\n " + " ".join(f"{k + 1:d}" for k in elem_arr[ielem, :]))

        # =========================
        # materials
        # =========================
        with (self.destination_folder / self.materials_file).open("w") as f:
            f.write("\n".join(str(mat) for mat in self.model.materials))

        # =========================
        # free surface
        # =========================
        with (self.destination_folder / self.free_surface_file).open("w") as f:
            # NotImplemented
            f.write(str(0))

        # =========================
        # absorbing bdries (if needed)
        # =========================
        if self.absorbing_surface_file is not None:
            with (self.destination_folder / self.absorbing_surface_file).open("w") as f:
                # NotImplemented
                f.write(str(0))

        # =========================
        # acoustic forcing (if needed)
        # =========================
        if self.acoustic_forcing_surface_file is not None:
            with (self.destination_folder / self.acoustic_forcing_surface_file).open(
                "w"
            ) as f:
                # NotImplemented
                f.write(str(0))

        # =========================
        # absorbing cpml (if needed)
        # =========================
        if self.absorbing_cpml_file is not None:
            with (self.destination_folder / self.absorbing_cpml_file).open("w") as f:
                # NotImplemented
                f.write(str(0))

        # =========================
        # tangential curve (if needed)
        # =========================
        if self.tangential_detection_curve_file is not None:
            with (self.destination_folder / self.tangential_detection_curve_file).open(
                "w"
            ) as f:
                # NotImplemented
                f.write(str(0))

        # =========================
        # nonconforming adjacencies (if needed)
        # =========================
        if self.nonconforming_adjacencies_file is not None:
            with (self.destination_folder / self.nonconforming_adjacencies_file).open(
                "w"
            ) as f:
                num_pairs = self.model.nonconforming_interfaces.edges_a.shape[0]
                f.write(str(num_pairs * 2))

                for ispec_a, ispec_b, edge_a, edge_b in zip(
                    self.model.nonconforming_interfaces.elements_a,
                    self.model.nonconforming_interfaces.elements_b,
                    self.model.nonconforming_interfaces.edges_a,
                    self.model.nonconforming_interfaces.edges_b,
                ):
                    f.write(
                        f"\n{ispec_a + 1:d} {ispec_b + 1:d} "
                        f"{NONCONFORMING_CONNECTION_TYPE:d} "
                        f"{model_edge_to_meshfem_edge(edge_b):d}"
                    )
                    f.write(
                        f"\n{ispec_b + 1:d} {ispec_a + 1:d} "
                        f"{NONCONFORMING_CONNECTION_TYPE:d} "
                        f"{model_edge_to_meshfem_edge(edge_a):d}"
                    )
