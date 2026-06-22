import os
from pathlib import Path, PurePath

from .model.model import Model
from .model.faces import FaceType
from ..helper import material_properties

# from .model.edges import EdgeType
# from .model.physical_group import (
#     NullPhysicalGroup,
#     PhysicalGroupBase,
# )

NONCONFORMING_CONNECTION_TYPE = 3


def model_face_to_meshfem_face(value: FaceType):
    #   bottom = 1,
    #   right = 2,
    #   top = 3,
    #   left = 4,
    #   front = 5,
    #   back = 6,
    if value == FaceType.BOTTOM:
        return 1
    elif value == FaceType.RIGHT:
        return 2
    elif value == FaceType.TOP:
        return 3
    elif value == FaceType.LEFT:
        return 4
    elif value == FaceType.FRONT:
        return 5
    elif value == FaceType.BACK:
        return 6
    err = ValueError(f"model_Face_to_meshfem_Face(): Cannot process value: {value}")
    err.add_note(
        f"`value` must be one of FaceType.TOP ({FaceType.TOP}),"
        f" FaceType.BOTTOM ({FaceType.BOTTOM}), FaceType.LEFT"
        f" ({FaceType.LEFT}), or FaceType.RIGHT ({FaceType.RIGHT})."
    )
    raise err


class Exporter:
    destination_folder: Path
    mesh_file: PurePath
    node_coords_file: PurePath
    materials_file: PurePath
    nummaterial_velocity_file: PurePath | None
    absorbing_surface_file_xmin: PurePath | None
    absorbing_surface_file_xmax: PurePath | None
    absorbing_surface_file_ymin: PurePath | None
    absorbing_surface_file_ymax: PurePath | None
    absorbing_surface_file_bottom: PurePath | None
    free_or_absorbing_surface_file_zmax: PurePath | None
    nonconforming_adjacencies_file: PurePath | None

    model: Model

    # acoustic_free_surface_physical_group: PhysicalGroupBase
    # absorbing_surface_physical_group: PhysicalGroupBase

    material_models: list[material_properties.MaterialModel] | None

    def __init__(
        self,
        model: Model,
        destination_folder: os.PathLike | str,
        mesh_file: str = "mesh",
        node_coords_file: str = "node_coords",
        materials_file: str = "materials",
        nummaterial_velocity_file: str | None = "nummaterial_velocity_file",
        absorbing_surface_file_xmin: str | None = "absorbing_surface_file_xmin",
        absorbing_surface_file_xmax: str | None = "absorbing_surface_file_xmax",
        absorbing_surface_file_ymin: str | None = "absorbing_surface_file_ymin",
        absorbing_surface_file_ymax: str | None = "absorbing_surface_file_ymax",
        absorbing_surface_file_bottom: str | None = "absorbing_surface_file_bottom",
        free_or_absorbing_surface_file_zmax: str
        | None = "free_or_absorbing_surface_file_zmax",
        nonconforming_adjacencies_file: str | None = "nonconforming_adjacencies",
        material_models: list[material_properties.MaterialModel] | None = None,
    ):
        """Initialize an Exporter3D object to write `model` to files for meshfem.

        Parameters
        ----------
        model : Model
            The model to export
        destination_folder : os.PathLike | str
            Base directory of the output files
        mesh_file : str, optional
            name of the file (path relative to `destination_folder`), by default "mesh"
        node_coords_file : str, optional
            name of the file (path relative to `destination_folder`), by default "node_coords"
        materials_file : str, optional
            name of the file (path relative to `destination_folder`), by default "materials"
        nummaterial_velocity_file : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "nummaterial_velocity_file"
        absorbing_surface_file_xmin : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "absorbing_surface_file_xmin"
        absorbing_surface_file_xmax : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "absorbing_surface_file_xmax"
        absorbing_surface_file_ymin : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "absorbing_surface_file_ymin"
        absorbing_surface_file_ymax : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "absorbing_surface_file_ymax"
        absorbing_surface_file_bottom : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "absorbing_surface_file_bottom"
        free_or_absorbing_surface_file_zmax : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "free_or_absorbing_surface_file_zmax"
        nonconforming_adjacencies_file : str | None, optional
            name of the file (path relative to `destination_folder`), or None for no export, by default "nonconforming_adjacencies"
        material_models : list[MaterialModel] | None, optional
            the material properties for this model. If this is not set, then the
            `nummaterial_velocity_file` is not exported, even if the variable is set.
            By default None
        """
        self.model = model
        self.destination_folder = Path(destination_folder)
        self.mesh_file = PurePath(mesh_file)
        self.node_coords_file = PurePath(node_coords_file)
        self.materials_file = PurePath(materials_file)

        self.nummaterial_velocity_file = (
            None
            if nummaterial_velocity_file is None
            else PurePath(nummaterial_velocity_file)
        )
        self.absorbing_surface_file_xmin = (
            None
            if absorbing_surface_file_xmin is None
            else PurePath(absorbing_surface_file_xmin)
        )
        self.absorbing_surface_file_xmax = (
            None
            if absorbing_surface_file_xmax is None
            else PurePath(absorbing_surface_file_xmax)
        )
        self.absorbing_surface_file_ymin = (
            None
            if absorbing_surface_file_ymin is None
            else PurePath(absorbing_surface_file_ymin)
        )
        self.absorbing_surface_file_ymax = (
            None
            if absorbing_surface_file_ymax is None
            else PurePath(absorbing_surface_file_ymax)
        )
        self.absorbing_surface_file_bottom = (
            None
            if absorbing_surface_file_bottom is None
            else PurePath(absorbing_surface_file_bottom)
        )
        self.free_or_absorbing_surface_file_zmax = (
            None
            if free_or_absorbing_surface_file_zmax is None
            else PurePath(free_or_absorbing_surface_file_zmax)
        )
        self.nonconforming_adjacencies_file = (
            None
            if nonconforming_adjacencies_file is None
            else PurePath(nonconforming_adjacencies_file)
        )
        self.material_models = material_models

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
            f.write(str(nnodes) + "\n")

            for inod in range(nnodes):
                f.write(
                    "{:d} {:.10f} {:.10f} {:.10f}\n".format(
                        inod + 1,
                        nodes_arr[inod, 0],
                        nodes_arr[inod, 1],
                        nodes_arr[inod, 2],
                    )
                )

        nelem = self.model.num_elements

        # =========================
        # elements
        # =========================
        with (self.destination_folder / self.mesh_file).open("w") as f:
            elem_arr = self.model.elements

            f.write(str(nelem) + "\n")
            for ielem in range(nelem):
                f.write(" ".join(f"{k + 1:d}" for k in elem_arr[ielem, :]) + "\n")

        # =========================
        # materials
        # =========================
        with (self.destination_folder / self.materials_file).open("w") as f:
            # no header entry
            mat_arr = self.model.materials

            for ielem in range(nelem):
                f.write(f"{ielem + 1:d} {mat_arr[ielem]:d}\n")

        # =========================
        # nummaterial_velocity_file
        # =========================
        if self.nummaterial_velocity_file and (self.material_models is not None):
            with (self.destination_folder / self.nummaterial_velocity_file).open(
                "w"
            ) as f:
                # domain_ID material_ID rho vp vs Qkappa Qmu anisotropy_flag
                for imat, mat in enumerate(self.material_models):
                    f.write(f"{mat.material_string3D(imat):s}\n")

        # =========================
        # boundaries
        # =========================
        for (filename,) in [
            (self.absorbing_surface_file_xmin,),
            (self.absorbing_surface_file_xmax,),
            (self.absorbing_surface_file_ymin,),
            (self.absorbing_surface_file_ymax,),
            (self.absorbing_surface_file_bottom,),
            (self.free_or_absorbing_surface_file_zmax,),
        ]:
            if filename is None:
                continue

            # TODO write later. below is 2d version:

            # with (self.destination_folder / filename).open("w") as f:
            #     elements, edgetypes = (
            #         self.acoustic_free_surface_physical_group.get_all_edges()
            #     )

            #     f.write(str(elements.shape[0]) + "\n")

            #     for elem, edgetype in zip(elements, edgetypes):
            #         node_indices = self.model.elements[
            #             elem, EdgeType.QUA_9_node_indices_on_type(edgetype)[::2]
            #         ]
            #         f.write(
            #             f"{elem + 1} 2 {node_indices[0] + 1} {node_indices[1] + 1}\n"
            #         )

        # =========================
        # nonconforming adjacencies (if needed)
        # =========================
        if self.nonconforming_adjacencies_file is not None:
            with (self.destination_folder / self.nonconforming_adjacencies_file).open(
                "w"
            ) as f:
                # decompose_mesh export to database is found:
                # fortran/decompose_mesh/write_mesh_databases.F90: 252

                num_pairs = self.model.nonconforming_interfaces.faces_a.shape[0]
                f.write(str(num_pairs * 2) + "\n")

                for ispec_a, ispec_b, face_a, face_b in zip(
                    self.model.nonconforming_interfaces.elements_a,
                    self.model.nonconforming_interfaces.elements_b,
                    self.model.nonconforming_interfaces.faces_a,
                    self.model.nonconforming_interfaces.faces_b,
                ):
                    f.write(
                        f"{ispec_a + 1:d} {ispec_b + 1:d} "
                        f"{NONCONFORMING_CONNECTION_TYPE:d} "
                        f"{model_face_to_meshfem_face(face_a):d}\n"
                    )
                    f.write(
                        f"{ispec_b + 1:d} {ispec_a + 1:d} "
                        f"{NONCONFORMING_CONNECTION_TYPE:d} "
                        f"{model_face_to_meshfem_face(face_b):d}\n"
                    )
