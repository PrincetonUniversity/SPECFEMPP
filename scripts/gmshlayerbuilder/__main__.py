import re
from argparse import Action, ArgumentParser
from pathlib import Path
from typing import override

gmshlayerbuilder_dir = Path(__file__).parent
# ensure gmsh2meshfem is in path.
# There may be a better way to go about this.
try:
    import gmsh2meshfem  # noqa: F401
except ImportError:
    import sys

    sys.path.append(str(gmshlayerbuilder_dir.parent))
    import gmsh2meshfem  # noqa: F401

try:
    import topo_import  # noqa: F401
except ImportError:
    import sys

    sys.path.append(str(gmshlayerbuilder_dir))
    import topo_import  # noqa: F401


from topo_import.tags import BOUNDARY_TYPES, IS_FLUID_PER_MATERIAL_STRCODE  # noqa: E402


class MaterialTypeStringCode(Action):
    def __init__(self, option_strings, dest, nargs=1, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    @override
    def __call__(self, parser, namespace, values, option_string=None):
        valid_chars = "".join(k for k in IS_FLUID_PER_MATERIAL_STRCODE.keys())
        if isinstance(values, str):
            strcode = values.upper()
            # do we have any characters not in IS_FLUID_PER_MATERIAL_STRCODE ?
            if not re.search(f"[^{valid_chars}]", strcode):
                # we know how to handle each character. Good.
                setattr(namespace, self.dest, strcode)
                return

        # failed somewhere, error out
        parser.error(
            f"argument {option_string}: Invalid string-code '{values}' "
            f"(Give a case-insensitive string containing only '{valid_chars}'. e.g. 'sf' for"
            "fluid layer on top of solid layer)"
        )


def shared_args(parser):
    parser.add_argument(
        "topo_file", type=str, help="The name of the topography file to load"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="The name of the folder to store the created files",
    )
    parser.add_argument(
        "--materials",
        help="A list of material types (F for fluid, S for solid, ...) from the "
        "bottom layer to the top.",
        dest="materialtype_strcode",
        default=None,
        action=MaterialTypeStringCode,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Shows a plot of the mesh using matplotlib.",
        dest="should_plot",
    )

    parser.add_argument(
        "--top",
        choices=BOUNDARY_TYPES,
        help="Boundary type on the top (defaults to neumann)",
        dest="bdry_top",
        default="neumann",
    )
    parser.add_argument(
        "--bottom",
        choices=BOUNDARY_TYPES,
        help="Boundary type on the bottom (defaults to neumann)",
        dest="bdry_bottom",
        default="neumann",
    )
    parser.add_argument(
        "--left",
        choices=BOUNDARY_TYPES,
        help="Boundary type on the left (defaults to neumann)",
        dest="bdry_left",
        default="neumann",
    )
    parser.add_argument(
        "--right",
        choices=BOUNDARY_TYPES,
        help="Boundary type on the right (defaults to neumann)",
        dest="bdry_right",
        default="neumann",
    )


def get_parser():
    parser = ArgumentParser(
        prog="gmshLayerBuilder",
        description=(
            "Converts a topography file used by the "
            "meshfem internal mesher and creates an external mesh "
            "with similar bathymetry but nonconforming interfaces."
        ),
    )
    dimension_dispatch = parser.add_subparsers(
        title="dimension", description="2d or 3d mesh", dest="dimension"
    )
    dim3_dispatch = dimension_dispatch.add_parser("3D", aliases=["3d"])
    dim2_dispatch = dimension_dispatch.add_parser("2D", aliases=["2d"])

    shared_args(dim3_dispatch)
    shared_args(dim2_dispatch)

    dim3_dispatch.add_argument(
        "--front",
        choices=BOUNDARY_TYPES,
        help="Boundary type on the front (defaults to neumann). Does nothing for 2d.",
        dest="bdry_front",
        default="neumann",
    )
    dim3_dispatch.add_argument(
        "--back",
        choices=BOUNDARY_TYPES,
        help="Boundary type on the back (defaults to neumann). Does nothing for 2d.",
        dest="bdry_back",
        default="neumann",
    )
    dim3_dispatch.add_argument(
        "--depth_block_km",
        help=(
            "Replaces `DEPTH_BLOCK_KM` parameter in a meshfem3d par file. "
            "If specified, the number of layers is taken to be the number "
            "of interfaces, where a flat-bottom interface is prepended to "
            "the interface list after read-in. Otherwise, the bottom is set "
            'to the first "interface", and the number of layers is taken to '
            "be the number of interfaces minus one. Does nothing for 2D."
        ),
        dest="depth_block_km",
        default=None,
    )
    return parser


def run2D():
    args = get_parser().parse_args()
    print(args)

    if args.dimension.lower() == "2d":
        from gmsh2meshfem.dim2.exporter import Exporter as Exporter2D

        builder = topo_import.builder_from_topo_file2d(
            args.topo_file,
            set_bottom_boundary=args.bdry_bottom,
            set_top_boundary=args.bdry_top,
            set_left_boundary=args.bdry_left,
            set_right_boundary=args.bdry_right,
            materialtype_strcode=args.materialtype_strcode,
        )

        model = builder.create_model()
        if args.should_plot:
            model.plot()

        Exporter2D(
            model, args.output_folder, nonconforming_adjacencies_file="nc_adjacencies"
        ).export_mesh()
    else:
        from gmsh2meshfem.dim3.exporter import Exporter as Exporter3D

        if args.depth_block_km is None:
            depth_block_m = None
        else:
            try:
                depth_block_m = float(args.depth_block_km) * 1000
            except (TypeError, ValueError) as cause:
                e = ValueError(
                    f'Cannot parse --depth_block_km value "{args.depth_block_km}" as a float!'
                )
                raise e from cause
        builder = topo_import.builder_from_topo_file3d(
            args.topo_file,
            set_bottom_boundary=args.bdry_bottom,
            set_top_boundary=args.bdry_top,
            set_left_boundary=args.bdry_left,
            set_right_boundary=args.bdry_right,
            set_back_boundary=args.bdry_back,
            set_front_boundary=args.bdry_front,
            materialtype_strcode=args.materialtype_strcode,
            depth_block_m=depth_block_m,
        )
        model = builder.create_model()
        Exporter3D(
            model, args.output_folder, nonconforming_adjacencies_file="nc_adjacencies"
        ).export_mesh()


if __name__ == "__main__":
    run2D()
