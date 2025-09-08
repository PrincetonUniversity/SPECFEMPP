from argparse import ArgumentParser

# ensure _gmsh2meshfem is in path.
# There may be a better way to go about this.
try:
    import _gmsh2meshfem  # noqa: F401
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    import _gmsh2meshfem  # noqa: F401


def get_parser():
    parser = ArgumentParser(
        prog="gmshLayerBuilder",
        description=(
            "Converts a topography file used by the "
            "meshfem internal mesher and creates an external mesh "
            "with similar bathymetry but nonconforming interfaces."
        ),
    )
    parser.add_argument(
        "topo_file", type=str, help="The name of the topography file to load"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="The name of the folder to store the created files",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Shows a plot of the mesh using matplotlib.",
        dest="should_plot",
    )
    return parser


def run2D():
    import _gmsh2meshfem.dim2
    import _gmsh2meshfem.topo_import

    args = get_parser().parse_args()
    builder = _gmsh2meshfem.topo_import.builder_from_topo_file(
        args.topo_file
    )

    model = builder.create_model()
    if args.should_plot:
        model.plot()

    _gmsh2meshfem.dim2.Exporter(
        model, args.output_folder, nonconforming_adjacencies_file="nc_adjacencies"
    ).export_mesh()


if __name__ == "__main__":
    run2D()
