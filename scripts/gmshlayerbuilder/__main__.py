from argparse import ArgumentParser

try:
    import _gmshlayerbuilder
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    import _gmshlayerbuilder


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


def run():
    args = get_parser().parse_args()
    builder = _gmshlayerbuilder.dim2.topo_reader.builder_from_topo_file(args.topo_file)

    model = builder.create_model()
    if args.should_plot:
        _gmshlayerbuilder.dim2.plotter.plot_model(model)

    _gmshlayerbuilder.dim2.exporter.Exporter2D(
        model, args.output_folder, nonconforming_adjacencies_file="nc_adjacencies"
    ).export_mesh()


if __name__ == "__main__":
    run()
