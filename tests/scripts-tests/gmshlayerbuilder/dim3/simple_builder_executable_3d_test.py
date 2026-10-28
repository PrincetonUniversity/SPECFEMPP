from pathlib import Path

import numpy as np

thisdir = Path(__file__).parent


def test_gmshlayerbuilder(path_scripts_root, execute_script, tmp_path):
    gmshlayerbuilder = path_scripts_root / "gmshlayerbuilder"
    topo_in_dir = thisdir / "trialmesh" / "interfaces.txt"
    mesh_out_dir = tmp_path / "meshes" / "test_gmshlayerbuilder"
    cmd = [
        str(gmshlayerbuilder),
        "3d",
        "--top",
        "neumann",
        "--bottom",
        "neumann",
        "--left",
        "neumann",
        "--right",
        "neumann",
        "--front",
        "neumann",
        "--back",
        "neumann",
        str(topo_in_dir),
        str(mesh_out_dir),
    ]
    mesh_out_dir.mkdir(parents=True, exist_ok=True)
    execute_script(cmd, False)

    # verify correct mesh
    layer_resolutions = [(13, 9, 13), (5, 3, 5)]
    nelem_per_block = [nx * ny * nz for nx, ny, nz in layer_resolutions]
    nnodes_per_block = [
        (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1) for nx, ny, nz in layer_resolutions
    ]

    with (mesh_out_dir / "mesh").open() as f:
        nelem = int(f.readline())
        assert nelem == sum(nelem_per_block)

    with (mesh_out_dir / "node_coords").open() as f:
        nnodes = int(f.readline())
        assert nnodes == sum(nnodes_per_block)

    materials = np.loadtxt(mesh_out_dir / "materials", dtype=int)
    values, counts = np.unique(materials[:, -1], return_counts=True)

    for val, count in zip(values, counts):
        assert nelem_per_block[val - 1] == count, (
            f"Material {val} should correspond to that block."
        )

    ## we don't really have a good test for this
    # with (mesh_out_dir / "nc_adjacencies").open() as f:
    #     ...
