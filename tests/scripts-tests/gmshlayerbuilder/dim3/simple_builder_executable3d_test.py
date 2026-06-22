from pathlib import Path

thisdir = Path(__file__).parent


def test_gmshlayerbuilder3d(path_scripts_root, execute_script, tmp_path):
    gmshlayerbuilder = path_scripts_root / "gmshlayerbuilder"
    topo_in_dir = thisdir / "trialmesh" / "interfaces.txt"
    mesh_out_dir = tmp_path / "meshes" / "test_gmshlayerbuilder3d"
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

    # file = mesh_out_dir / "nc_adjacencies"
    # # we may wish to have a more thorough test that does not require an exact recreation.
    # with file.open() as f:
    #     assert f.read().strip() == test_gmshlayerbuilder_expected_adjacencies.strip(), (
    #         "Got different nonconforming adjacencies than expected."
    #     )
