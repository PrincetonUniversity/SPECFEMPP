from pathlib import Path

thisdir = Path(__file__).parent


test_gmshlayerbuilder_expected_adjacencies = """
98
23 2 3 3
2 23 3 1
23 40 3 3
40 23 3 1
74 40 3 3
40 74 3 1
74 41 3 3
41 74 3 1
81 41 3 3
41 81 3 1
81 42 3 3
42 81 3 1
81 43 3 3
43 81 3 1
88 43 3 3
43 88 3 1
88 44 3 3
44 88 3 1
95 44 3 3
44 95 3 1
95 45 3 3
45 95 3 1
102 45 3 3
45 102 3 1
102 46 3 3
46 102 3 1
102 47 3 3
47 102 3 1
109 47 3 3
47 109 3 1
109 48 3 3
48 109 3 1
116 50 3 3
50 116 3 1
116 48 3 3
48 116 3 1
116 49 3 3
49 116 3 1
123 50 3 3
50 123 3 1
123 51 3 3
51 123 3 1
130 51 3 3
51 130 3 1
130 52 3 3
52 130 3 1
137 52 3 3
52 137 3 1
137 53 3 3
53 137 3 1
137 54 3 3
54 137 3 1
144 54 3 3
54 144 3 1
144 55 3 3
55 144 3 1
151 55 3 3
55 151 3 1
151 56 3 3
56 151 3 1
158 56 3 3
56 158 3 1
158 57 3 3
57 158 3 1
158 58 3 3
58 158 3 1
165 58 3 3
58 165 3 1
165 59 3 3
59 165 3 1
172 59 3 3
59 172 3 1
172 60 3 3
60 172 3 1
172 61 3 3
61 172 3 1
179 61 3 3
61 179 3 1
179 62 3 3
62 179 3 1
186 62 3 3
62 186 3 1
186 63 3 3
63 186 3 1
193 63 3 3
63 193 3 1
193 64 3 3
64 193 3 1
193 65 3 3
65 193 3 1
200 65 3 3
65 200 3 1
200 66 3 3
66 200 3 1
207 66 3 3
66 207 3 1
207 67 3 3
67 207 3 1
"""


def test_gmshlayerbuilder(path_scripts_root, execute_script, tmp_path):
    gmshlayerbuilder = path_scripts_root / "gmshlayerbuilder"
    topo_in_dir = thisdir / "basic_topo.dat"
    mesh_out_dir = tmp_path / "meshes" / "test_gmshlayerbuilder"
    cmd = [
        str(gmshlayerbuilder),
        "2d",
        "--top",
        "neumann",
        "--bottom",
        "neumann",
        "--left",
        "neumann",
        "--right",
        "neumann",
        str(topo_in_dir),
        str(mesh_out_dir),
    ]
    mesh_out_dir.mkdir(parents=True, exist_ok=True)
    execute_script(cmd, False)

    # verify correct mesh
    file = mesh_out_dir / "absorbing_surface"
    with file.open() as f:
        assert f.read().strip() == "0", (
            f"No absorbing surface expected. Got: {str(file)}"
        )
    file = mesh_out_dir / "free_surface"
    with file.open() as f:
        assert f.read().strip() == "0", f"No free surface expected. Got: {str(file)}"

    file = mesh_out_dir / "nc_adjacencies"
    # we may wish to have a more thorough test that does not require an exact recreation.
    with file.open() as f:
        assert f.read().strip() == test_gmshlayerbuilder_expected_adjacencies.strip(), (
            "Got different nonconforming adjacencies than expected."
        )
