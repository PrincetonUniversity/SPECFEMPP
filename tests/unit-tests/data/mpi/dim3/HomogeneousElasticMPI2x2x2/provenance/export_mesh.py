"""Convert cube.msh into the SPECFEM3D text MESH format consumed by xdecompose.

Run from this directory (``provenance/``) after create_mesh.py:

    uv run --group scripts python export_mesh.py
"""

import os
import sys
from export_gmsh3d import export2SPECFEM3D_gmsh, MaterialSpec

HERE = os.path.dirname(os.path.abspath(__file__))
## provenance/ -> .../HomogeneousElasticMPI2x2x2/dim3/mpi/data/unit-tests/tests/<repo>
REPO = os.path.abspath(os.path.join(HERE, *([".."] * 7)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

## Homogeneous elastic granite (matches nummaterial_velocity_file).
materials = {
    1: MaterialSpec(
        mat_id=1,
        domain_id=2,  ## 2 = elastic
        rho=2700.0,
        vp=6000.0,
        vs=3500.0,
        q_kappa=9999.0,
        q_mu=9999.0,
    ),
}

export2SPECFEM3D_gmsh(
    os.path.join(HERE, "cube.msh"),
    materials,
    outdir=os.path.join(HERE, "MESH"),
)
