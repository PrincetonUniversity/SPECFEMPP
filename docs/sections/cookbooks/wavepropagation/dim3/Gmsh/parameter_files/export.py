import sys

sys.path.insert(0, "<PATH TO SPECFEM++ DIRECTORY>/scripts")

from export_gmsh3d import export2SPECFEM3D_gmsh, MaterialSpec

## Material properties for material_1 (elastic granite)
materials = {
    1: MaterialSpec(
        mat_id=1,
        domain_id=2,  ## 2 = elastic
        rho=2700.0,  ## density (kg/m^3)
        vp=6000.0,  ## P-wave velocity (m/s)
        vs=3500.0,  ## S-wave velocity (m/s)
        q_kappa=9999.0,  ## no attenuation
        q_mu=9999.0,
    ),
}

export2SPECFEM3D_gmsh("halfspace.msh", materials, outdir="MESH")
