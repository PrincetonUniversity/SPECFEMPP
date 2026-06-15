#!/usr/bin/env python
"""
export_gmsh3d.py -- Export a Gmsh 3D hexahedral mesh to SPECFEM++ / xdecompose_mesh format.

USAGE
-----
From the command line::

    python export_gmsh3d.py mesh.msh [output_dir]
    python export_gmsh3d.py mesh.msh MESH --mat 1 2 2700 6000 3500

From Python::

    from export_gmsh3d import export2SPECFEM3D_gmsh, MaterialSpec

    materials = {
        1: MaterialSpec(mat_id=1, domain_id=2, rho=2700, vp=6000, vs=3500),
    }
    export2SPECFEM3D_gmsh("halfspace.msh", materials, outdir="MESH")

PHYSICAL GROUP NAMING CONVENTIONS
----------------------------------
Volume groups (dim=3)
~~~~~~~~~~~~~~~~~~~~~
Name pattern: ``material_<id>``  e.g. ``material_1``, ``material_2``

These groups define which hexahedral elements belong to which material region.
Material properties (rho, vp, vs, ...) are passed separately via the *materials*
dict because Gmsh does not have an equivalent of CUBIT block attributes.

Surface groups (dim=2)
~~~~~~~~~~~~~~~~~~~~~~
The group *name* determines which output boundary file is written:

    ==================  ======================================
    Group name          Output file
    ==================  ======================================
    abs_xmin            absorbing_surface_file_xmin
    abs_xmax            absorbing_surface_file_xmax
    abs_ymin            absorbing_surface_file_ymin
    abs_ymax            absorbing_surface_file_ymax
    abs_bottom          absorbing_surface_file_bottom
    free_or_abs_zmax    free_or_absorbing_surface_file_zmax
    ==================  ======================================

Example Python API commands to assign groups::

    gmsh.model.addPhysicalGroup(3, [volume_tag], name="material_1")
    gmsh.model.addPhysicalGroup(2, [surface_tag], name="abs_xmin")

OUTPUT FILES
------------
    nodes_coords_file                       (always)
    mesh_file                               (always)
    materials_file                          (always)
    nummaterial_velocity_file               (if materials dict provided)
    absorbing_surface_file_xmin             (if 'abs_xmin' group present)
    absorbing_surface_file_xmax             (if 'abs_xmax' group present)
    absorbing_surface_file_ymin             (if 'abs_ymin' group present)
    absorbing_surface_file_ymax             (if 'abs_ymax' group present)
    absorbing_surface_file_bottom           (if 'abs_bottom' group present)
    free_or_absorbing_surface_file_zmax     (if 'free_or_abs_zmax' group present)

NODE AND ELEMENT NUMBERING
---------------------------
xdecompose_mesh uses element IDs and node IDs as direct array indices, so both
must be in the range [1, N].  Gmsh assigns element tags across all dimensions
(0-D, 1-D, 2-D, 3-D) so 3-D hex tags typically start after all surface elements
and can exceed the hex element count.  This script renumbers:

  - Hex elements to 1..nspec
  - Nodes to 1..nnodes (keeping only nodes used by hexes and boundary faces)

BOUNDARY ELEMENT IDs
--------------------
Each line in a boundary file must contain the **parent hex element ID** — the ID
of the 3-D hex element that owns that face — not the Gmsh surface element ID.
This script finds the parent hex for every boundary quad by matching their node
sets against the six faces of each hex element.

NODE ORDERING
-------------
Gmsh HEX8 and SPECFEM++ both use the same VTK/MSH convention::

    Node 0: (xmin,ymin,zmin)   Node 1: (xmax,ymin,zmin)
    Node 2: (xmax,ymax,zmin)   Node 3: (xmin,ymax,zmin)
    Node 4: (xmin,ymin,zmax)   Node 5: (xmax,ymin,zmax)
    Node 6: (xmax,ymax,zmax)   Node 7: (xmin,ymax,zmax)

No reordering is needed between the two.

FILE FORMAT REFERENCE
---------------------
The formats consumed by xdecompose_mesh are::

    nodes_coords_file   : <nnode>
                          <id> <x> <y> <z>
                          ...

    mesh_file           : <nelem>
                          <id> n1 n2 n3 n4 n5 n6 n7 n8    (HEX8)
                          ...

    materials_file      : <id> <material_id>    (no header)
                          ...

    nummaterial_velocity_file:
                          <domain_id> <mat_id> <rho> <vp> <vs> <Q_k> <Q_mu> <ani>
                          ...

    absorbing/free files: <count>
                          <hex_elem_id> n1 n2 n3 n4
                          ...

HEX8 node ordering (SPECFEM++ convention, from read_mesh_files.F90)::

         8 ------ 7
        /|      / |       z
       5 ----- 6  |       |
       | |     |  |       |-------> y
       | 4 ----|- 3        \\
       |/      |/           x
       1 ----- 2

    Bottom face (z-): nodes 1, 2, 3, 4
    Top    face (z+): nodes 5, 6, 7, 8
"""
from __future__ import print_function
import os
import sys

try:
    import gmsh
except ImportError:
    print("error: could not import gmsh -- install it with: pip install gmsh")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Node ordering: Gmsh HEX8 (VTK/MSH) and SPECFEM++ use the same convention:
#
#   Node 0: (xmin, ymin, zmin)   Node 1: (xmax, ymin, zmin)
#   Node 2: (xmax, ymax, zmin)   Node 3: (xmin, ymax, zmin)
#   Node 4: (xmin, ymin, zmax)   Node 5: (xmax, ymin, zmax)
#   Node 6: (xmax, ymax, zmax)   Node 7: (xmin, ymax, zmax)
#
# No reordering is needed.  GMSH_HEX8_TO_SPECFEM is kept as the identity so
# that the code structure is explicit.
# ---------------------------------------------------------------------------

GMSH_HEX8_TO_SPECFEM = [0, 1, 2, 3, 4, 5, 6, 7]

# Gmsh element type IDs (from Gmsh MSH format specification)
_GMSH_HEX8  = 5    # 8-node hexahedron
_GMSH_QUAD4 = 3    # 4-node quadrilateral

# Face node indices within a SPECFEM-ordered HEX8 (0-indexed):
#   Each tuple names 4 of the 8 nodes that form one face.
_HEX8_FACE_INDICES = (
    (0, 1, 2, 3),  # bottom (z-)
    (4, 5, 6, 7),  # top    (z+)
    (0, 1, 5, 4),  # front  (x-)
    (1, 2, 6, 5),  # right  (y+)
    (2, 3, 7, 6),  # back   (x+)
    (3, 0, 4, 7),  # left   (y-)
)

# ---------------------------------------------------------------------------
# Routing table: physical group name -> output filename
# ---------------------------------------------------------------------------

BOUNDARY_NAME_TO_FILE = {
    "abs_xmin":         "absorbing_surface_file_xmin",
    "abs_xmax":         "absorbing_surface_file_xmax",
    "abs_ymin":         "absorbing_surface_file_ymin",
    "abs_ymax":         "absorbing_surface_file_ymax",
    "abs_bottom":       "absorbing_surface_file_bottom",
    "free_or_abs_zmax": "free_or_absorbing_surface_file_zmax",
}


# ---------------------------------------------------------------------------
# MaterialSpec -- material properties for one material region
# ---------------------------------------------------------------------------

class MaterialSpec:
    """
    Properties for one material region.

    Parameters
    ----------
    mat_id    : int   -- positive integer matching the suffix in 'material_<id>'
    domain_id : int   -- 1 = acoustic, 2 = elastic
    rho       : float -- density (kg/m^3)
    vp        : float -- P-wave velocity (m/s)
    vs        : float -- S-wave velocity (m/s); use 0 for acoustic
    q_kappa   : float -- quality factor; 9999 = no Q_kappa attenuation
    q_mu      : float -- quality factor; 9999 = no Q_mu attenuation
    anisotropy: int   -- 0 = isotropic
    """

    def __init__(self, mat_id, domain_id, rho, vp, vs,
                 q_kappa=9999.0, q_mu=9999.0, anisotropy=0):
        self.mat_id     = mat_id
        self.domain_id  = domain_id
        self.rho        = rho
        self.vp         = vp
        self.vs         = vs
        self.q_kappa    = q_kappa
        self.q_mu       = q_mu
        self.anisotropy = anisotropy

    def nummaterial_line(self):
        """Return a formatted line for nummaterial_velocity_file."""
        return (
            "{:d}  {:d}  {:16.6f}  {:16.6f}  {:16.6f}"
            "  {:16.6f}  {:16.6f}  {:d}\n"
        ).format(
            self.domain_id, self.mat_id,
            self.rho, self.vp, self.vs,
            self.q_kappa, self.q_mu, self.anisotropy,
        )


# ---------------------------------------------------------------------------
# MeshWriter3D -- main exporter class
# ---------------------------------------------------------------------------

class MeshWriter3D:
    """
    Reads a Gmsh .msh file and writes the text files expected by xdecompose_mesh.

    The .msh file must contain physical groups whose names follow the
    convention documented in this module's docstring.

    **ID renumbering** (important):
    Gmsh numbers elements across ALL dimensions so 3-D hex tags can exceed
    nspec.  This class renumbers hex elements 1..nspec and nodes 1..nnodes
    before writing, and resolves boundary face IDs to their parent hex IDs.

    Parameters
    ----------
    msh_path  : str
        Path to the .msh file produced by Gmsh.
    materials : dict[int, MaterialSpec], optional
        Maps material_id -> MaterialSpec.  If omitted, nummaterial_velocity_file
        is not written and must be created manually.
    """

    def __init__(self, msh_path, materials=None):
        self.msh_path  = msh_path
        self.materials = materials or {}
        # Populated by _collect_all():
        self._node_coords = {}   # node_id (int, 1-indexed) -> (x, y, z)
        self._elements    = []   # [(elem_id, mat_id, [node_ids x8]), ...]  1-indexed IDs
        self._boundaries  = {}   # group_name -> [(hex_elem_id, [node_ids x4]), ...]
        self._collect_all()

    # ------------------------------------------------------------------
    # Mesh data collection
    # ------------------------------------------------------------------

    def _collect_all(self):
        """
        Open .msh, read raw Gmsh data, renumber IDs, resolve boundary parents,
        then close Gmsh.  After this method returns, no Gmsh context is needed.
        """
        print("Reading {} ...".format(self.msh_path))
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(self.msh_path)

        # -- Raw node coordinates (keyed by Gmsh node tag) --
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        raw_coords = {}
        for i, tag in enumerate(node_tags):
            raw_coords[int(tag)] = (coords[3*i], coords[3*i+1], coords[3*i+2])

        # -- Raw hex elements: [(gmsh_elem_tag, mat_id, [8 node gmsh_tags]), ...] --
        raw_hexes = []

        # -- Raw boundary quads: group_name -> [[4 node gmsh_tags], ...] --
        raw_quads = {}

        for dim, pg_tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, pg_tag).strip()

            if dim == 3 and name.lower().startswith("material_"):
                self._read_hex_group(name, pg_tag, raw_hexes)

            elif dim == 2 and name.lower() in BOUNDARY_NAME_TO_FILE:
                self._read_quad_group(name, pg_tag, raw_quads)

            else:
                print("  [skip] physical group dim={} '{}': not recognised.".format(
                    dim, name))

        gmsh.finalize()

        if not raw_hexes:
            raise RuntimeError(
                "No hex elements found.  Make sure the .msh file contains at least "
                "one dim=3 physical group named 'material_<id>'."
            )

        # Post-processing (no Gmsh context needed beyond this point)
        self._renumber_and_resolve(raw_coords, raw_hexes, raw_quads)

        print("  {:d} nodes, {:d} hex elements, {:d} boundary group(s) loaded.".format(
            len(self._node_coords), len(self._elements), len(self._boundaries)))

    def _read_hex_group(self, name, pg_tag, raw_hexes):
        """Append HEX8 elements from one volume physical group to *raw_hexes*."""
        try:
            mat_id = int(name.strip().lower()[len("material_"):])
        except ValueError:
            print("  [warn] cannot parse material id from '{}'. Skipping.".format(name))
            return

        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(3, pg_tag)
        n_added = 0
        for etag in entity_tags:
            types, tag_lists, node_lists = gmsh.model.mesh.getElements(3, etag)
            for etype, etags, ntags in zip(types, tag_lists, node_lists):
                if etype != _GMSH_HEX8:
                    print("  [skip] element type {} in '{}': "
                          "only HEX8 (type {:d}) is supported.".format(
                              etype, name, _GMSH_HEX8))
                    continue
                n_per = 8
                for i, eid in enumerate(etags):
                    gmsh_nodes = [int(ntags[i*n_per + j]) for j in range(n_per)]
                    specfem_nodes = [gmsh_nodes[k] for k in GMSH_HEX8_TO_SPECFEM]
                    raw_hexes.append((int(eid), mat_id, specfem_nodes))
                    n_added += 1

        print("  material group '{}' (id={:d}) -> {:d} hex elements.".format(
            name, mat_id, n_added))

    def _read_quad_group(self, name, pg_tag, raw_quads):
        """Append QUAD4 node lists from one surface physical group to *raw_quads*."""
        key = name.strip().lower()
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(2, pg_tag)
        node_lists_for_group = []
        for etag in entity_tags:
            types, tag_lists, node_lists = gmsh.model.mesh.getElements(2, etag)
            for etype, etags, ntags in zip(types, tag_lists, node_lists):
                if etype != _GMSH_QUAD4:
                    print("  [skip] element type {} in '{}': "
                          "only QUAD4 (type {:d}) is supported.".format(
                              etype, name, _GMSH_QUAD4))
                    continue
                n_per = 4
                for i, _eid in enumerate(etags):
                    nodes = [int(ntags[i*n_per + j]) for j in range(n_per)]
                    node_lists_for_group.append(nodes)

        raw_quads[key] = node_lists_for_group
        print("  boundary group '{}' -> {:d} quad faces.".format(
            key, len(node_lists_for_group)))

    def _renumber_and_resolve(self, raw_coords, raw_hexes, raw_quads):
        """
        Renumber hex elements 1..nspec, renumber nodes 1..nnodes, and resolve
        boundary face node-sets to their parent hex element IDs.

        xdecompose_mesh uses element IDs and node IDs as direct array indices,
        so both must be in [1, N].  Gmsh numbers elements across all dimensions
        (0-D, 1-D, 2-D, 3-D), so 3-D hex tags usually start after all lower-
        dimensional elements and can exceed nspec.

        Boundary files must contain the **parent hex element ID** (the 3-D hex
        that owns the face), not the Gmsh surface element ID.
        """

        # -- Renumber hex elements 1..nspec --
        self._elements = [
            (new_id, mat_id, nodes)
            for new_id, (_, mat_id, nodes) in enumerate(raw_hexes, start=1)
        ]

        # -- Build face-node-set -> hex ID lookup --
        # Key: frozenset of the 4 Gmsh node tags on one hex face (using Gmsh tags
        # still, before node renumbering, because raw_quads also uses Gmsh tags).
        face_to_hex = {}
        for hex_id, _mat_id, nodes in self._elements:
            for fi in _HEX8_FACE_INDICES:
                key = frozenset(nodes[i] for i in fi)
                face_to_hex[key] = hex_id

        # -- Resolve boundary quads to parent hex IDs --
        for bname, quad_node_lists in raw_quads.items():
            resolved = []
            missing  = 0
            for quad_nodes in quad_node_lists:
                hex_id = face_to_hex.get(frozenset(quad_nodes))
                if hex_id is None:
                    missing += 1
                else:
                    resolved.append((hex_id, quad_nodes))
            if missing:
                print("  [warn] {:d} faces in '{}' had no matching hex -- "
                      "check that hex and boundary physical groups share the "
                      "same mesh nodes.".format(missing, bname))
            self._boundaries[bname] = resolved
            print("  boundary '{}' -> {:d} faces resolved to parent hex IDs -> {}.".format(
                bname, len(resolved), BOUNDARY_NAME_TO_FILE[bname]))

        # -- Collect only nodes used by hexes and boundary faces --
        used = set()
        for _, _, nodes in self._elements:
            used.update(nodes)
        for quads in self._boundaries.values():
            for _, nodes in quads:
                used.update(nodes)

        # -- Renumber nodes 1..nnodes --
        node_remap = {old: new for new, old in enumerate(sorted(used), start=1)}

        self._node_coords = {
            node_remap[old]: raw_coords[old]
            for old in used
        }
        self._elements = [
            (eid, mat_id, [node_remap[n] for n in nodes])
            for eid, mat_id, nodes in self._elements
        ]
        self._boundaries = {
            bname: [(eid, [node_remap[n] for n in nodes]) for eid, nodes in quads]
            for bname, quads in self._boundaries.items()
        }

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def write_nodes_coords(self, filepath):
        """
        Write nodes_coords_file.

        Format::

            <nnode>
            <id> <x> <y> <z>
            ...
        """
        print("Writing {} ...".format(filepath))
        node_ids = sorted(self._node_coords.keys())
        with open(filepath, "w") as f:
            f.write("{}\n".format(len(node_ids)))
            for nid in node_ids:
                x, y, z = self._node_coords[nid]
                f.write("{} {:.6f} {:.6f} {:.6f}\n".format(nid, x, y, z))
        print("  {:d} nodes written.".format(len(node_ids)))

    def write_mesh(self, filepath):
        """
        Write mesh_file (element connectivity).

        Format::

            <nelem>
            <id> n1 n2 n3 n4 n5 n6 n7 n8
            ...
        """
        print("Writing {} ...".format(filepath))
        with open(filepath, "w") as f:
            f.write("{}\n".format(len(self._elements)))
            for eid, _mat_id, nodes in self._elements:
                f.write("{} {}\n".format(eid, " ".join(str(n) for n in nodes)))
        print("  {:d} elements written.".format(len(self._elements)))

    def write_materials(self, filepath):
        """
        Write materials_file (element-to-material mapping, no header).

        Format::

            <hex_id> <material_id>
            ...
        """
        print("Writing {} ...".format(filepath))
        with open(filepath, "w") as f:
            for eid, mat_id, _nodes in self._elements:
                f.write("{} {}\n".format(eid, mat_id))
        print("  {:d} element-material mappings written.".format(len(self._elements)))

    def write_nummaterial(self, filepath):
        """
        Write nummaterial_velocity_file from the materials dict.

        Format::

            <domain_id> <mat_id> <rho> <vp> <vs> <Q_kappa> <Q_mu> <anisotropy>
            ...
        """
        if not self.materials:
            print("  [skip] no materials dict supplied; "
                  "nummaterial_velocity_file not written.  Write it manually.")
            return
        print("Writing {} ...".format(filepath))
        lines = [self.materials[mid].nummaterial_line()
                 for mid in sorted(self.materials.keys())]
        with open(filepath, "w") as f:
            f.writelines(lines)
        print("  {:d} material definition(s) written.".format(len(lines)))

    def write_boundary(self, name, filepath):
        """
        Write one absorbing or free-surface boundary file.

        Format::

            <n_faces>
            <hex_elem_id> n1 n2 n3 n4
            ...
        """
        print("Writing {} ...".format(filepath))
        quads = self._boundaries.get(name, [])
        with open(filepath, "w") as f:
            f.write("{:10d}\n".format(len(quads)))
            for eid, nodes in quads:
                f.write("{} {}\n".format(eid, " ".join(str(n) for n in nodes)))
        print("  {:d} boundary faces written.".format(len(quads)))

    def write_all(self, outdir="."):
        """
        Write all mesh files to *outdir*, creating it if necessary.

        Always writes:   nodes_coords_file, mesh_file, materials_file
        Conditionally:   nummaterial_velocity_file (if materials dict provided)
                         absorbing/free-surface files (if named groups present)
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        base = outdir.rstrip("/\\") + os.sep

        self.write_nodes_coords(base + "nodes_coords_file")
        self.write_mesh(base + "mesh_file")
        self.write_materials(base + "materials_file")
        self.write_nummaterial(base + "nummaterial_velocity_file")

        for bname in BOUNDARY_NAME_TO_FILE:
            if bname in self._boundaries:
                self.write_boundary(bname, base + BOUNDARY_NAME_TO_FILE[bname])
            else:
                print("  [skip] {} (no physical group named '{}').".format(
                    BOUNDARY_NAME_TO_FILE[bname], bname))

        print("\nDone.  Mesh files written to: {}".format(os.path.abspath(outdir)))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export2SPECFEM3D_gmsh(msh_path, materials=None, outdir="."):
    """
    Export a Gmsh .msh file to SPECFEM++ / xdecompose_mesh format.

    Parameters
    ----------
    msh_path  : str
        Path to the .msh file produced by Gmsh.
    materials : dict[int, MaterialSpec], optional
        Maps material_id (matching the suffix in physical group 'material_<id>')
        to a MaterialSpec instance.  If omitted, nummaterial_velocity_file is
        not written and must be created manually.
    outdir : str
        Output directory for the mesh text files.  Created if it does not exist.
        Default: current working directory.

    Example
    -------
    ::

        from export_gmsh3d import export2SPECFEM3D_gmsh, MaterialSpec

        materials = {
            1: MaterialSpec(mat_id=1, domain_id=2, rho=2700, vp=6000, vs=3500),
        }
        export2SPECFEM3D_gmsh("halfspace.msh", materials, outdir="MESH")
    """
    print("# BEGIN SPECFEM3D export (Gmsh) ...")
    writer = MeshWriter3D(msh_path, materials)
    writer.write_all(outdir)
    print("# END SPECFEM3D export.")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a Gmsh .msh file to SPECFEM3D / xdecompose_mesh text files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export_gmsh3d.py halfspace.msh MESH\n"
            "  python export_gmsh3d.py halfspace.msh MESH --mat 1 2 2700 6000 3500\n"
            "  python export_gmsh3d.py two_layer.msh MESH \\\n"
            "      --mat 1 2 2700 6000 3500 \\\n"
            "      --mat 2 2 2300 3500 2000\n"
        ),
    )
    parser.add_argument("msh_file",
                        help="Input .msh file from Gmsh")
    parser.add_argument("outdir", nargs="?", default=".",
                        help="Output directory (default: current directory)")
    parser.add_argument(
        "--mat", nargs="+", action="append",
        metavar="VALUE",
        help=(
            "Material: ID DOMAIN_ID RHO VP VS [QKAPPA QMU].  "
            "Repeat for multiple materials.  "
            "DOMAIN_ID: 1=acoustic, 2=elastic."
        ),
    )
    args = parser.parse_args()

    materials = {}
    if args.mat:
        for m in args.mat:
            if len(m) < 5:
                parser.error("--mat requires at least ID DOMAIN_ID RHO VP VS")
            mat_id    = int(m[0])
            domain_id = int(m[1])
            rho       = float(m[2])
            vp        = float(m[3])
            vs        = float(m[4])
            q_kappa   = float(m[5]) if len(m) > 5 else 9999.0
            q_mu      = float(m[6]) if len(m) > 6 else 9999.0
            materials[mat_id] = MaterialSpec(mat_id, domain_id, rho, vp, vs,
                                             q_kappa, q_mu)

    export2SPECFEM3D_gmsh(args.msh_file, materials, args.outdir)


if __name__ == "__main__":
    _cli()
