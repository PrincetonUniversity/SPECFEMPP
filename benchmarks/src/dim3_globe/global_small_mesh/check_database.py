#!/usr/bin/env python3
"""Validate the thin SPECFEM++ mesh database written by xmeshfem3D_globe.

There is no C++ reader for this format yet (issue #1995), so this script is the
verification harness for the writer added in issue #2000. It parses the Fortran
sequential-unformatted records and asserts the invariants the future reader will
depend on.

Usage:
    python3 check_database.py [DATABASES_MPI_DIR]
"""

import glob
import os
import struct
import sys

NGNOD = 27
MAGIC = "SPECFEMPP_GLOBE_DB"
VERSION = 1

REGION_CRUST_MANTLE = 1
REGION_OUTER_CORE = 2
REGION_INNER_CORE = 3

MEDIUM_ACOUSTIC = 1
MEDIUM_ELASTIC = 2

FACE_BOTTOM = 1
FACE_TOP = 3


class Failure(Exception):
    pass


class RecordReader:
    """Reads Fortran sequential-unformatted records, checking both length markers."""

    def __init__(self, path):
        with open(path, "rb") as handle:
            self.data = handle.read()
        self.path = path
        self.pos = 0
        self.nrecords = 0

    def record(self):
        if self.pos + 4 > len(self.data):
            raise Failure(f"{self.path}: truncated at record {self.nrecords}")
        (head,) = struct.unpack_from("<i", self.data, self.pos)
        start = self.pos + 4
        end = start + head
        if end + 4 > len(self.data):
            raise Failure(
                f"{self.path}: record {self.nrecords} claims {head} bytes but the file ends first"
            )
        (tail,) = struct.unpack_from("<i", self.data, end)
        if tail != head:
            raise Failure(
                f"{self.path}: record {self.nrecords} markers disagree ({head} vs {tail})"
            )
        self.pos = end + 4
        self.nrecords += 1
        return self.data[start:end]

    def at_eof(self):
        return self.pos == len(self.data)


def unpack(blob, fmt_counts):
    """Unpack a record described as a list of (format char, count) pairs.

    Raises if the record is not consumed exactly -- the same strictness the C++
    fortran_io reader applies.
    """
    sizes = {"i": 4, "d": 8, "l": 4}
    out = []
    off = 0
    for code, count in fmt_counts:
        if code == "c":
            out.append(blob[off : off + count].decode("ascii"))
            off += count
            continue
        width = sizes[code]
        raw = struct.unpack_from(
            "<" + ("i" if code == "l" else code) * count, blob, off
        )
        off += width * count
        if code == "l":
            raw = tuple(bool(v) for v in raw)
        out.append(list(raw) if count > 1 else raw[0])
    if off != len(blob):
        raise Failure(f"record not consumed exactly: used {off} of {len(blob)} bytes")
    return out


def read_database(path):
    reader = RecordReader(path)
    db = {"path": path}

    magic, version = unpack(reader.record(), [("c", 32), ("i", 1)])
    db["magic"] = magic.strip()
    db["version"] = version

    if db["magic"] != MAGIC:
        raise Failure(f"{path}: bad magic {db['magic']!r}")
    if version != VERSION:
        raise Failure(f"{path}: unsupported format_version {version}")

    db["planet_type"], db["r_planet"], db["rhoav"] = unpack(
        reader.record(), [("i", 1), ("d", 1), ("d", 1)]
    )

    header = unpack(reader.record(), [("i", 5)])[0]
    db["ngnod"], db["ngllx"], db["nglly"], db["ngllz"], db["nregions"] = header

    flags = unpack(reader.record(), [("l", 8)])[0]
    (
        db["ellipticity"],
        db["topography"],
        db["gravity"],
        db["full_gravity"],
        db["rotation"],
        db["attenuation"],
        db["oceans"],
        db["has_reference_geometry"],
    ) = flags

    db["material_mode"] = unpack(reader.record(), [("i", 1)])[0]

    db["model"] = unpack(reader.record(), [("c", 512)])[0].strip()

    blob = reader.record()
    (n_codes,) = struct.unpack_from("<i", blob, 0)
    db["codes"] = unpack(blob, [("i", 1), ("i", n_codes)])[1]

    blob = reader.record()
    (n_flags,) = struct.unpack_from("<i", blob, 0)
    db["model_flags"] = unpack(blob, [("i", 1), ("l", n_flags)])[1]

    nnode = unpack(reader.record(), [("i", 1)])[0]
    db["nnode"] = nnode
    coords = unpack(reader.record(), [("d", 3 * nnode)])[0]
    db["x"] = coords[0:nnode]
    db["y"] = coords[nnode : 2 * nnode]
    db["z"] = coords[2 * nnode : 3 * nnode]

    if db["has_reference_geometry"]:
        coords = unpack(reader.record(), [("d", 3 * nnode)])[0]
        db["xref"] = coords[0:nnode]
        db["yref"] = coords[nnode : 2 * nnode]
        db["zref"] = coords[2 * nnode : 3 * nnode]
    else:
        db["xref"] = None

    nspec = unpack(reader.record(), [("i", 1)])[0]
    db["nspec"] = nspec

    tags = unpack(reader.record(), [("i", 4 * nspec)])[0]
    db["region"] = tags[0:nspec]
    db["medium_tag"] = tags[nspec : 2 * nspec]
    db["property_tag"] = tags[2 * nspec : 3 * nspec]
    db["idoubling"] = tags[3 * nspec : 4 * nspec]

    radii = unpack(reader.record(), [("d", 2 * nspec)])[0]
    db["rmin"] = radii[0:nspec]
    db["rmax"] = radii[nspec : 2 * nspec]

    db["elem_in_crust"] = unpack(reader.record(), [("l", nspec)])[0]

    # node_ids is written as a (NGNOD,nspec) Fortran array, so it arrives
    # element-major with the anchor index varying fastest
    ids = unpack(reader.record(), [("i", NGNOD * nspec)])[0]
    db["node_ids"] = [ids[i * NGNOD : (i + 1) * NGNOD] for i in range(nspec)]

    for name in ("free", "cmb", "icb", "ocean"):
        nfaces = unpack(reader.record(), [("i", 1)])[0]
        if nfaces > 0:
            pair = unpack(reader.record(), [("i", 2 * nfaces)])[0]
            db[name + "_ispec"] = pair[0:nfaces]
            db[name + "_face"] = pair[nfaces : 2 * nfaces]
        else:
            db[name + "_ispec"] = []
            db[name + "_face"] = []

    nb_adj = unpack(reader.record(), [("i", 1)])[0]
    db["nb_adj_edges"] = nb_adj
    db["xadj"] = unpack(reader.record(), [("i", nspec + 1)])[0]
    db["adjncy"] = unpack(reader.record(), [("i", nb_adj)])[0]
    db["adj_type"] = unpack(reader.record(), [("i", nb_adj)])[0]

    num_neighbors = unpack(reader.record(), [("i", 1)])[0]
    db["neighbors"] = []
    for _ in range(num_neighbors):
        rank, count = unpack(reader.record(), [("i", 1), ("i", 1)])
        nodes = unpack(reader.record(), [("i", count)])[0] if count > 0 else []
        if count == 1:
            nodes = [nodes] if not isinstance(nodes, list) else nodes
        db["neighbors"].append({"rank": rank, "count": count, "nodes": nodes})

    if not reader.at_eof():
        raise Failure(
            f"{path}: {len(reader.data) - reader.pos} trailing bytes after the last record"
        )

    db["nrecords"] = reader.nrecords
    return db


def check_one(db, problems):
    path = os.path.basename(db["path"])

    def bad(msg):
        problems.append(f"{path}: {msg}")

    if db["ngnod"] != NGNOD:
        bad(f"NGNOD is {db['ngnod']}, expected {NGNOD}")
    if not (db["ngllx"] == db["nglly"] == db["ngllz"]):
        bad(f"non-cubic GLL grid {db['ngllx']}x{db['nglly']}x{db['ngllz']}")
    if db["material_mode"] != 1:
        bad(f"material_mode is {db['material_mode']}, expected 1 (ORACLE)")
    if not 1 <= db["nregions"] <= 3:
        bad(f"nregions is {db['nregions']}")

    nspec, nnode = db["nspec"], db["nnode"]

    # node ids
    flat = [nid for elem in db["node_ids"] for nid in elem]
    if min(flat) < 1 or max(flat) > nnode:
        bad(f"node id out of range [1,{nnode}]: min {min(flat)} max {max(flat)}")
    if len(set(flat)) != nnode:
        bad(f"{nnode - len(set(flat))} nodes are never referenced by an element")
    if nnode > NGNOD * nspec:
        bad(f"nnode {nnode} exceeds {NGNOD}*nspec {NGNOD * nspec}")

    # dimensionalization: the outermost node must sit on R_PLANET. Ellipticity and
    # topography deform the surface by up to a percent or so, hence the wider band.
    r_planet = db["r_planet"]
    deformed = db["has_reference_geometry"]
    tolerance = r_planet * 1e-2 if deformed else max(1.0, r_planet * 1e-6)
    rmax_node = max(
        (x * x + y * y + z * z) ** 0.5 for x, y, z in zip(db["x"], db["y"], db["z"])
    )
    if abs(rmax_node - r_planet) > tolerance:
        bad(
            f"largest node radius {rmax_node:.3f} is not R_PLANET {r_planet:.3f}"
            " -- coordinates may not be dimensionalized"
        )

    # the reference geometry is the undeformed sphere, so it must not exceed R_PLANET
    if db["xref"] is not None:
        rmax_ref = max(
            (x * x + y * y + z * z) ** 0.5
            for x, y, z in zip(db["xref"], db["yref"], db["zref"])
        )
        if rmax_ref > r_planet * (1.0 + 1e-6):
            bad(f"reference node radius {rmax_ref:.1f} exceeds R_PLANET {r_planet:.1f}")
        if (db["topography"] or db["ellipticity"]) and db["xref"] == db["x"]:
            bad("NODES_REFERENCE is identical to NODES although the mesh is deformed")
    elif db["has_reference_geometry"]:
        bad("HAS_REFERENCE_GEOMETRY is on but NODES_REFERENCE was not written")

    if (db["topography"] or db["ellipticity"]) and not db["has_reference_geometry"]:
        bad("deformation is enabled but HAS_REFERENCE_GEOMETRY is off")

    # per-element context
    for i in range(nspec):
        if db["region"][i] not in (
            REGION_CRUST_MANTLE,
            REGION_OUTER_CORE,
            REGION_INNER_CORE,
        ):
            bad(f"element {i + 1} has region {db['region'][i]}")
            break
    for i in range(nspec):
        expected = (
            MEDIUM_ACOUSTIC if db["region"][i] == REGION_OUTER_CORE else MEDIUM_ELASTIC
        )
        if db["medium_tag"][i] != expected:
            bad(
                f"element {i + 1} in region {db['region'][i]} has medium_tag"
                f" {db['medium_tag'][i]}, expected {expected}"
            )
            break
    for i in range(nspec):
        if db["rmin"][i] >= db["rmax"][i]:
            bad(f"element {i + 1} has rmin {db['rmin'][i]} >= rmax {db['rmax'][i]}")
            break

    # CSR adjacency: shape, types, symmetry
    xadj, adjncy, adj_type = db["xadj"], db["adjncy"], db["adj_type"]
    if len(xadj) != nspec + 1:
        bad(f"xadj has {len(xadj)} entries, expected nspec+1 = {nspec + 1}")
    if xadj[0] != 1 or xadj[-1] != db["nb_adj_edges"] + 1:
        bad(f"xadj endpoints {xadj[0]}..{xadj[-1]} inconsistent with nb_adj_edges")
    if adj_type and (min(adj_type) < 1 or max(adj_type) > 26):
        bad(f"adj_type out of [1,26]: min {min(adj_type)} max {max(adj_type)}")

    neighbors = [set() for _ in range(nspec + 1)]
    for i in range(1, nspec + 1):
        for k in range(xadj[i - 1], xadj[i]):
            neighbors[i].add(adjncy[k - 1])
    asymmetric = 0
    for i in range(1, nspec + 1):
        for j in neighbors[i]:
            if i not in neighbors[j]:
                asymmetric += 1
    if asymmetric:
        bad(f"{asymmetric} adjacency edges are not symmetric")

    # the CMB and ICB must show up as adjacency edges, i.e. the cross-region nodes
    # were actually welded by get_global
    cross = {}
    for i in range(1, nspec + 1):
        for j in neighbors[i]:
            ri, rj = db["region"][i - 1], db["region"][j - 1]
            if ri != rj:
                cross[(min(ri, rj), max(ri, rj))] = (
                    cross.get((min(ri, rj), max(ri, rj)), 0) + 1
                )
    if db["nregions"] >= 2 and cross.get((1, 2), 0) == 0:
        bad("no crust/mantle <-> outer core adjacency: the CMB nodes were not welded")
    if db["nregions"] >= 3 and cross.get((2, 3), 0) == 0:
        bad("no outer core <-> inner core adjacency: the ICB nodes were not welded")

    # boundary faces
    for name, face in (("free", FACE_TOP), ("cmb", None), ("icb", None)):
        for ispec, fid in zip(db[name + "_ispec"], db[name + "_face"]):
            if not 1 <= ispec <= nspec:
                bad(f"{name} boundary references element {ispec}")
                break
            if fid not in (FACE_BOTTOM, FACE_TOP):
                bad(f"{name} boundary has face_id {fid}")
                break
            if face is not None and fid != face:
                bad(f"{name} boundary has face_id {fid}, expected {face}")
                break

    # the CMB and ICB are each seen from both sides, so both face sets must be present
    # and of equal size
    for name, lower, upper in (("cmb", 1, 2), ("icb", 2, 3)):
        faces = db[name + "_face"]
        if not faces:
            continue
        n_bottom = sum(1 for f in faces if f == FACE_BOTTOM)
        n_top = sum(1 for f in faces if f == FACE_TOP)
        if n_bottom != n_top:
            bad(f"{name} has {n_bottom} bottom faces but {n_top} top faces")

    if db["oceans"] and len(db["ocean_ispec"]) != len(db["free_ispec"]):
        bad("OCEANS is on but the ocean-load block does not match the free surface")
    if not db["oceans"] and db["ocean_ispec"]:
        bad("OCEANS is off but the ocean-load block is not empty")

    return db


def check_cross_rank(dbs, problems):
    """The two ranks of every neighbor pair must share the same nodes, in the same order."""
    by_rank = {}
    for path, db in dbs.items():
        rank = int(os.path.basename(path)[4:10])
        by_rank[rank] = db

    for rank, db in sorted(by_rank.items()):
        for entry in db["neighbors"]:
            other = entry["rank"]
            if other not in by_rank:
                problems.append(f"rank {rank}: neighbor {other} has no database file")
                continue
            back = [e for e in by_rank[other]["neighbors"] if e["rank"] == rank]
            if not back:
                problems.append(
                    f"rank {rank} lists {other} but not the other way round"
                )
                continue
            if back[0]["count"] != entry["count"]:
                problems.append(
                    f"ranks {rank}<->{other} disagree on the shared node count:"
                    f" {entry['count']} vs {back[0]['count']}"
                )
                continue
            # same physical points, in the same order
            here = by_rank[rank]
            there = by_rank[other]
            for k, (na, nb) in enumerate(zip(entry["nodes"], back[0]["nodes"])):
                dx = here["x"][na - 1] - there["x"][nb - 1]
                dy = here["y"][na - 1] - there["y"][nb - 1]
                dz = here["z"][na - 1] - there["z"][nb - 1]
                if (dx * dx + dy * dy + dz * dz) ** 0.5 > 1.0:
                    problems.append(
                        f"ranks {rank}<->{other} shared node {k} differs by"
                        f" {(dx * dx + dy * dy + dz * dz) ** 0.5:.3f} m"
                        " -- the anchor filter produced different orderings"
                    )
                    break


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "DATABASES_MPI"
    paths = sorted(glob.glob(os.path.join(directory, "proc*_specfempp_database.bin")))
    if not paths:
        print(f"no proc*_specfempp_database.bin found in {directory}", file=sys.stderr)
        return 1

    problems = []
    dbs = {}
    for path in paths:
        try:
            db = read_database(path)
        except Failure as exc:
            problems.append(str(exc))
            continue
        dbs[path] = db
        check_one(db, problems)

    if len(dbs) == len(paths):
        check_cross_rank(dbs, problems)

    for path in paths:
        db = dbs.get(path)
        if db is None:
            continue
        print(
            f"{os.path.basename(path)}: {db['nspec']} elements, {db['nnode']} nodes,"
            f" {db['nb_adj_edges']} adjacency edges,"
            f" {len(db['free_ispec'])}/{len(db['cmb_ispec'])}/{len(db['icb_ispec'])}"
            f" free/CMB/ICB faces, {len(db['neighbors'])} MPI neighbors,"
            f" {db['nrecords']} records"
        )

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
