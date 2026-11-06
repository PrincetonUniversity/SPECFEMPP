"""Dump corner GLL coordinates using the original globe Fortran.

From GlobalSmallMesh run:
python3 provenance/dump_spherical_coordinates.py /path/to/specfem3d_globe
Requires gfortran; no Python packages are needed.
"""

import pathlib
import re
import struct
import subprocess
import sys
import tempfile


def records(path):
    data = path.read_bytes()
    offset = 0
    result = []
    while offset < len(data):
        size = struct.unpack_from("<i", data, offset)[0]
        assert struct.unpack_from("<i", data, offset + 4 + size)[0] == size
        result.append(data[offset + 4 : offset + 4 + size])
        offset += size + 8
    return result


def subroutine(path, name):
    return re.search(
        rf"^  subroutine {name}\(.*?^  end subroutine {name}$",
        path.read_text(),
        re.MULTILINE | re.DOTALL,
    ).group(0)


def main():
    globe = pathlib.Path(sys.argv[1]).resolve()
    fixture = records(pathlib.Path("DATABASES_MPI/proc000000_specfempp_database.bin"))
    assert struct.unpack("<3i", fixture[1]) == (1, 2, 6)
    r_planet = struct.unpack("<6d", fixture[2])[0]
    nnode = struct.unpack("<i", fixture[11])[0]
    xyz = struct.unpack(f"<{3 * nnode}d", fixture[12])
    nspec = struct.unpack("<i", fixture[14])[0]
    nodes = struct.unpack(f"<{27 * nspec}i", fixture[18])
    elements = [0, 1024, 4096, 8192, 9200, nspec - 1]
    points = []
    for element in elements:
        node = nodes[27 * element] - 1
        points.append([xyz[node + dim * nnode] / r_planet for dim in range(3)])

    # Constants used by the two unmodified routines, from setup/constants.h.in.
    source = """module constants
  implicit none
  double precision, parameter :: ZERO=0.d0, SMALL_VAL_ANGLE=1.d-10
  double precision, parameter :: PI=3.141592653589793d0, TWO_PI=2.d0*PI
  double precision, parameter :: TINYVAL=1.d-9
end module
"""
    source += subroutine(globe / "src/shared/rthetaphi_xyz.f90", "xyz_2_rthetaphi_dble")
    source += "\n" + subroutine(globe / "src/shared/reduce.f90", "reduce")
    source += """
program dump
  implicit none
  double precision :: x,y,z,r,theta,phi
  integer :: i
  do i=1,6
    read(*,*) x,y,z
    call xyz_2_rthetaphi_dble(x,y,z,r,theta,phi)
    call reduce(theta,phi)
    write(*,'(3(es25.17,1x))') r,theta,phi
  enddo
end program
"""
    with tempfile.TemporaryDirectory() as directory:
        work = pathlib.Path(directory)
        (work / "dump.f90").write_text(source)
        subprocess.run(["gfortran", "dump.f90", "-o", "dump"], cwd=work, check=True)
        output = subprocess.run(
            [str(work / "dump")],
            input="\n".join(" ".join(map(str, point)) for point in points),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
    pathlib.Path("spherical_coordinates.txt").write_text(
        "".join(
            f"{element} {line.strip()}\n" for element, line in zip(elements, output)
        )
    )


if __name__ == "__main__":
    main()
