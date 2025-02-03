from os import getcwd
from subprocess import check_call

nx = 80
nz = 60

mediums = {
    "iso": "1 1 2700.d0 3000.d0 0 0 0 9999 9999 0 0 0 0 0 0",
    "eiso": "1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0",
    "eani": "1 2 7100. 16.5d10 5.d10 0 6.2d10 0 3.96d10 0 0 0 0 0 0",
}

files = "Par_file", "source.yaml", "specfem_config.yaml", "topography.dat"

for medium in mediums:
    cwd = f"{getcwd()}/data_{medium}"
    check_call(f"rm -rf {cwd} && mkdir {cwd}", shell=True)

    for file in files:
        with open(f"{getcwd()}/data/{file}", "r") as f:
            content = (
                f.read()
                .replace("${cwd}", cwd)
                .replace("${medium}", mediums[medium])
                .replace("${nx}", str(nx))
                .replace("${nz}", str(nz))
                .replace("${title}", f"{medium}_{nx}x{nz}")
            )

        with open(f"{cwd}/{file}", "w") as f:
            f.write(content)

    check_call("xmeshfem2D -p Par_file", cwd=cwd, shell=True)
