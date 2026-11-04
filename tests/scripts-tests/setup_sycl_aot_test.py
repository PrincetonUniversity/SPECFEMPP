import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "setup_sycl_aot", ROOT / "scripts/setup_sycl_aot.py"
)
aot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aot)


class SyclAotTest(unittest.TestCase):
    def test_regeneration_preserves_custom_flags_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CMakeUserPresets.json"
            custom = ["-O2", "-D_GLIBCXX_USE_CXX11_ABI=0", '-DLABEL="two words"']
            for typed in (False, True):
                with self.subTest(typed=typed):
                    flags = shlex.join(custom + aot.aot_flags("0x7d51"))
                    if typed:
                        flags = {"type": "STRING", "value": flags}
                    path.write_text(
                        json.dumps(
                            {
                                "version": 6,
                                "configurePresets": [
                                    {"name": "unrelated"},
                                    {
                                        "name": aot.PRESET_NAME,
                                        "cacheVariables": {
                                            "CMAKE_CXX_FLAGS": flags,
                                            "CUSTOM_SETTING": "keep",
                                        },
                                        "environment": {"CUSTOM_ENV": "keep"},
                                    },
                                ],
                            }
                        )
                    )
                    for device in ("0x7d55", "0x7d55", "0x7d51"):
                        aot.update_presets(path, device, "/compiler/icpx")
                        data = json.loads(path.read_text())
                        self.assertEqual(
                            data["configurePresets"][0], {"name": "unrelated"}
                        )
                        preset = data["configurePresets"][1]
                        cache = preset["cacheVariables"]
                        self.assertEqual(
                            shlex.split(cache["CMAKE_CXX_FLAGS"]),
                            custom + aot.aot_flags(device),
                        )
                        self.assertEqual(cache["CUSTOM_SETTING"], "keep")
                        self.assertEqual(preset["environment"], {"CUSTOM_ENV": "keep"})
                        self.assertEqual(len(data["buildPresets"]), 1)

    def test_preserves_other_backend_options(self):
        existing = (
            "-O2 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "
            '"-device 0x7d51 -options custom"'
        )
        self.assertEqual(
            shlex.split(aot.merge_aot_flags(existing, "0x7d55")),
            ["-O2", "-Xsycl-target-backend=spir64_gen", "-options custom"]
            + aot.aot_flags("0x7d55"),
        )

    def test_relative_compiler_is_absolute_before_validation(self):
        for discovered in ("./icpx", "relative/bin/icpx"):
            with self.subTest(discovered=discovered):
                with (
                    patch.object(
                        sys,
                        "argv",
                        ["setup_sycl_aot.py", "--device", "7d51", "--configure"],
                    ),
                    patch.object(aot.shutil, "which", return_value=discovered),
                    patch.object(aot, "validate_compiler") as validate,
                    patch.object(aot, "update_presets") as update,
                    patch.object(aot.subprocess, "run") as configure,
                ):
                    self.assertEqual(aot.main(), 0)
                    compiler = os.path.abspath(discovered)
                    validate.assert_called_once_with(compiler, "0x7d51")
                    self.assertEqual(update.call_args.args[2], compiler)
                    self.assertEqual(configure.call_args.kwargs["cwd"], ROOT)


class BackendCacheTest(unittest.TestCase):
    def test_switching_backends_recomputes_both_cached_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.17.5)\n"
                "project(backend_cache NONE)\n"
                f'include("{ROOT}/cmake/variables.cmake")\n'
            )
            build = Path(directory) / "build"
            for backend in ("CPU", "SYCL", "CPU", "CUDA", "CPU", "HIP", "CPU"):
                command = ["cmake", "-S", str(source), "-B", str(build)]
                command += [
                    f"-DKokkos_ENABLE_{name}={'ON' if name == backend else 'OFF'}"
                    for name in ("SYCL", "CUDA", "HIP")
                ]
                subprocess.run(command, capture_output=True, text=True, check=True)
                cache = (build / "CMakeCache.txt").read_text()
                self.assertIn(
                    f"SPECFEM_CPU_BUILD:BOOL={'TRUE' if backend == 'CPU' else 'FALSE'}",
                    cache,
                )
                self.assertIn(
                    f"SPECFEM_GPU_BUILD:BOOL={'FALSE' if backend == 'CPU' else 'TRUE'}",
                    cache,
                )


if __name__ == "__main__":
    unittest.main()
