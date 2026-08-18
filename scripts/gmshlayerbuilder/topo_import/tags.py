from typing import Literal

IS_FLUID_PER_MATERIAL_STRCODE: dict[str, bool] = {"S": False, "F": True}

BoundaryConditionType = Literal["neumann", "acoustic_free_surface", "absorbing"]
BOUNDARY_TYPES = ["neumann", "acoustic_free_surface", "absorbing"]

EPS = 1e-6
