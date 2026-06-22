from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override


@dataclass(frozen=True)
class MaterialModel(ABC):
    @abstractmethod
    def material_string3D(self, model_number: int) -> str:
        # domain_ID material_ID rho vp vs Qkappa Qmu anisotropy_flag
        ...


# For classical materials (i.e., spectral elements for which the velocity and
# density model will not be assigned by calling an external function to define
# for instance a tomographic model), the format is:
#         domain_ID material_ID rho vp vs Qkappa Qmu anisotropy_flag
# where domain_ID is 1 for acoustic and 2 for elastic or viscoelastic materials,
# material_ID a unique identifier, rho the density in $kg\, m^{-3}$,
# vp the P-wave speed in $m\, s^{-1}$, vs the S-wave speed in $m\, s^{-1}$,
# Q the quality factor and anisotropy_flag an identifier for anisotropic models.
# Note that both Qkappa and Qmu are ignored by the code unless ATTENUATION is set.
# If you want a model with no Qmu attenuation, both set ATTENUATION to .false. in
# the DATA/Par_file and set Qmu to 9999 here. If you want a model with no Qkappa
# attenuation, set Qkappa to 9999 here. Note that Qmu is always equal to Qs, but
# Qkappa is in general not equal to Qp.


@dataclass(frozen=True)
class MaterialModelAcoustic(MaterialModel):
    typecode: int = field(init=False, default=1)
    rho: float
    vp: float
    Qkappa: float = 9999
    Qmu: float = 9999

    @override
    def material_string3D(self, model_number: int) -> str:
        return f"1 {model_number} {self.rho} {self.vp} 0 {self.Qkappa} {self.Qmu} 0"


@dataclass(frozen=True)
class MaterialModelElastic(MaterialModel):
    typecode: int = field(init=False, default=1)
    rho: float
    vp: float
    vs: float
    Qkappa: float = 9999
    Qmu: float = 9999

    @override
    def material_string3D(self, model_number: int) -> str:
        return (
            f"2 {model_number} {self.rho} {self.vp} {self.vs} "
            f"{self.Qkappa} {self.Qmu} 0"
        )
