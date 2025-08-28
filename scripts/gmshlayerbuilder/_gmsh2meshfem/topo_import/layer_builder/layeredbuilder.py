import itertools

from _gmsh2meshfem.gmsh_dep import GmshContext
from _gmsh2meshfem.dim2.model import Model

from .layer import Layer, LayerBoundary


class LayeredBuilder:
    """Generates a layer topography domain in 2D, spanning from x=xlow to x=xhigh.
    Each layer `layers[i]` is bounded below by `boundaries[i]` and above by `boundaries[i+1]`.
    """

    xlow: float
    xhigh: float

    boundaries: list[LayerBoundary]
    layers: list[Layer]

    @property
    def width(self):
        return self.xhigh - self.xlow

    def __init__(self, xlow: float, xhigh: float):
        self.xlow = xlow
        self.xhigh = xhigh
        self.layers = []
        self.boundaries = []

    def create_model(self) -> Model:
        with GmshContext() as gmsh:
            built_layerbds = [
                bdlayer.build_layer(self.xlow, self.xhigh, gmsh=gmsh)
                for bdlayer in self.boundaries
            ]
            for ilayer, layerbd in enumerate(built_layerbds):
                layerbd.initialize_curve_copy(
                    None if ilayer == 0 else self.layers[ilayer - 1],
                    None if ilayer == len(self.layers) else self.layers[-1],
                    gmsh,
                )

            # store tags
            surfaces = []
            for i, (l0, l1) in enumerate(itertools.pairwise(built_layerbds)):
                layer_result = self.layers[i].generate_layer(l0, l1, gmsh)
                surfaces.append(layer_result.surface_index)

            # required for ngnod = 9
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate()

            # === uncomment this to see GUI ===
            # gmsh.fltk.run()

            # =====================================================================
            #                      extract mesh model
            # =====================================================================
            return Model.from_meshed_surface(surface=surfaces,gmsh=gmsh)
