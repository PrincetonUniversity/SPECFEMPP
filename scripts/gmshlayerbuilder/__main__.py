try:
    import _gmshlayerbuilder
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    import _gmshlayerbuilder


from _gmshlayerbuilder.dim2 import layer2d

layer2d.demo()
