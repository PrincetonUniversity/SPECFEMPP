from types import TracebackType
from typing import Any


gmsh_inactive = True

try:
    import gmsh as _gmsh_module  # type: ignore  # noqa: F401

    gmsh_inactive = False
except ImportError:
    _gmsh_module: Any = None


_ACTIVE_ENVIRONMENT: "GmshContext | None" = None


class GmshContext:
    """Wraps the gmsh module
    """
    def is_active(self):
        return _ACTIVE_ENVIRONMENT == self

    def for_element_types_in_entity(self, *args, **kwargs):
        mapping = {}
        if "mapping" in kwargs:
            mapping = kwargs["mapping"]
            del kwargs["mapping"]
        elif len(args) > 0:
            mapping = args[-1]
            args = args[:-1]

        for eltype, elems, nodes in zip(
            *self.model.mesh.get_elements(*args, **kwargs), strict=True
        ):
            # element types:
            # https://gitlab.onelab.info/gmsh/gmsh/blob/master/src/common/GmshDefines.h
            if eltype in mapping:
                mapping[eltype](elems, nodes)
            else:
                msg = (
                    f"Unsupported element type {eltype}. Please verify "
                    "the element type is expected as per gmsh's MSH file format ("
                    "https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format"
                    "). Alternatively, verify against GmshDefines ("
                    "https://gitlab.onelab.info/gmsh/gmsh/blob/master/src/common/GmshDefines.h"
                    "). If the type should be supported, please raise an issue."
                )
                raise ValueError(msg)

    def __enter__(self):
        global _ACTIVE_ENVIRONMENT
        if gmsh_inactive:
            msg = "gmsh was not imported. Cannot enter gmsh context."
            raise RuntimeError(msg)
        if _ACTIVE_ENVIRONMENT is not None:
            msg = (
                "gmsh was already activated. "
                "Only one GmshContext instance "
                "can be active at a time."
            )
            raise RuntimeError(msg)
        _ACTIVE_ENVIRONMENT = self
        _gmsh_module.initialize()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Any, exc_tb: TracebackType):
        global _ACTIVE_ENVIRONMENT
        # context manager:
        # exc_type: type of the exception, or None if no exception was made
        # exc_val: the exception (of type given by exc_type), or None if no exception
        # exc_tb: traceback obj, or None if no exception
        #       (https://docs.python.org/3/reference/datamodel.html#traceback-objects)
        # return True to suppress. Otherwise, exception will continue as normal
        # (should not reraise)
        if gmsh_inactive:
            msg = (
                "gmsh was not imported. You did something crazy to "
                "try to exit the gmsh context without it imported. "
                "I would not have let you."
            )
            raise RuntimeError(msg)
        if self != _ACTIVE_ENVIRONMENT:
            msg = (
                "The active GmshContext instance is different from the"
                " one being exited. Only one instance can be active at a "
                "time, so something went wrong."
            )
            raise ValueError(msg)

        _gmsh_module.finalize()
        _ACTIVE_ENVIRONMENT = None

    def __getattribute__(self, name):
        # if active, have getattr() first fallback to the gmsh module
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            if self.is_active():
                # override gmsh behavior

                # context manager already handles initialize and finalize
                if name == "initialize" or name == "finalize":
                    raise AttributeError
                return getattr(_gmsh_module, name)
            else:
                raise e
