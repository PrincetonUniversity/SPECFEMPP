from typing import TypedDict
import yaml

from ._core import _default_file_path, _execute


ParameterDict = TypedDict(
    "ParameterDict",
    {
        "header": dict,
        "simulation-setup": dict,
        "receivers": dict,
        "run-setup": dict,
        "databases": dict,
    },
    total=False,
)


SourceDict = TypedDict(
    "SourceDict", {"number-of-sources": int, "sources": dict}, total=False
)


# cache for runtime and default parameters
_runtime_config = {}
_default_config = {}

# whether default file specified by C++ has been loaded
_default_loaded = False


def _merge(dict1, dict2):
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                _merge(dict1[key], dict2[key])
            else:
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]


def _get(target: dict, keys: tuple[str, ...]):
    if len(keys) == 0:
        raise ValueError("At least one key is required")

    if keys[0] in target:
        if len(keys) == 1:
            return target[keys[0]]

        return _get(target, keys[1:])

    return None


def _load_default():
    global _default_loaded
    if not _default_loaded:
        with open(_default_file_path, "r") as f:
            _merge(_default_config, yaml.safe_load(f))
        _default_loaded = True


def set_par(pars: ParameterDict):
    """Set runtime parameters."""
    _merge(_runtime_config, {"parameters": pars})


def set_default_par(pars: ParameterDict):
    """Set default parameters."""
    _load_default()
    _merge(_default_config, {"parameters": pars})


def load_par(src: str):
    """Load runtime parameters from a YAML file."""
    with open(src, "r") as f:
        _merge(_runtime_config, yaml.safe_load(f))


def load_default_par(src: str):
    """Load default parameters from a YAML file."""
    global _default_loaded
    _default_loaded = True
    with open(src, "r") as f:
        _merge(_default_config, yaml.safe_load(f))


def get_par(*keys: str):
    """Get a runtime parameter."""
    return _get(_runtime_config, keys)


def get_default_par(*keys):
    """Get a default parameter."""
    return _get(_default_config, keys)


def execute():
    """Execute the simulation."""
    _load_default()
    _execute(
        yaml.dump(_runtime_config, sort_keys=False),
        yaml.dump(_default_config, sort_keys=False),
    )
