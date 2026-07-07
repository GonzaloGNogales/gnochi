import numpy as np


_NUMPY_LEGACY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}

for alias_name, alias_value in _NUMPY_LEGACY_ALIASES.items():
    if alias_name not in np.__dict__:
        setattr(np, alias_name, alias_value)
