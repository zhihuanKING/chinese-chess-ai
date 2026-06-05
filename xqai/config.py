"""Configuration loading for xqai.

Reads ``configs/default.yaml`` (INTERFACES.md §9) and wraps the nested mapping
in a recursive dotted-access object so callers can write ``cfg.network.channels``
instead of ``cfg["network"]["channels"]``.

Example
-------
>>> from xqai.config import load_config
>>> cfg = load_config()
>>> cfg.network.channels
128
>>> cfg.train.batch_size
2048
>>> cfg.to_dict()["mcts"]["c_puct"]
1.5
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import yaml

# Repo layout: this file is <root>/xqai/config.py
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_PKG_DIR)
DEFAULT_CONFIG_PATH = os.path.join(_ROOT_DIR, "configs", "default.yaml")


class Config:
    """Recursive dotted-access wrapper around a (possibly nested) mapping.

    Nested dicts become nested ``Config`` objects; lists are wrapped element by
    element so dicts inside lists are also dotted-access. Attribute access,
    item access and ``in`` all work. Use :meth:`to_dict` to get plain data back.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None):
        object.__setattr__(self, "_data", {})
        if data:
            for key, value in data.items():
                self._data[key] = _wrap(value)

    # --- attribute access -------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup (incl. __slots__) fails.
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key {name!r} (available: {sorted(self._data)})"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = _wrap(value)

    # --- item access ------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = _wrap(value)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # --- conversion / repr ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain (recursively unwrapped) ``dict``."""
        return _unwrap(self)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_wrap(v) for v in value)
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return {k: _unwrap(v) for k, v in value._data.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(v) for v in value)
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load a YAML config file into a dotted-access :class:`Config`.

    Parameters
    ----------
    path:
        Path to a YAML file. Defaults to ``configs/default.yaml`` at the repo
        root (resolved relative to this package).
    """
    cfg_path = os.fspath(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Top-level YAML in {cfg_path!r} must be a mapping, got {type(data)}")
    return Config(data)


__all__ = ["Config", "load_config", "DEFAULT_CONFIG_PATH"]
