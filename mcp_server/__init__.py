"""
Runtime state shared between tools and the main driver.

The main.py entry point calls set_runtime(driver, persistence) once at startup,
and every tool function reads from get_driver() / get_persistence().
"""

from typing import Optional

_driver = None
_persistence = None
_run_id = "default"


def set_runtime(driver, persistence, run_id: str = "default"):
    global _driver, _persistence, _run_id
    _driver = driver
    _persistence = persistence
    _run_id = run_id


def get_driver():
    if _driver is None:
        raise RuntimeError("EPDriver not set. Call set_runtime() first.")
    return _driver


def get_persistence():
    if _persistence is None:
        raise RuntimeError("Persistence not set. Call set_runtime() first.")
    return _persistence


def get_run_id() -> str:
    return _run_id