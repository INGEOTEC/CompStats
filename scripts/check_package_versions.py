#!/usr/bin/env python3
"""Check the package's local version against PyPI.

``publish-pypi.yml`` is run by hand and uploads whatever version the checked
out code declares. Nothing in it checks whether the local version is actually
ahead of what PyPI has, or whether it jumped more than one release past it.
This script reports PyPI's latest version, the
local version (read from the attribute ``pyproject.toml`` declares as the
dynamic version, without importing the package, so none of its dependencies
need to be installed), and whether the local version is a valid single-step
jump ahead of PyPI: the next patch, or the next minor with patch reset to 0.
A package PyPI has never published always passes.

Usage::

    python scripts/check_package_versions.py
"""

import ast
import json
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

#: Name of the distribution on PyPI.
PACKAGE = "CompStats"


def pypi_latest_version(package: str = PACKAGE):
    """Return the latest version PyPI reports for `package`, or `None` if it
    has never been published (a 404 from PyPI's JSON API)."""
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)["info"]["version"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def local_version(package_dir: Path = ROOT) -> str:
    """Return the version ``pyproject.toml`` resolves dynamically, read
    straight out of the source file its ``attr`` points at (no import)."""
    with open(package_dir / "pyproject.toml", "rb") as fpt:
        pyproject = tomllib.load(fpt)
    attr = pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    module_name, attr_name = attr.rsplit(".", 1)
    module_path = package_dir / module_name.replace(".", "/") / "__init__.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"),
                     filename=str(module_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == attr_name
               for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{attr} not found in {module_path}")


def next_versions(version: Version):
    """Return the two versions immediately ahead of `version`: the next
    patch, and the next minor with patch reset to 0."""
    major, minor, patch = (tuple(version.release) + (0, 0, 0))[:3]
    return (Version(f"{major}.{minor}.{patch + 1}"),
            Version(f"{major}.{minor + 1}.0"))


def check_package(package: str = PACKAGE, package_dir: Path = ROOT):
    """Return ``(package, pypi_version, local_version, ok, detail)``."""
    pypi = pypi_latest_version(package)
    local = local_version(package_dir)
    if pypi is None:
        return package, pypi, local, True, "never published"
    local_v, pypi_v = Version(local), Version(pypi)
    if local_v <= pypi_v:
        return package, pypi, local, False, f"local is not ahead of PyPI ({pypi})"
    next_patch, next_minor = next_versions(pypi_v)
    if local_v not in (next_patch, next_minor):
        return (package, pypi, local, False,
                f"jumps more than one release past PyPI ({pypi}); "
                f"expected {next_patch} or {next_minor}")
    return package, pypi, local, True, "ok"


def main() -> int:
    """Print the verdict and return the process exit status."""
    package, pypi, local, ok, detail = check_package()
    print(f"{'package':<12}  {'pypi':<10}  {'local':<10}  verdict")
    verdict = "OK" if ok else "FAIL"
    print(f"{package:<12}  {pypi or '-':<10}  {local:<10}  {verdict}: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
