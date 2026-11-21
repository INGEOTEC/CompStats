import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_package_versions.py"
WORKFLOWS = ROOT / ".github" / "workflows"
# Built from fragments so that this file does not contain the legacy names it checks for.
LEGACY_RUNNER = "nose" + "tests"
LEGACY_WORKFLOW = "test" + ".yaml"

pytest.importorskip("tomllib")    # Python 3.11+
pytest.importorskip("packaging")
pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(), reason="needs the repository checkout (scripts/)")

from packaging.version import Version  # noqa: E402


def _load_script():
    """Import scripts/check_package_versions.py as a module"""
    spec = importlib.util.spec_from_file_location("check_package_versions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_version_matches_package():
    """The version read from the source equals the imported one"""
    import CompStats
    module = _load_script()
    assert module.local_version() == CompStats.__version__


def test_next_versions():
    """Next patch and next minor"""
    module = _load_script()
    assert module.next_versions(Version("0.1.15")) == (Version("0.1.16"),
                                                       Version("0.2.0"))


def test_check_package_never_published(monkeypatch):
    """A package PyPI does not know always passes"""
    module = _load_script()
    monkeypatch.setattr(module, "pypi_latest_version", lambda package=None: None)
    assert module.check_package()[3] is True


def test_check_package_single_step(monkeypatch):
    """Next minor and next patch are accepted"""
    module = _load_script()
    monkeypatch.setattr(module, "local_version", lambda package_dir=None: "1.4.0")
    monkeypatch.setattr(module, "pypi_latest_version",
                        lambda package=None: "1.3.7")
    assert module.check_package()[3] is True
    monkeypatch.setattr(module, "local_version",
                        lambda package_dir=None: "1.3.10")
    monkeypatch.setattr(module, "pypi_latest_version",
                        lambda package=None: "1.3.9")
    assert module.check_package()[3] is True


def test_check_package_not_ahead(monkeypatch):
    """Equal or greater PyPI version fails"""
    module = _load_script()
    monkeypatch.setattr(module, "local_version", lambda package_dir=None: "1.4.0")
    for pypi in ("1.4.0", "1.5.0"):
        monkeypatch.setattr(module, "pypi_latest_version",
                            lambda package=None, pypi=pypi: pypi)
        assert module.check_package()[3] is False


def test_check_package_jump(monkeypatch):
    """Jumping more than one release fails and reports the expectation"""
    module = _load_script()
    monkeypatch.setattr(module, "local_version", lambda package_dir=None: "1.5.0")
    monkeypatch.setattr(module, "pypi_latest_version",
                        lambda package=None: "1.3.7")
    _, _, _, ok, detail = module.check_package()
    assert ok is False
    assert "1.3.8" in detail and "1.4.0" in detail


def test_workflows():
    """Legacy workflows are gone and the new ones have the key steps"""
    assert (WORKFLOWS / "test.yml").is_file()
    assert (WORKFLOWS / "publish-pypi.yml").is_file()
    assert not (WORKFLOWS / LEGACY_WORKFLOW).exists()
    assert not (WORKFLOWS / "pip.yaml").exists()
    test = (WORKFLOWS / "test.yml").read_text(encoding="utf-8")
    assert "pytest CompStats" in test
    assert "scripts/check_package_versions.py" in test
    assert LEGACY_RUNNER not in test
    publish = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "secrets.TWINE" in publish
    assert "twine check" in publish
    # Publishing is manual only: neither a tag nor a release triggers it.
    assert "workflow_dispatch" in publish
    for trigger in ("push:", "tags:", "release:"):
        assert trigger not in publish


def test_pyproject_build_system_and_test_extra():
    """pyproject declares the build backend and a pytest test extra"""
    import tomllib
    with open(ROOT / "pyproject.toml", "rb") as fpt:
        data = tomllib.load(fpt)
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    extra = data["project"]["optional-dependencies"]["test"]
    assert any(item.startswith("pytest") for item in extra)
