import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "source"

pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="needs the repository checkout (docs/source)")

_AUTOMODULE_RE = re.compile(r"^\.\.\s+automodule::\s+(\S+)\s*$", re.MULTILINE)

MODULES = ["bootstrap", "interface", "measurements", "metrics",
           "performance", "utils"]


def test_every_module_is_documented_exactly_once():
    """Each CompStats module appears in exactly one automodule directive"""
    stems = sorted(path.stem for path in (ROOT / "CompStats").glob("*.py")
                   if path.name != "__init__.py")
    documented = []
    for rst in DOCS.glob("*.rst"):
        documented += _AUTOMODULE_RE.findall(rst.read_text(encoding="utf-8"))
    for stem in stems:
        assert documented.count(f"CompStats.{stem}") == 1, stem


def test_public_api_has_docstrings():
    """Every public function, class, method and property has a docstring"""
    missing = []
    for name in MODULES:
        module = importlib.import_module(f"CompStats.{name}")
        for obj_name, obj in vars(module).items():
            if obj_name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if not (inspect.getdoc(obj) or "").strip():
                missing.append(f"{name}.{obj_name}")
            if not inspect.isclass(obj):
                continue
            for member_name, member in vars(obj).items():
                if member_name.startswith("_"):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if not (inspect.isfunction(member)
                        or isinstance(member, property)):
                    continue
                if not (member.__doc__ or "").strip():
                    missing.append(f"{name}.{obj_name}.{member_name}")
    assert not missing, missing


def test_index_sections_in_order():
    """The landing page has the expected sections, in order"""
    text = (DOCS / "index.rst").read_text(encoding="utf-8")
    position = -1
    for title in ["Statistical background", "Architecture",
                  "Invariants to preserve", "Development", "Citing", "API"]:
        found = text.find(f"\n{title}\n", position + 1)
        assert found > position, title
        position = found


def test_index_cites_both_papers():
    """Both papers are cited with a well-formed DOI"""
    text = (DOCS / "index.rst").read_text(encoding="utf-8")
    assert "10.1016/j.patrec.2024.03.010" in text
    assert "10.1007/978-3-031-33783-3_9" in text
    assert "/COVER" not in text


def test_no_readthedocs_in_developer_docs():
    """ReadTheDocs is not used any more"""
    word = "readthe" + "docs"
    files = list(DOCS.glob("*.rst")) + [DOCS / "conf.py"]
    for path in files:
        assert word not in path.read_text(encoding="utf-8").lower(), path.name
    assert not (ROOT / (".read" + "thedocs.yaml")).exists()


def test_conf_uses_furo_and_intersphinx():
    """The theme is furo and the external inventories are mapped"""
    text = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert "html_theme = 'furo'" in text
    assert "sys.path" not in text
    mapping = text[text.index("intersphinx_mapping"):]
    for name in ["numpy", "pandas", "scipy", "sklearn", "statsmodels",
                 "seaborn"]:
        assert f"'{name}'" in mapping, name


def test_docs_job_in_ci():
    """CI runs the strict Sphinx build"""
    workflow = ROOT / ".github" / "workflows" / "test.yml"
    if not workflow.is_file():
        pytest.skip("needs the repository checkout (.github/workflows)")
    command = "python -m sphinx -n -W --keep-going -b html docs/source docs/build/html"
    assert command in workflow.read_text(encoding="utf-8")
