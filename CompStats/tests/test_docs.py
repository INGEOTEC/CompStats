import hashlib
import importlib
import inspect
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "source"

pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="needs the repository checkout (docs/source)")

SITE = ROOT / "quarto"
_SKIPPED_DIRS = {"_site", "_freeze", ".quarto"}
needs_site = pytest.mark.skipif(
    not SITE.is_dir(), reason="needs the repository checkout (quarto/)")

_AUTOMODULE_RE = re.compile(r"^\.\.\s+automodule::\s+(\S+)\s*$", re.MULTILINE)



def _site_files(suffix):
    """Source pages of the Quarto site with the given suffix"""
    return sorted(p for p in SITE.rglob(f"*{suffix}")
                  if not _SKIPPED_DIRS & set(p.relative_to(SITE).parts))


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


def test_no_legacy_hosting_in_developer_docs():
    """The previous hosting service is not referenced any more"""
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


@needs_site
def test_freeze_matches_notebooks():
    """The freeze entry of every notebook is up to date"""
    notebooks = _site_files(".ipynb")
    assert notebooks
    for notebook in notebooks:
        relative = notebook.relative_to(SITE).with_suffix("")
        entry = SITE / "_freeze" / relative / "execute-results" / "html.json"
        fix = ('render locally with QUARTO_PYTHON="$PWD/.venv/bin/python" '
               'quarto render quarto and commit quarto/_freeze/')
        assert entry.is_file(), f"{entry} is missing: {fix}"
        digest = hashlib.md5(notebook.read_bytes()).hexdigest()
        stored = json.loads(entry.read_text(encoding="utf-8"))["hash"]
        assert stored == digest, f"{notebook.name} and its freeze differ: {fix}"


@needs_site
def test_no_orphan_freeze_entries():
    """Every freeze entry belongs to an existing notebook"""
    freeze = SITE / "_freeze"
    if not freeze.is_dir():
        pytest.skip("no freeze directory")
    for entry in freeze.rglob("execute-results/html.json"):
        page = entry.parent.parent.relative_to(freeze)
        assert (SITE / page).with_suffix(".ipynb").is_file(), page


@needs_site
def test_install_cells_are_not_executed():
    """The Colab-only install cell is shown but not executed"""
    for notebook in _site_files(".ipynb"):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        cells = ["".join(c["source"]) for c in cells
                 if c["cell_type"] == "code"]
        cells = [c for c in cells if "!pip install" in c]
        assert len(cells) == 1, notebook.name
        assert cells[0].startswith("#| eval: false"), notebook.name


@needs_site
def test_navbar_pages_exist():
    """Every local link of the navbar names an existing page"""
    text = (SITE / "_quarto.yml").read_text(encoding="utf-8")
    hrefs = re.findall(r"^\s*-?\s*href:\s*(\S+)\s*$", text, re.MULTILINE)
    local = [h for h in hrefs if not h.startswith("http")]
    for href in local:
        assert (SITE / href).is_file(), href
    for page in ["index.qmd", "pages/install.qmd", "pages/methodology.qmd",
                 "pages/citing.qmd", "pages/sklearn_users.ipynb",
                 "pages/competition_organizers.ipynb",
                 "pages/multiple_testing.ipynb"]:
        assert page in local, page


@needs_site
def test_user_docs_use_the_perf_api():
    """User-facing examples do not use the DataFrame-based functions"""
    forbidden = ["from CompStats.performance", "from CompStats import performance"]
    texts = [p.read_text(encoding="utf-8") for p in _site_files(".qmd")]
    for notebook in _site_files(".ipynb"):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        texts += ["".join(c["source"]) for c in cells
                  if c["cell_type"] == "code"]
    readme = ROOT / "README.rst"
    if readme.is_file():
        texts.append(readme.read_text(encoding="utf-8"))
    for text in texts:
        for item in forbidden:
            assert item not in text, item


@needs_site
def test_methodology_page_content():
    """The methodology page covers the published methodology"""
    text = (SITE / "pages" / "methodology.qmd").read_text(encoding="utf-8")
    lower = text.lower()
    for word in ["bootstrap", "confidence interval", "paired", "p-value",
                 "Bonferroni", "Holm", "Benjamini", "Berg-Kirkpatrick",
                 "coefficient of variation", "PPI"]:
        assert word.lower() in lower, word
    assert "10.1016/j.patrec.2024.03.010" in text


def test_no_legacy_hosting_anywhere():
    """No tracked text file refers to the previous hosting service"""
    word = "readthe" + "docs"
    try:
        output = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                                capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout")
    for name in output.decode("utf-8").split("\0"):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert word not in text.lower(), name


def test_readme():
    """The README links both sites and is valid reStructuredText"""
    readme = ROOT / "README.rst"
    if not readme.is_file():
        pytest.skip("needs the repository checkout (README.rst)")
    text = readme.read_text(encoding="utf-8")
    assert "https://ingeotec.github.io/CompStats/" in text
    assert "https://ingeotec.github.io/CompStats/dev/" in text
    assert "10.1016/j.patrec.2024.03.010" in text
    assert "raw/docs/docs" not in text
    core = pytest.importorskip("docutils.core")
    core.publish_string(text, writer_name="html",
                        settings_overrides={"halt_level": 2,
                                            "report_level": 2})


def test_site_workflow():
    """The site is published by one workflow that deploys to GitHub Pages"""
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip("needs the repository checkout (.github/workflows)")
    website = workflows / "website.yml"
    assert website.is_file()
    assert not (workflows / "publish.yml").exists()
    text = website.read_text(encoding="utf-8")
    assert "quarto render quarto" in text
    assert ("python -m sphinx -n -W --keep-going -b html docs/source "
            "quarto/_site/dev") in text
    assert "actions/deploy-pages" in text
    assert "conda" not in text
    assert "gh-pages" not in text


def test_project_urls():
    """The project URLs point at the GitHub Pages site"""
    tomllib = pytest.importorskip("tomllib")
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("needs the repository checkout (pyproject.toml)")
    with open(pyproject, "rb") as fpt:
        urls = tomllib.load(fpt)["project"]["urls"]
    assert urls["Homepage"] == "https://ingeotec.github.io/CompStats/"
    assert urls["Documentation"] == "https://ingeotec.github.io/CompStats/"


@needs_site
def test_site_follows_the_article():
    """The wording of the site agrees with the published article"""
    index = (SITE / "index.qmd").read_text(encoding="utf-8").lower()
    assert "developed" + " by" not in index
    assert "validated" not in index
    lines = (SITE / "_quarto.yml").read_text(encoding="utf-8").splitlines()
    marker = next(i for i, line in enumerate(lines)
                  if "Code under the Apache 2.0 license" in line)
    assert "INGEOTEC" not in "\n".join(lines[max(marker - 1, 0):marker + 1])
    methodology = (SITE / "pages" / "methodology.qmd").read_text(encoding="utf-8")
    assert "m(m-1)/2" in methodology.replace(" ", "")
    assert "four" + " indicators" not in methodology
