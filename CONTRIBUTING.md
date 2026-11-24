# Contributing

## Check changes locally

Create and activate the environment in `environment.yml`, then run:

```bash
cd Python
python -m pytest --ignore-glob='tests/test_plugin_*.py'
```

This is the backend environment used by CI. For a pip setup, use a Python
3.12 environment and run the following from the repository root:

```bash
python -m pip install './Python/lumic[all]'
cd Python
python -m pytest --ignore-glob='tests/test_plugin_*.py'
```

The backend test suite includes plotting examples and mocked FPbase requests,
so it uses those extras. To run notebooks and examples without pytest, install
`.[examples,fpbase]` in a dedicated Python 3.12 environment and select that
environment in Jupyter. The offline simulation only needs `.[examples]`.

Conda packages use only `conda-forge`. The `nodefaults` entry prevents channels
from your existing Conda configuration being appended during environment
creation. This applies whether you use Miniforge or another Conda installer.
The existing pip dependencies still come from PyPI. CI sets conda-forge with
strict channel priority and verifies installed package sources before pytest.
Changing the YAML does not retroactively replace packages in existing local
environments or change global Conda settings. Fresh creation provides the
cleanest verification. For subsequent manual Conda installs, use
`conda install --override-channels -c conda-forge PACKAGE` to retain this policy.

Keep this local check for feedback before pushing. Include a regression test
for bug fixes and relevant tests for new behavior. Keep tests deterministic
and explain the expected result. Existing tests cannot verify behavior they
do not exercise.

See the [test suite guide](Python/tests/README.md) for test locations,
focused commands, and fixture conventions. For the Napari tests, follow the
[plugin test instructions](Python/plugin/README.md#run-plugin-tests).

## Pull requests and GitHub checks

Work on a branch and open a PR targeting `main`. Explain the change, its
verification, and relevant limitations. GitHub Actions creates `unmixing-env`
from the root `environment.yml` on Ubuntu, then runs pytest from `Python/`.
The check is named **Python tests (unmixing-env)**.

The same workflow also runs **Backend wheel** checks on Linux and Windows
with Python 3.12. Each job builds a source distribution
and a wheel from it, installs the wheel in a clean virtual environment, and
runs the installed backend smoke test outside the checkout. The smoke test
reads a bundled CZI fixture from the repository, uses the installed FPbase
spectra for an offline fit, and saves and reloads abundance maps as OME-TIFF.

It runs on all PRs targeting `main` and pushes to `main`. Documentation-only
PRs run too, so a required check is never missing due to a path filter. New PR
commits rerun the check and cancel obsolete runs. After merging the workflow,
manual runs are available from the Actions tab.

Installation needs network access; tests themselves are offline. The suite
uses mocked FPbase responses and small bundled CZI fixtures, without private
microscope data. Larger evaluations and benchmarks remain separate. CI
initially covers Linux and Python 3.12 only. Dependency ranges are not a full
lock; fresh installation can reveal issues absent from older local environments.

If CI fails, open **Details** and inspect the failing step: environment setup,
package sources, dependency consistency (`pip check`), and pytest are separate
steps. Fix the cause and push another commit.

## Preview the documentation locally

From the repository root, create and activate the Conda environment for MkDocs:

```bash
conda env create -f environment-docs.yml
conda activate docs-env
mkdocs serve
```

Open `http://127.0.0.1:8000/` to preview the site. To check the static build,
stop the preview and run `mkdocs build --strict`. MkDocs writes the
generated files to `site/`.

The COS-7 tutorial renders the code and text from
`Python/examples/notebooks/minimal_example.ipynb` and offers a copy with
outputs cleared. Edit that notebook to update both.
The docs build reads the file without running its cells or downloading the CZI.

GitHub Actions runs the same strict build in the **Documentation build** check
for PRs to `main` and after merges. The check also verifies the COS-7 tutorial,
notebook download, and result figure are present in the site output.

## Prepare a backend release

The [backend release workflow](.github/workflows/release-backend.yml) builds
the source archive and wheel, checks their metadata, and installs the wheel in
a clean environment for a smoke test.

To rehearse a build, open **Actions → Build backend release → Run workflow**
and select `main`. Download the resulting `backend-rehearsal-<run-id>`
artifact from the run summary. Use this to check packaging and installation
without creating a release tag.

To prepare a release, update the [package version](Python/lumic/pyproject.toml)
and [changelog](Python/lumic/CHANGELOG.md) in a PR. After merging, tag that
commit with `lumic-v<version>` and push the tag. The workflow verifies that
the tag matches the package version and points to a commit on `main`.
Review the source archive and wheel in the `<tag>-distributions` artifact.

### Publish a backend release to PyPI

Configure the Trusted Publisher for the existing `lumic` project on PyPI
with owner `CCI-GU-Sweden`, repository `lumic`, and workflow filename
`publish-backend-pypi.yml`. Leave the environment field blank.

1. Review the wheel and source archive from a successful `lumic-vX.Y.Z`
   release build or `lumic-vX.Y.ZrcN` candidate build. Copy the numeric
   run ID from the build run URL.
2. Open **Actions → Publish backend release to PyPI → Run workflow**.
   Select `main`, enter the reviewed run ID, and confirm the public upload.
3. After the upload, install the exact version from PyPI in a fresh
   Python 3.12 environment and run the installed backend smoke test.

The [publish workflow](.github/workflows/publish-backend-pypi.yml) checks
that the run ID comes from a successful tag-triggered build and uploads its
reviewed artifacts. The TestPyPI workflow remains in
[workflow-templates](.github/workflow-templates/).

## Upgrading an existing Python 3.9 setup

Keep the old environment until the new one is verified; avoid an in-place
Python upgrade when you still need its notebooks or recorded results.
From the repository root:

```bash
conda env create --name unmixing-env-py312 --file environment.yml
conda activate unmixing-env-py312
python -m pip check
cd Python
python -m pytest --ignore-glob='tests/test_plugin_*.py'
```

To use the new environment in Jupyter, register a separate kernel:

```bash
python -m ipykernel install --user --name unmixing-env-py312 --display-name "unmixing-env-py312"
```

Select `unmixing-env-py312` in the notebook and restart its kernel before
rerunning all cells. An existing `unmixing-env` kernel still points to the
old environment. Python and dependency updates can change numerical details;
check the tests and relevant examples before comparing new results with older
reference runs. Archived evaluation metadata records its original versions.

## Maintainer setup: require tests before merging

The workflow runs tests; a GitHub branch rule enforces their result. Adding
the workflow file alone does not block merging a failing PR. After the initial
CI PR passes and is merged, configure this in the web interface:

1. Open **Settings → Rules → Rulesets → New ruleset → New branch ruleset**.
   This requires admin access or an appropriate custom role. Name the rule
   `main requires tests` and set **Enforcement status: Active**.
2. Under **Target branches**, add an inclusion for the default branch (`main`).
3. Enable **Require a pull request before merging**. Human approval requirements
   are a separate team decision; they are not needed to enforce pytest.
4. Enable **Require status checks to pass**, choose **Add checks**, and select
   **Python tests (unmixing-env)**, both **Backend wheel** checks, and
   **Documentation build**. Select GitHub Actions as the expected source if
   offered. A recent run makes each check available for selection.
5. Enable **Require branches to be up to date before merging**, so changes are
   checked against current `main`. Updating a branch reruns the tests.
6. Keep the bypass list empty if everyone should use checked PRs. Save, then
   confirm the next PR lists the check as required.

Routine changes, including working-document updates, then go through PRs
rather than direct pushes to `main`. The push trigger checks merged commits
and any explicitly permitted bypass push.

If Rulesets are unavailable, use **Settings → Branches → Add branch protection
rule** for `main` with equivalent PR and required-check settings. Availability
depends on repository visibility and GitHub plan. If the check is missing,
confirm the workflow ran recently before selecting it.

See GitHub's [ruleset instructions](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
and [protected branch guide](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Adding image formats

Follow the [image-reader contributor guide](docs/contributing/adding-an-image-reader.md)
and [working reader template](Python/examples/reader_template.py). New readers
inspect headers before pixel decoding, expose the shared structural contract,
and return SpectralImage. Add contract/known-value tests and explicit dispatcher
routing while preserving existing format-specific entry points.
