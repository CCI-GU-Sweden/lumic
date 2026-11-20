"""Publish the tracked COS-7 notebook as runnable code and a download."""

import json
from pathlib import Path
import re

from mkdocs.structure.files import File


NOTEBOOK = Path(__file__).resolve().parents[1] / "examples/notebooks/minimal_example.ipynb"
DOWNLOAD = "backend/minimal_example.ipynb"
TUTORIAL = "backend/cos7-workflow.md"


def on_files(files, *, config):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # Keep the download portable and avoid publishing local execution outputs.
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    files.append(File.generated(config, DOWNLOAD, content=content))
    return files


def on_page_markdown(markdown, *, page, files, **kwargs):
    if page.file.src_uri != TUTORIAL:
        return markdown

    notebook = json.loads(files.get_file_from_path(DOWNLOAD).content_string)
    blocks = [markdown.rstrip()]
    for cell in notebook["cells"]:
        source = "".join(cell["source"]).rstrip()
        if not source:
            continue
        if cell["cell_type"] == "markdown":
            # Nest notebook headings under the tutorial's page title.
            blocks.append(re.sub(r"^(#{1,5})(?= )", r"#\1", source, flags=re.MULTILINE))
        elif cell["cell_type"] == "code":
            blocks.append(f"```python\n{source}\n```")
    return "\n\n".join(blocks) + "\n"
