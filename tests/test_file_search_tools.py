from pathlib import Path

import pytest

from app.tools.file_tools import read_file
from app.tools.search_tools import glob_search, grep_search


def test_read_file_rejects_path_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside workspace"):
        read_file(workspace=tmp_path, path=outside)


def test_read_file_returns_utf8_text(tmp_path: Path) -> None:
    target = tmp_path / "src" / "main.py"
    target.parent.mkdir()
    target.write_text("print('hello')", encoding="utf-8")

    result = read_file(workspace=tmp_path, path="src/main.py")

    assert result["path"] == "src/main.py"
    assert result["content"] == "print('hello')"


def test_glob_search_returns_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "main.txt").write_text("", encoding="utf-8")

    assert glob_search(workspace=tmp_path, pattern="**/*.py") == ["src/main.py"]


def test_grep_search_returns_matching_lines(tmp_path: Path) -> None:
    target = tmp_path / "src" / "main.py"
    target.parent.mkdir()
    target.write_text("alpha\nneedle here\n", encoding="utf-8")

    result = grep_search(workspace=tmp_path, pattern="needle")

    assert result == [{"path": "src/main.py", "line": 2, "text": "needle here"}]
