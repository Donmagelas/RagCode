from pathlib import Path

import pytest

from app.tools.file_tools import edit_file, write_file


def test_edit_file_replaces_old_string_when_it_appears_once(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("before\nold()\nafter\n", encoding="utf-8")

    result = edit_file(workspace=tmp_path, path="main.py", old_string="old()", new_string="new()")

    assert result == {"path": "main.py", "changed": True}
    assert target.read_text(encoding="utf-8") == "before\nnew()\nafter\n"


def test_edit_file_rejects_missing_old_string(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("print('hello')", encoding="utf-8")

    with pytest.raises(ValueError, match="0 times"):
        edit_file(workspace=tmp_path, path="main.py", old_string="missing", new_string="new")


def test_edit_file_rejects_repeated_old_string(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("old\nold\n", encoding="utf-8")

    with pytest.raises(ValueError, match="2 times"):
        edit_file(workspace=tmp_path, path="main.py", old_string="old", new_string="new")


def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    result = write_file(workspace=tmp_path, path="src/main.py", content="print('ok')")

    assert result == {"path": "src/main.py", "changed": True}
    assert (tmp_path / "src" / "main.py").read_text(encoding="utf-8") == "print('ok')"
