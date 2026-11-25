from pathlib import PurePosixPath
import re
from zipfile import ZipFile

import pytest


@pytest.mark.release
def test_zip_contains_source_direct_run_app_and_report(submission):
    assert submission.is_file(), f"找不到繳交檔 {submission}"
    assert re.fullmatch(r"\d+_[^_]+_作業三\.zip", submission.name)
    with ZipFile(submission) as archive:
        assert archive.testzip() is None
        files = [PurePosixPath(item.filename) for item in archive.infolist()
                 if not item.is_dir() and item.file_size > 0]
    assert any(p.name == "main.py" for p in files), "缺少原始碼入口"
    for module in ("data", "hopfield", "som", "workflows", "gui"):
        assert any(p.parts[-2:] == ("mem_storage_machine", f"{module}.py") for p in files), module
    assert any(p.suffix.lower() == ".exe" for p in files), "缺少可直接啟動的 Windows 執行檔"
    assert any(p.suffix.lower() == ".pdf" and p.name != "類神經網路作業三說明.pdf"
               for p in files), "缺少書面報告 PDF"
