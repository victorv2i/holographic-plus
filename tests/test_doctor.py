from __future__ import annotations

import json
import os
import tempfile

import pytest

from enfold.bootstrap import main as enfold_main
from enfold.doctor import main, run_self_test


def test_doctor_self_test_writes_recalls_and_returns_evidence():
    report = run_self_test()

    assert report["ok"] is True
    assert report["retrieval"] == "local-lexical"
    assert report["write"] == "pass"
    assert report["recall"] == "pass"
    assert report["evidence"] == "pass"
    assert isinstance(report["fact_id"], int)
    assert report["evidence_count"] >= 1
    assert report["output_truncated"] is False
    assert "dark mode" in report["content"]
    assert report["diagnosis"]


def test_doctor_main_prints_pass_report_and_exits_zero(capsys):
    assert main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["retrieval"] == "local-lexical"
    assert report["network"] == "disabled"


def test_enfold_doctor_is_on_the_product_cli_and_reports_pass(capsys):
    result = enfold_main(["doctor"])
    captured = capsys.readouterr()
    assert result == 0
    report = json.loads(captured.out)
    assert report["ok"] is True
    assert report["write"] == "pass"
    assert report["recall"] == "pass"
    assert report["evidence"] == "pass"
    assert isinstance(report["fact_id"], int)
    assert report["diagnosis"]


def test_doctor_uses_a_bindable_socket_when_temp_root_is_too_long(
    tmp_path, monkeypatch
):
    long_tmp = tmp_path
    while len(os.fsencode(os.fspath(long_tmp / "enfold-doctor-xxxx" / "enfold.sock"))) <= 107:
        long_tmp = long_tmp / ("pad-" + "x" * 40)
    long_tmp.mkdir(parents=True)
    os.chmod(long_tmp, 0o700)
    real_temporary_directory = tempfile.TemporaryDirectory

    def forced_long_temporary_directory(*args, **kwargs):
        kwargs["dir"] = str(long_tmp)
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(
        "enfold.doctor.tempfile.TemporaryDirectory",
        forced_long_temporary_directory,
    )
    report = run_self_test()
    assert report["ok"] is True
    assert report["write"] == "pass"
    assert report["recall"] == "pass"
    assert report["evidence"] == "pass"


def test_enfold_help_lists_doctor(capsys):
    with pytest.raises(SystemExit) as exited:
        enfold_main(["--help"])
    assert exited.value.code == 0
    assert "doctor" in capsys.readouterr().out
