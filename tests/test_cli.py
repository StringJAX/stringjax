"""Unit tests for the command-line interface."""

from stringjax.cli import main


def test_versions_command(capsys):
    rc = main(["versions"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "stringjax" in out


def test_doctor_command(capsys):
    rc = main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "environment report" in out


def test_default_is_doctor(capsys):
    rc = main([])
    assert rc == 0
    assert "environment report" in capsys.readouterr().out.lower()
