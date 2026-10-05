from typer.testing import CliRunner

from ragx.cli import app


def test_doctor_reports_safe_configuration() -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "provider=mock" in result.output
    assert "provider_spend_cap_usd=0" in result.output


def test_benchmark_is_not_silently_implemented() -> None:
    result = CliRunner().invoke(app, ["benchmark"])

    assert result.exit_code == 2
    assert "introduced with the evaluation milestone" in result.output
