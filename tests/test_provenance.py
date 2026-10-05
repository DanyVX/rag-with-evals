from pathlib import Path

from ragx.provenance import collect_environment


def test_environment_provenance_is_non_sensitive(tmp_path: Path) -> None:
    provenance = collect_environment(tmp_path)

    assert provenance["git_revision"] is None
    assert provenance["python_version"]
    assert "api_key" not in provenance
