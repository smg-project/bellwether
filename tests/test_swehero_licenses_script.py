import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIT = "Permission is hereby granted, free of charge, to any person obtaining a copy of this software"
BEFORE, AFTER = "a" * 40, "b" * 40  # commits before and after the repository added its license file


@pytest.fixture
def script(monkeypatch):
    """``scripts/swehero_licenses.py``, with GitHub answering for a repository that added its license file late."""
    spec = importlib.util.spec_from_file_location("swehero_licenses", ROOT / "scripts" / "swehero_licenses.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    readme = {"path": "README.md", "type": "blob", "mode": "100644"}
    trees = {BEFORE: [readme], AFTER: [readme, {"path": "LICENSE", "type": "blob", "mode": "100644"}]}
    pulls = {"7": BEFORE, "9": BEFORE, "3400": AFTER}

    def gh(path, cache):
        parent, _, key = path.rpartition("/")
        if parent.endswith("/pulls"):
            return {"base": {"sha": pulls[key]}} if key in pulls else None
        return {"tree": trees[key]} if key in trees else None

    monkeypatch.setattr(module, "gh", gh)
    monkeypatch.setattr(module, "raw", lambda repository, commit, path, cache: MIT.encode())
    return module


def test_a_repository_whose_first_row_predates_its_license_file_takes_the_newest_pull_requests(script, tmp_path):
    found = script.survey("acme/web", [("pull", "7")], tmp_path, newest="3400")
    assert (found["commit"], found["from"], found["license"]) == (AFTER, "pull 3400", "MIT")
    assert "left_out" not in found


def test_a_repository_with_no_license_file_at_its_newest_pull_request_either_is_left_out(script, tmp_path):
    found = script.survey("acme/web", [("pull", "7")], tmp_path, newest="9")
    assert (found["commit"], found["left_out"]) == (BEFORE, "no license file at the root")
