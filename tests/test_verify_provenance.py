"""What a verify report says of bellwether itself: its version, and its commit only from its own checkout.

``git`` answers through a stub, so the rule is tested whatever checkout, if any, the suite runs from.
"""

import subprocess
from pathlib import Path

from bellwether import __version__
from bellwether.verify import report


def test_the_commit_is_given_only_for_bellwethers_own_checkout(monkeypatch):
    package = Path(report.__file__).resolve().parents[1]  # src/bellwether
    answers = {
        "rev-parse --show-toplevel": str(package.parents[1]),
        "rev-parse HEAD": "c" * 40,
        "status --porcelain": " M x",
    }
    monkeypatch.setattr(report, "_git", lambda where, *args: answers[" ".join(args)])
    assert report.bellwether_version() == {"version": __version__, "commit": "c" * 40, "dirty": True}

    answers |= {"rev-parse --show-toplevel": "/another/repository", "status --porcelain": ""}
    assert report.bellwether_version() == {"version": __version__, "commit": None, "dirty": None}

    def no_checkout(where, *args):
        raise subprocess.CalledProcessError(128, ["git", *args])

    monkeypatch.setattr(report, "_git", no_checkout)
    assert report.bellwether_version() == {"version": __version__, "commit": None, "dirty": None}
