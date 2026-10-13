"""scripts/add_model_rows.py: rows join the list of checkpoints only when the Hub opens their files to the token."""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("add_model_rows", ROOT / "scripts" / "add_model_rows.py")
add_model_rows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(add_model_rows)

HEADER = (
    "# The checkpoints `bellwether manifests` builds fixtures/*/manifest.toml from.\n"
    "# model\trevision\tdownloads\tday\ttier\n"
)


class Hub:
    """Answers auth_check as the Hub would for a token with access to every model but the ``closed`` ones."""

    def __init__(self, closed: dict[str, Exception]) -> None:
        self.closed = closed
        self.asked: list[str] = []

    def auth_check(self, model: str) -> None:
        self.asked.append(model)
        if model in self.closed:
            raise self.closed[model]


def gated(model: str) -> Exception:
    import httpx
    from huggingface_hub.errors import GatedRepoError

    request = httpx.Request("GET", f"https://huggingface.co/api/models/{model}/auth-check")
    return GatedRepoError(f"{model} is gated", response=httpx.Response(403, request=request))


def test_open_rows_join_the_list_in_model_order_without_case(tmp_path):
    listed = tmp_path / "models.tsv"
    listed.write_text(HEADER + "acme/Zeta\tz1\t5\t2026-10-08\t3\nacme/beta\tb1\t9\t2026-10-08\t2\n")
    rows = "acme/Alpha\ta1\t7\t2026-10-08\t2\nacme/gamma\tg1\t-\t-\t3\n"

    added, left_out = add_model_rows.add_rows(listed, rows, Hub({}))

    assert added == ["acme/Alpha", "acme/gamma"]
    assert left_out == []
    assert listed.read_text() == HEADER + (
        "acme/Alpha\ta1\t7\t2026-10-08\t2\n"
        "acme/beta\tb1\t9\t2026-10-08\t2\n"
        "acme/gamma\tg1\t-\t-\t3\n"
        "acme/Zeta\tz1\t5\t2026-10-08\t3\n"
    )


def test_a_row_the_token_cannot_open_is_left_out_with_why(tmp_path):
    listed = tmp_path / "models.tsv"
    listed.write_text(HEADER)

    rows = "acme/Open\to1\t1\t2026-10-08\t3\nacme/Gated\tg1\t2\t2026-10-08\t3\n"

    added, left_out = add_model_rows.add_rows(listed, rows, Hub({"acme/Gated": gated("acme/Gated")}))

    assert added == ["acme/Open"]
    assert left_out == [("acme/Gated", "gated: the token's account has not been given access")]
    assert "acme/Gated" not in listed.read_text()


def test_a_listed_model_keeps_its_row_and_the_hub_is_not_asked(tmp_path):
    listed = tmp_path / "models.tsv"
    listed.write_text(HEADER + "acme/Kept\told\t5\t2026-10-01\t2\n")
    hub = Hub({})

    added, left_out = add_model_rows.add_rows(listed, "acme/Kept\tnew\t9\t2026-10-08\t3\n", hub)

    assert (added, left_out, hub.asked) == ([], [("acme/Kept", "listed already")], [])
    assert listed.read_text() == HEADER + "acme/Kept\told\t5\t2026-10-01\t2\n"


@pytest.mark.parametrize("row", ["acme/Short\tr1\t5", "acme/Long\tr1\t5\t2026-10-08\t3\textra"])
def test_a_row_without_five_fields_is_left_out(tmp_path, row):
    listed = tmp_path / "models.tsv"
    listed.write_text(HEADER)

    added, left_out = add_model_rows.add_rows(listed, row + "\n", Hub({}))

    assert added == [] and len(left_out) == 1 and "fields, not 5" in left_out[0][1]
    assert listed.read_text() == HEADER
