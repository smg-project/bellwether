"""Build the table of tool schemas ``bellwether import tau2`` reads, by running tau2's own code in a sandbox.

    uv run python scripts/tau2_tools.py [--check]

The runs record no schemas (``tool_defs`` is null): tau2 built them at run time with ``as_tool``, from each tool
function's signature and docstring through pydantic. So the schemas are tau2's output at the release the importer's
prompts come from (``tau2.PROMPT_COMMIT``), built where bellwether runs a vendor's code:

1. **On the host.** tau2's ``src/``, ``data/tau2/domains/`` and ``pdm.lock`` are checked out at that commit (git
   verifies each object by its id). The lock's default group becomes a requirements file with every file's hash.
2. **The image** (``docker/tau2/Dockerfile``): Python 3.13 pinned by digest, the locked packages installed by hash,
   and tau2's source and domain data, unbuilt.
3. **The run**: no network, a read-only root, an unprivileged user, no capabilities, limits on CPU, memory and
   processes. Inside, this file (``--inside``) builds each domain's environment as ``run.py`` does and prints each
   agent kind's schemas: ``default`` (``llm_agent``, ``llm_agent_gt``) is ``get_tools()``; ``solo``
   (``llm_agent_solo``) is the solo-mode environment's ``get_tools()`` and ``get_user_tools()``, then ``done``, built
   as ``LLMSoloAgent.add_stop_tool`` builds it.

The table is written to ``src/bellwether/importers/tau2_tools.json``, with the commit, the image's id and the versions
of Python and pydantic that built it; ``--check`` builds it again and compares the schemas.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "src" / "bellwether" / "importers" / "tau2_tools.json"
DOCKERFILE = ROOT / "docker" / "tau2" / "Dockerfile"
REPOSITORY = "https://github.com/sierra-research/tau2-bench"
COMMIT = "37199f36924c8896f5e048360691f8476cd89ba1"  # tau2.PROMPT_COMMIT; the image needs no bellwether import
CHECKOUT = ("src", "data/tau2/domains", "pdm.lock")
# The domains the runs use, and the agent kinds each needs: only telecom's runs have a solo agent.
DOMAINS = {
    "airline": ("default",),
    "retail": ("default",),
    "telecom": ("default", "solo"),
    "telecom-workflow": ("default", "solo"),
}
STOP_TOKEN = "###STOP###"
ISOLATION = (
    "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m,mode=1777", "--user", "10001:10001",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", "2", "--memory", "4g", "--pids-limit", "256",
)  # fmt: skip


def inside() -> None:
    """Run in the container: print each domain's schemas as tau2 builds them."""
    sys.path.insert(0, "/opt/tau2/repo/src")
    import pydantic
    from tau2.environment.tool import as_tool
    from tau2.registry import registry

    def done() -> str:
        """Call this function when you are done with the task."""
        return STOP_TOKEN

    tools = {}
    for domain, kinds in DOMAINS.items():
        constructor = registry.get_env_constructor(domain)
        entry = {"default": [tool.openai_schema for tool in constructor().get_tools()]}
        if "solo" in kinds:
            environment = constructor(solo_mode=True)
            user = environment.get_user_tools() if environment.user_tools else []
            entry["solo"] = [tool.openai_schema for tool in environment.get_tools() + user] + [
                as_tool(done).openai_schema
            ]
        tools[domain] = entry
    print(json.dumps({"python": sys.version.split()[0], "pydantic": pydantic.VERSION, "tools": tools}))


def requirements(lock: dict) -> str:
    """The lock's default group as a requirements file: each package pinned by version, with every file's hash and
    its environment marker, so pip installs only what the lock holds and checks each download."""
    lines = []
    for package in lock["package"]:
        if "default" not in package.get("groups", []):
            continue
        marker = f" ; {package['marker']}" if package.get("marker") else ""
        hashes = " ".join(f"--hash={entry['hash']}" for entry in package["files"])
        lines.append(f"{package['name']}=={package['version']}{marker} {hashes}")
    return "\n".join(lines) + "\n"


def checkout(target: Path) -> None:
    """tau2's files the image needs, at ``COMMIT``."""
    run = lambda *args: subprocess.run(["git", "-C", str(target), *args], check=True, capture_output=True)  # noqa: E731
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    run("remote", "add", "origin", REPOSITORY)
    run("sparse-checkout", "set", "--no-cone", *CHECKOUT)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    run("checkout", "-q", "FETCH_HEAD")
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    if head.stdout.strip() != COMMIT:
        raise SystemExit(f"checked out {head.stdout.strip()}, not {COMMIT}")


def build() -> dict:
    """The table, built in the sandbox."""
    with tempfile.TemporaryDirectory(prefix="bellwether-tau2-") as tmp:
        context = Path(tmp)
        checkout(context / "repo")
        lock = tomllib.loads((context / "repo" / "pdm.lock").read_text("utf-8"))
        (context / "requirements.txt").write_text(requirements(lock), "utf-8")
        (context / "Dockerfile").write_text(DOCKERFILE.read_text("utf-8"), "utf-8")
        (context / "tau2_tools.py").write_text(Path(__file__).read_text("utf-8"), "utf-8")
        tag = "bellwether-tau2-tools"
        subprocess.run(
            ["docker", "build", "--quiet", "--tag", tag, str(context)], check=True, stdout=subprocess.DEVNULL
        )
    image = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    command = ["docker", "run", "--rm", *ISOLATION, image, "python", "-I", "/opt/tau2/tau2_tools.py", "--inside"]
    found = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return {
        "tau2": COMMIT,
        "image": image,
        "python": found["python"],
        "pydantic": found["pydantic"],
        "tools": found["tools"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="build again and compare the schemas; exit 1 on a difference"
    )
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside:
        inside()
        return 0
    table = build()
    if args.check:
        committed = json.loads(TABLE.read_text("utf-8"))
        if committed["tools"] != table["tools"]:
            print(f"{TABLE}: its schemas differ from a fresh build", file=sys.stderr)
            return 1
        print(f"{TABLE}: its schemas equal a fresh build")
        return 0
    TABLE.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", "utf-8")
    counts = {domain: {kind: len(tools) for kind, tools in kinds.items()} for domain, kinds in table["tools"].items()}
    print(f"{TABLE}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
