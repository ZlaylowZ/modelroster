"""
Decide whether the nightly refresh warrants an automatic data-only release,
and prepare it if so.

Run from the repository root after a fully-green `modelroster update --all`
whose data is already committed. Compares the data files at HEAD against the
last release tag and:

  * releases only on REAL drift — models added/removed, capabilities or
    metadata changed, or a provider present now that had no data at the tag.
    `retrieved_at` timestamp churn alone never releases.
  * refuses to auto-release when anything under src/modelroster (other than
    data/) or pyproject.toml changed since the tag — code and packaging
    changes are released by a human tagging deliberately (MAINTAINERS.md).

On a "yes": bumps the patch version in src/modelroster/_version.py, prepends
a CHANGELOG entry summarising the drift, and prints the decision. The
workflow then commits, tags, and dispatches publish.yml — which still runs
the full test suite and data validation before anything reaches PyPI.

Output (stdout, KEY=VALUE lines for $GITHUB_OUTPUT):
  release=yes|no
  version=<new version>          (when yes)
  reason=<human-readable summary>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "modelroster" / "data"
VERSION_FILE = ROOT / "src" / "modelroster" / "_version.py"
CHANGELOG = ROOT / "CHANGELOG.md"


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def out(**kv: str) -> None:
    for k, v in kv.items():
        print("%s=%s" % (k, v))


def data_at_tag(tag: str, provider: str) -> dict | None:
    try:
        text = sh("git", "show", "%s:src/modelroster/data/%s.json" % (tag, provider))
    except subprocess.CalledProcessError:
        return None
    return json.loads(text)


def summarize(provider: str, d: dict) -> str | None:
    parts = []
    if d.get("first_run"):
        return "%s: new provider (%d models)" % (provider, d["counts"]["current"])
    if d["added_models"]:
        parts.append("+%d model(s)" % len(d["added_models"]))
    if d["removed_models"]:
        parts.append("-%d model(s)" % len(d["removed_models"]))
    if d["changed_capabilities"]:
        parts.append("%d record(s) changed" % len(d["changed_capabilities"]))
    if d["new_snapshots"] or d["removed_snapshots"]:
        parts.append("snapshots %+d" % (len(d["new_snapshots"]) - len(d["removed_snapshots"])))
    return "%s: %s" % (provider, ", ".join(parts)) if parts else None


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from modelroster.validate import diff

    last_tag = sh("git", "describe", "--tags", "--abbrev=0")

    # Code/packaging changes since the tag block the auto-release: a human
    # tags those deliberately. Data, docs, tests and workflows ride along.
    changed = sh("git", "diff", "--name-only", "%s..HEAD" % last_tag).splitlines()
    code = [f for f in changed
            if (f.startswith("src/modelroster/") and not f.startswith("src/modelroster/data/"))
            or f == "pyproject.toml"]
    if code:
        out(release="no", reason="code changes since %s pending a deliberate release: %s"
            % (last_tag, ", ".join(sorted(code)[:5])))
        return 0

    lines = []
    for path in sorted(DATA.glob("*.json")):
        if path.name.endswith((".previous.json", ".drift.json")):
            continue
        provider = path.stem
        current = json.loads(path.read_text("utf-8"))
        previous = data_at_tag(last_tag, provider)
        line = summarize(provider, diff(previous, current))
        if line:
            lines.append(line)
    if not lines:
        out(release="no", reason="no model or capability drift since %s" % last_tag)
        return 0

    version = re.search(r'"([0-9]+)\.([0-9]+)\.([0-9]+)"', VERSION_FILE.read_text())
    if not version:
        out(release="no", reason="could not parse _version.py")
        return 1
    major, minor, patch = (int(x) for x in version.groups())
    if "v%d.%d.%d" % (major, minor, patch) != last_tag:
        out(release="no", reason="version %s does not match last tag %s; release in progress?"
            % (version.group(0), last_tag))
        return 0
    new = "%d.%d.%d" % (major, minor, patch + 1)

    VERSION_FILE.write_text('__version__ = "%s"\n' % new)
    today = sh("date", "-u", "+%Y-%m-%d")
    entry = "## [%s] — %s\n\nData-only automatic release. Drift since %s:\n\n%s\n" % (
        new, today, last_tag, "\n".join("- %s" % l for l in lines))
    text = CHANGELOG.read_text()
    marker = "## [Unreleased]\n"
    if marker not in text:
        out(release="no", reason="CHANGELOG.md has no Unreleased section")
        return 1
    CHANGELOG.write_text(text.replace(marker, marker + "\n" + entry, 1))

    out(release="yes", version=new, reason="; ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
