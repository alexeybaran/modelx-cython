from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def git(*args: str) -> str:
    return run("git", *args)


def require(path: str) -> Path:
    p = ROOT / path
    if not p.exists():
        raise AssertionError(f"missing required path: {path}")
    return p


def main() -> int:
    checks: dict[str, object] = {}

    for path in [
        "GUIDING_PRINCIPLES.md",
        "TARGET_ARCHITECTURE.md",
        "PLAN.md",
        "AGENTS.md",
        "CODEX_DESKTOP_SETUP.md",
        "INITIAL_CODEX_PROMPT.md",
        "TESTING.md",
        "DECISIONS.md",
        "project_refs/STATE.json",
        "project_refs/repo_all.bundle",
        "project_refs/donors/prototype_0.8.2a1/README.md",
        "project_refs/donors/previous_phase2r1b/README.md",
        "project_refs/h09/evidence/cache_map.csv",
        "project_refs/h09/tools/h09_benchmark.py",
        "project_refs/models/adjusted/MYGA_US_S/model/_system.json",
        "project_refs/evidence/LINUX_PREMIGRATION_CI.md",
        "project_refs/h10/H10_NEGATIVE_EVIDENCE.md",
    ]:
        require(path)
    checks["required_paths"] = "PASS"

    expected_refs = {
        "upstream-pristine",
        "core",
        "experiment/helpers-exact-linux-control",
        "feature/helpers",
        "reference/h08-compatible-core",
        "reference/h09-compatible",
    }
    refs = set(git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines())
    missing = sorted(expected_refs - refs)
    if missing:
        raise AssertionError(f"missing git branches: {missing}")
    checks["required_git_refs"] = "PASS"

    compiler_diff = git("diff", "--name-only", "upstream-pristine..core", "--", "modelx_cython")
    if compiler_diff:
        raise AssertionError(f"core compiler differs from pristine upstream:\n{compiler_diff}")
    checks["core_compiler_identical_to_upstream"] = True

    helper_diff = git(
        "diff", "--name-only",
        "experiment/helpers-exact-linux-control..feature/helpers",
        "--", "modelx_cython",
    )
    if helper_diff:
        raise AssertionError(f"rebased helper compiler differs from exact Linux control:\n{helper_diff}")
    checks["feature_helpers_compiler_identical_to_linux_control"] = True

    state = json.loads(require("project_refs/STATE.json").read_text())
    branch_map = {
        "upstream_pristine": "upstream-pristine",
        "core": "core",
        "feature_helpers": "feature/helpers",
        "exact_linux_helper_control": "experiment/helpers-exact-linux-control",
        "reference_h09_compatible": "reference/h09-compatible",
        "reference_h08_compatible_core": "reference/h08-compatible-core",
    }
    for key, branch in branch_map.items():
        actual = git("rev-parse", branch)
        if state.get(key) != actual:
            raise AssertionError(f"STATE.json {key}={state.get(key)!r}, git {branch}={actual}")
    checks["state_git_identities"] = "PASS"

    bundle = require("project_refs/repo_all.bundle")
    out = run("git", "bundle", "verify", str(bundle))
    checks["repo_bundle"] = "PASS"

    for path in [
        "project_refs/vendor/modelx-cython-frozen-source.zip",
        "project_refs/vendor/modelx-main.zip",
        "project_refs/vendor/lifelib-main.zip",
    ]:
        with zipfile.ZipFile(require(path)) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise AssertionError(f"bad ZIP member in {path}: {bad}")
    checks["vendor_zip_crc"] = "PASS"

    junit = require("project_refs/evidence/linux_min_helpers_full_ci.xml")
    root = ET.parse(junit).getroot()
    counts = {"upstream_pass": 0, "upstream_skip": 0, "helper_pass": 0, "fail": 0}
    for tc in root.iter("testcase"):
        cls = tc.attrib.get("classname", "")
        file = tc.attrib.get("file", "")
        helper = "h06_helpers" in cls or "h06_helpers" in file
        children = list(tc)
        skipped = any(c.tag.endswith("skipped") for c in children)
        failed = any(c.tag.endswith("failure") or c.tag.endswith("error") for c in children)
        if failed:
            counts["fail"] += 1
        elif skipped:
            counts["upstream_skip"] += 1
        elif helper:
            counts["helper_pass"] += 1
        else:
            counts["upstream_pass"] += 1
    if counts != {"upstream_pass": 171, "upstream_skip": 7, "helper_pass": 2, "fail": 0}:
        raise AssertionError(f"unexpected Linux pre-migration JUnit counts: {counts}")
    checks["linux_pre_migration_ci_receipt"] = counts

    print(json.dumps({
        "status": "PASS",
        "repo": str(ROOT),
        "core": git("rev-parse", "core"),
        "feature_helpers": git("rev-parse", "feature/helpers"),
        "checks": checks,
        "note": "This verifies package/source identity and retained Linux receipts. It does not replace native-Windows Stage 0 CI or benchmarks.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
        raise
