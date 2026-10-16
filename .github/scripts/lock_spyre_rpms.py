#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regenerate spyre-rpms.lock from the Spyre RPM bundle set in pyproject.toml.

A bundle is a published set of RPMs built against each other, in
<ARTIFACTORY_BUNDLE_PATH>/<arch>/prod/<bundle>/ next to a build_hashes.json
naming each component's commit. CI keeps downloading from ARTIFACTORY_RPM_PATH,
so each bundled RPM must have a bit-exact (SHA-1) copy there.

  python3 .github/scripts/lock_spyre_rpms.py

Bundles do not ship the -devel packages yet: each one in [tool.spyre-rpms.devel]
takes the version of the bundled package it was built with. ppc64le and s390x
pin the x86_64 commits with the build number wildcarded.

Env: ARTIFACTORY_BASE_URL (or ARTIFACTORY_URL), ARTIFACTORY_RPM_PATH (or
ARTIFACTORY_RPM_REPO), ARTIFACTORY_BUNDLE_PATH (<repo>/<subpath>), ARTIFACTORY_TOKEN.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from resolve_rpms import ARCHES, _env, _match_pattern  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ARCH = "x86_64"
# `+<count>.<sha7>_<build>` ends every version this tool can pin.
COMMIT_BUILD = re.compile(r"\.([0-9a-f]{7,40})_(\d+)$")


class LockError(Exception):
    pass


class Artifactory:
    def __init__(self, base_url, token):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(self, path, data=None):
        req = urllib.request.Request(
            f"{self.base_url}/artifactory/{path}",
            data=data,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "text/plain"},
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)

    def get_json(self, path):
        return self._request(path)

    def listing(self, repo, path):
        """{filename: sha1} of the RPMs directly in <repo>/<path>."""
        query = (
            'items.find({"repo":%s,"path":%s,"name":{"$match":"*.rpm"}})'
            '.include("repo","path","name","actual_sha1")' % (json.dumps(repo), json.dumps(path))
        )
        results = self._request("api/search/aql", query.encode())["results"]
        return {r["name"]: r["actual_sha1"] for r in results}


def load_config(pyproject):
    with open(pyproject, "rb") as f:
        cfg = tomllib.load(f).get("tool", {}).get("spyre-rpms")
    if not cfg or "bundle" not in cfg or "packages" not in cfg:
        raise LockError(f"{pyproject} needs [tool.spyre-rpms] with `bundle` and `packages`")
    unknown = set(cfg.get("devel", {})) - set(cfg["packages"])
    if unknown:
        raise LockError(
            f"[tool.spyre-rpms.devel] names packages not in `packages`: {sorted(unknown)}"
        )
    return cfg


def _version(name, filename):
    prefix, suffix = f"{name}-", f".el10.{BUNDLE_ARCH}.rpm"
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        return None
    version = filename[len(prefix) : -len(suffix)]
    # `ibm-flex-` also prefixes `ibm-flex-test-2.0.0...`; a version starts with a digit.
    return version if version[:1].isdigit() else None


def _bundled_version(name, bundled, commits):
    """The one bundled build of `name` at a commit build_hashes.json names.

    Re-published bundles keep stale builds next to the current one, so the
    commit, not the folder, decides.
    """
    found = []
    for filename in bundled:
        version = _version(name, filename)
        m = version and COMMIT_BUILD.search(version)
        if m and any(c.startswith(m.group(1)) for c in commits):
            found.append(version)
    if len(found) != 1:
        raise LockError(
            f"{name}: expected one build at a bundled commit, found {len(found)}: {sorted(found)}"
        )
    return found[0]


def resolve(cfg, client, bundle_path, rpm_repo):
    """(build_hashes, {name: x86_64 version}) for the configured bundle, verified."""
    bundle = cfg["bundle"]
    repo, _, subpath = bundle_path.strip("/").partition("/")
    folder = "/".join(p for p in (subpath, BUNDLE_ARCH, "prod", bundle) if p)
    try:
        build_hashes = client.get_json(f"{repo}/{folder}/build_hashes.json")
    except urllib.error.HTTPError as e:
        raise LockError(f"no build_hashes.json for bundle {bundle} ({repo}/{folder}): {e}") from e
    commits = [v for v in build_hashes.values() if re.fullmatch(r"[0-9a-f]{40}", str(v))]
    bundled = client.listing(repo, folder)
    published = {arch: client.listing(rpm_repo, arch) for arch in ARCHES}
    devel = cfg.get("devel", {})

    errors = []
    versions = {}
    for name in cfg["packages"]:
        if name in devel:
            continue
        try:
            versions[name] = _bundled_version(name, bundled, commits)
        except LockError as e:
            errors.append(str(e))
            continue
        filename = _match_pattern(name, versions[name], BUNDLE_ARCH)
        sha1 = published[BUNDLE_ARCH].get(filename)
        if sha1 != bundled[filename]:
            what = "missing" if sha1 is None else f"sha1 {sha1}, bundle has {bundled[filename]}"
            errors.append(f"{name}: {rpm_repo}/{BUNDLE_ARCH}/{filename} {what}")
    for name, base in devel.items():
        if base not in versions:
            errors.append(f"{name}: its base package {base} did not resolve")
            continue
        versions[name] = versions[base]
        filename = _match_pattern(name, versions[name], BUNDLE_ARCH)
        if filename not in published[BUNDLE_ARCH]:
            errors.append(f"{name}: {rpm_repo}/{BUNDLE_ARCH}/{filename} missing")
    for arch in ARCHES:
        if arch == BUNDLE_ARCH:
            continue
        for name, version in versions.items():
            pattern = _match_pattern(name, wildcard(version), arch)
            if not fnmatch.filter(published[arch], pattern):
                errors.append(f"{name}: nothing matches {rpm_repo}/{arch}/{pattern}")
    if errors:
        raise LockError("\n".join(errors))
    return build_hashes, {name: versions[name] for name in cfg["packages"]}


def wildcard(version):
    m = COMMIT_BUILD.search(version)
    return f"{version[: m.start(2)]}*"


def _key(k):
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def render(bundle, build_hashes, versions):
    width = max(map(len, versions))

    def table(header, fmt):
        rows = [
            f"{name:<{width}} = {{ version = {json.dumps(fmt(v))} }}"
            for name, v in versions.items()
        ]
        return "\n".join([header, *rows])

    hashes = "\n".join(f"{_key(k)} = {json.dumps(str(v))}" for k, v in build_hashes.items())
    return f"""\
# Generated from the Spyre RPM bundle below by .github/scripts/lock_spyre_rpms.py.
# Change [tool.spyre-rpms] in pyproject.toml and re-run it instead of editing.
#
# x86_64 is pinned to the bundle's exact builds. Power (ppc64le) and Z (s390x)
# pin the same commits with the build number wildcarded: builds differ across
# platforms.

version = 2

[defaults]
# Artifactory subdir (<repo>/<tree>/<arch>/); "" is the prod/base tree.
tree = ""

[bundle]
id = {json.dumps(bundle)}

[bundle.build_hashes]
{hashes}

# x86_64
{table("[packages]", lambda v: v)}

# Power
{table("[overrides.ppc64le]", wildcard)}

# Z
{table("[overrides.s390x]", wildcard)}
"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pyproject", default=REPO_ROOT / "pyproject.toml")
    p.add_argument("--lock", default=REPO_ROOT / "spyre-rpms.lock")
    args = p.parse_args()

    client = Artifactory(_env("ARTIFACTORY_BASE_URL", "ARTIFACTORY_URL"), _env("ARTIFACTORY_TOKEN"))
    try:
        cfg = load_config(args.pyproject)
        build_hashes, versions = resolve(
            cfg,
            client,
            _env("ARTIFACTORY_BUNDLE_PATH"),
            _env("ARTIFACTORY_RPM_PATH", "ARTIFACTORY_RPM_REPO"),
        )
    except LockError as e:
        sys.exit(f"{e}\n{args.lock} left unchanged")
    Path(args.lock).write_text(render(cfg["bundle"], build_hashes, versions))
    for name, version in versions.items():
        print(f"{name:<{max(map(len, versions))}} {version}")
    print(f"wrote {args.lock} from bundle {cfg['bundle']}")


if __name__ == "__main__":
    main()
