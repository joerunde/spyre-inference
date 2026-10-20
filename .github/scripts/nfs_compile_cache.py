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

"""Share the Inductor + Spyre kernel compile cache across CI jobs via the NFS mount.

Every cache inside TORCHINDUCTOR_CACHE_DIR that we ship (fxgraph, aotautograd,
PyCodeCache modules, inductor-spyre-cache) is content-addressed and write-once,
so merging the caches of many shards is a plain file union and needs no lock:

    <nfs-root>/<scope>/<arch>/<key>/
        base-<ns>.tar     compacted union of earlier deltas
        base-<ns>.json    names of the deltas folded into that base
        delta-<id>.tar    files one job added on top of what it restored

A job restores the newest base plus the deltas it does not cover, records what
it restored, and at job end writes only the files it added as its own delta
(unique name, atomic rename, so concurrent shards never touch the same file).
One compaction job per run folds the deltas into a new base. Nothing a reader
may still be extracting is deleted until it is older than the grace period.

The key hashes the lockfiles, so any dependency change (torch, torch-spyre,
vLLM, RPMs) starts a fresh epoch rather than reusing entries a keying gap might
let through. Scopes are the uv cache's: main writes `main`, PR #N writes
`pr-<N>` and reads it on top of `main`, the merge queue only reads. Restore and
save are best-effort: on any failure the job compiles cold, never goes red.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfs_uv_cache import read_scopes, write_scope  # noqa: E402

# Bump to start every scope from an empty cache.
KEY_SALT = "nfs-compile-cache-v1"
KEY_LEN = 16

# Per-compile scratch, lock files and host-specific precompiled headers; none is
# a cache entry another job can use.
EXCLUDED_TOP_DIRS = {"inductor-spyre", "locks", "precompiled_headers"}
FAILED_KERNELS = ("inductor-spyre-cache", "failed")


def _log(msg):
    print(msg, flush=True)


def _warn(msg):
    print(f"::warning::{msg}", flush=True)


def compute_key(key_files, extras=(), salt=KEY_SALT):
    digest = hashlib.sha256(salt.encode())
    for extra in extras:
        digest.update(b"\0extra\0" + extra.encode())
    for path in key_files:
        digest.update(b"\0" + os.fsencode(os.path.basename(path)) + b"\0")
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:KEY_LEN]


def _key_dir(nfs_root, scope, arch, key):
    return Path(nfs_root) / scope / arch / key


def _shippable(rel):
    """Whether a path relative to the cache root is a finished, shareable entry."""
    parts = rel.parts
    if parts[0] in EXCLUDED_TOP_DIRS or parts[:2] == FAILED_KERNELS:
        return False
    # Uncommitted kernel dirs (<key>.tmp.<uuid>) and write_atomic temp files.
    return not any(".tmp" in part for part in parts)


def list_files(root):
    root = Path(root)
    if not root.is_dir():
        return set()
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            rel = (Path(dirpath) / name).relative_to(root)
            if _shippable(rel):
                out.add(rel.as_posix())
    return out


def _merged(base):
    """Deltas folded into a base; a missing or torn sidecar means none."""
    try:
        return set(json.loads(base.with_suffix(".json").read_text()))
    except (OSError, ValueError):
        return set()


def layers(key_dir):
    """Tars to extract for one scope: the newest base, then the deltas it lacks."""
    if not key_dir.is_dir():
        return []
    bases = sorted(key_dir.glob("base-*.tar"))
    merged = _merged(bases[-1]) if bases else set()
    deltas = sorted(p for p in key_dir.glob("delta-*.tar") if p.name not in merged)
    return bases[-1:] + deltas


def _extract(tar_path, dest):
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(dest)], check=True)


def _tar_files(src, rel_paths, tar_path):
    listing = tar_path.with_name(tar_path.name + ".list")
    listing.write_bytes(b"\0".join(os.fsencode(p) for p in sorted(rel_paths)) + b"\0")
    try:
        subprocess.run(
            ["tar", "-cf", str(tar_path), "-C", str(src), "--null", "-T", str(listing)],
            check=True,
        )
    finally:
        listing.unlink(missing_ok=True)


def _publish(tmp, target):
    """Rename tmp into place; the temp lives beside target so the rename is atomic."""
    os.replace(tmp, target)
    return target.stat().st_size / (1024 * 1024)


def _write_kv(path, pairs):
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            for key, value in pairs.items():
                handle.write(f"{key}={value}\n")


def _reset_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def cmd_restore(args):
    local = Path(args.local_dir)
    _reset_dir(local)
    start = time.monotonic()
    restored, key = [], ""
    try:
        key = compute_key(args.key_file, args.key_extra)
        # Main first: a PR scope holds only what that PR added on top of it.
        for scope in reversed(read_scopes(args.event_name, args.ref, args.pr_number)):
            for tar in layers(_key_dir(args.nfs_root, scope, args.arch, key)):
                _extract(tar, local)
                restored.append(f"{scope}/{tar.name}")
    except (subprocess.CalledProcessError, OSError) as exc:
        # A half-extracted file is worse than a cold start.
        _warn(f"compile cache restore failed ({type(exc).__name__}: {exc}); starting cold")
        _reset_dir(local)
        restored = []

    files = list_files(local)
    Path(args.manifest).write_text("\n".join(sorted(files)), encoding="utf-8")
    kernels = local / "inductor-spyre-cache"
    n_kernels = sum(1 for p in kernels.iterdir() if p.is_dir()) if kernels.is_dir() else 0
    _write_kv(args.github_env, {"TORCHINDUCTOR_CACHE_DIR": str(local)})
    _write_kv(args.github_output, {"cache-key": key, "restored-files": len(files)})
    elapsed = time.monotonic() - start
    if restored:
        _log(
            f"✅ compile cache: restored {len(files)} files ({n_kernels} Spyre kernels) "
            f"from {len(restored)} tars in {elapsed:.0f}s: {', '.join(restored)}"
        )
    else:
        _log(f"🧊 compile cache MISS key {key} ({args.arch}); compiling cold")
    return 0


def cmd_save(args):
    scope = write_scope(args.event_name, args.ref, args.pr_number)
    if scope is None:
        _log(f"read-only context (event={args.event_name}); skipping compile cache save")
        return 0
    if not args.key:
        _log("restore computed no cache key; skipping compile cache save")
        return 0
    local = Path(args.local_dir)
    try:
        key = args.key
        before = set(Path(args.manifest).read_text(encoding="utf-8").splitlines())
        new = sorted(list_files(local) - before)
        new_kernels = {p.split("/")[1] for p in new if p.startswith("inductor-spyre-cache/")}
        _log(f"compile cache: this job added {len(new)} files ({len(new_kernels)} Spyre kernels)")
        if not new:
            return 0
        key_dir = _key_dir(args.nfs_root, scope, args.arch, key)
        key_dir.mkdir(parents=True, exist_ok=True)
        target = key_dir / f"delta-{args.name}.tar"
        tmp = key_dir / f".delta-{args.name}.{os.getpid()}.tmp"
        try:
            _tar_files(local, new, tmp)
            size = _publish(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        _log(f"✅ saved compile cache delta {scope}/{args.arch}/{key}/{target.name} ({size:.0f}MB)")
    except (subprocess.CalledProcessError, OSError) as exc:
        _warn(f"compile cache save failed ({type(exc).__name__}: {exc}); continuing")
    return 0


def _older_than(path, seconds):
    try:
        return time.time() - path.stat().st_mtime > seconds
    except OSError:
        return False


def _gc(key_dir, grace_s):
    """Drop what the newest base supersedes once it has been visible for grace_s.

    A reader that listed the previous base still needs it and the deltas it
    lacked, so the clock runs from the newest base, not from the files dropped.
    """
    bases = sorted(key_dir.glob("base-*.tar"))
    if not bases or not _older_than(bases[-1], grace_s):
        return
    for old in bases[:-1]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
        _log(f"🧹 evicted superseded {old.name}")
    merged = _merged(bases[-1])
    for delta in key_dir.glob("delta-*.tar"):
        if delta.name in merged:
            delta.unlink(missing_ok=True)


def _newest_mtime(path):
    return max((p.stat().st_mtime for p in path.iterdir()), default=path.stat().st_mtime)


def _evict_epochs(arch_dir, keep):
    """Keep the `keep` most recently written keys of one scope; old keys are dead weight.

    Runs before compaction, which would otherwise make every key look fresh.
    """
    epochs = sorted((p for p in arch_dir.iterdir() if p.is_dir()), key=_newest_mtime, reverse=True)
    for stale in epochs[keep:]:
        shutil.rmtree(stale, ignore_errors=True)
        _log(f"🧹 evicted old compile cache epoch {stale.name}")


def _sweep_pr_scopes(nfs_root, ttl_days):
    cutoff = time.time() - ttl_days * 86400
    for key_dir in Path(nfs_root).glob("pr-*/*/*"):
        with contextlib.suppress(OSError):
            if key_dir.is_dir() and _newest_mtime(key_dir) < cutoff:
                shutil.rmtree(key_dir, ignore_errors=True)
                _log(f"🧹 evicted stale PR compile cache {key_dir.relative_to(nfs_root)}")


def compact_epoch(key_dir, staging, grace_s):
    """Fold the deltas of one key dir into a new base, then GC what it supersedes."""
    todo = layers(key_dir)
    deltas = [p for p in todo if p.name.startswith("delta-")]
    if deltas:
        start = time.monotonic()
        _reset_dir(staging)
        for tar in todo:
            _extract(tar, staging)
        merged = _merged(todo[0]) if todo[0].name.startswith("base-") else set()
        merged |= {p.name for p in deltas}
        stamp = time.time_ns()
        target = key_dir / f"base-{stamp}.tar"
        tmp = key_dir / f".base-{stamp}.tmp"
        # The sidecar lands first: a reader that sees the base without it only
        # extracts the folded deltas again, which is harmless.
        target.with_suffix(".json").write_text(json.dumps(sorted(merged)))
        try:
            _tar_files(staging, list_files(staging), tmp)
            size = _publish(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
            shutil.rmtree(staging, ignore_errors=True)
        _log(
            f"✅ compacted {len(deltas)} deltas into {key_dir.name}/{target.name} "
            f"({size:.0f}MB) in {time.monotonic() - start:.0f}s"
        )
    _gc(key_dir, grace_s)


def cmd_compact(args):
    with contextlib.suppress(OSError):
        _sweep_pr_scopes(args.nfs_root, args.pr_ttl_days)
    scope = write_scope(args.event_name, args.ref, args.pr_number)
    if scope is None:
        _log(f"read-only context (event={args.event_name}); nothing to compact")
        return 0
    # Every arch and key with pending deltas: the keys depend on the image the
    # test legs ran on, which this job need not share.
    for arch_dir in sorted(p for p in (Path(args.nfs_root) / scope).glob("*") if p.is_dir()):
        with contextlib.suppress(OSError):
            _evict_epochs(arch_dir, args.keep_epochs)
        for key_dir in sorted(p for p in arch_dir.iterdir() if p.is_dir()):
            try:
                compact_epoch(key_dir, Path(args.local_dir), args.grace_hours * 3600)
            except (subprocess.CalledProcessError, OSError) as exc:
                _warn(f"compacting {key_dir} failed ({type(exc).__name__}: {exc}); continuing")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p, local_help):
        p.add_argument("--nfs-root", required=True, help="e.g. $STORAGE_1_DIR/.cache/compile")
        p.add_argument("--local-dir", required=True, help=local_help)
        p.add_argument("--event-name", default=os.getenv("GITHUB_EVENT_NAME", ""))
        p.add_argument("--ref", default=os.getenv("GITHUB_REF", ""))
        p.add_argument("--pr-number", default="")

    r = sub.add_parser("restore")
    add_common(r, "the job's TORCHINDUCTOR_CACHE_DIR")
    r.add_argument("--arch", default=platform.machine())
    r.add_argument("--key-file", action="append", required=True, dest="key_file")
    r.add_argument("--key-extra", action="append", default=[], dest="key_extra")
    r.add_argument("--manifest", required=True, help="where to record the restored files")
    r.add_argument("--github-env", default=os.getenv("GITHUB_ENV", ""))
    r.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT", ""))
    r.set_defaults(func=cmd_restore)

    s = sub.add_parser("save")
    add_common(s, "the job's TORCHINDUCTOR_CACHE_DIR")
    s.add_argument("--arch", default=platform.machine())
    s.add_argument("--key", required=True, help="the restore's cache-key output")
    s.add_argument("--manifest", required=True, help="the restore's file list")
    s.add_argument("--name", required=True, help="unique delta id, e.g. <run>-<attempt>-<slot>")
    s.set_defaults(func=cmd_save)

    c = sub.add_parser("compact")
    add_common(c, "scratch dir to merge in")
    c.add_argument("--keep-epochs", type=int, default=2, help="keys to retain per scope/arch")
    c.add_argument("--grace-hours", type=float, default=6)
    c.add_argument("--pr-ttl-days", type=int, default=14)
    c.set_defaults(func=cmd_compact)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
