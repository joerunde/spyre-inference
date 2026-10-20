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

"""Logic tests for the NFS-backed compile cache helper: delta save, layered restore,
compaction, scope isolation and GC, on real tars."""

from __future__ import annotations

import os
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nfs_compile_cache as ncc  # noqa: E402

MAIN = {"event_name": "push", "ref": "refs/heads/main"}
MQ = {"event_name": "merge_group", "ref": "refs/heads/gh-readonly-queue/main/x"}


def _pr(n):
    return {"event_name": "pull_request", "ref": f"refs/pull/{n}/merge", "pr_number": n}


@pytest.fixture
def env(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    return {"nfs": tmp_path / "nfs", "lock": str(lock), "tmp": tmp_path}


def _flags(ctx):
    out = []
    for k, v in ctx.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def _restore(env, job, ctx, extras=()):
    local = env["tmp"] / job
    out = env["tmp"] / f"{job}.out"
    argv = ["restore", "--nfs-root", str(env["nfs"]), "--local-dir", str(local)]
    argv += ["--arch", "x86_64", "--key-file", env["lock"], "--github-output", str(out)]
    argv += ["--manifest", str(env["tmp"] / f"{job}.manifest")]
    for extra in extras:
        argv += ["--key-extra", extra]
    assert ncc.main(argv + _flags(ctx)) == 0
    key = dict(line.split("=", 1) for line in out.read_text().splitlines())["cache-key"]
    return local, key


def _save(env, job, ctx, key):
    argv = ["save", "--nfs-root", str(env["nfs"]), "--local-dir", str(env["tmp"] / job)]
    argv += ["--arch", "x86_64", "--key", key, "--name", job]
    argv += ["--manifest", str(env["tmp"] / f"{job}.manifest")]
    assert ncc.main(argv + _flags(ctx)) == 0


def _compact(env, ctx, **extra):
    argv = ["compact", "--nfs-root", str(env["nfs"])]
    argv += ["--local-dir", str(env["tmp"] / "staging")]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    assert ncc.main(argv + _flags(ctx)) == 0


def _put(root, rel, text="x"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _job(env, job, ctx, files):
    local, key = _restore(env, job, ctx)
    for rel in files:
        _put(local, rel, rel)
    _save(env, job, ctx, key)
    return key


def test_key_tracks_lockfiles_and_extras(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("a")
    base = ncc.compute_key([str(lock)])
    assert ncc.compute_key([str(lock)], ["image=1"]) != base
    lock.write_text("b")
    assert ncc.compute_key([str(lock)]) != base


@pytest.mark.parametrize(
    "rel,shippable",
    [
        ("fxgraph/ab/abc/entry", True),
        ("aotautograd/abc/entry", True),
        ("ab/cabcdef.py", True),
        ("inductor-spyre-cache/k1/bundle.mlir", True),
        ("inductor-spyre-cache/k1.tmp.0f00/bundle.mlir", False),
        ("inductor-spyre-cache/failed/k1/bundle.mlir", False),
        ("fxgraph/ab/abc/.123.456.tmp", False),
        ("inductor-spyre/abcd_sdsc_x/bundle.mlir", False),
        ("locks/abc.lock", False),
        ("precompiled_headers/x.gch", False),
    ],
)
def test_shippable(rel, shippable):
    assert ncc._shippable(pathlib.PurePosixPath(rel)) is shippable


def test_shards_union_through_deltas_without_compaction(env):
    _job(env, "a", MAIN, ["fxgraph/a", "inductor-spyre-cache/ka/bundle.mlir"])
    _job(env, "b", MAIN, ["fxgraph/b", "locks/b.lock"])
    local, _ = _restore(env, "c", MAIN)
    assert ncc.list_files(local) == {
        "fxgraph/a",
        "fxgraph/b",
        "inductor-spyre-cache/ka/bundle.mlir",
    }


def test_save_ships_only_new_files(env):
    _job(env, "a", MAIN, ["fxgraph/a"])
    local, key = _restore(env, "b", MAIN)
    _put(local, "fxgraph/b")
    _save(env, "b", MAIN, key)
    key_dir = env["nfs"] / "main" / "x86_64" / key
    assert ncc.list_files(_unpack(env, key_dir / "delta-b.tar")) == {"fxgraph/b"}


def _unpack(env, tar):
    dest = env["tmp"] / f"unpack-{tar.name}"
    dest.mkdir()
    ncc._extract(tar, dest)
    return dest


def test_nothing_new_writes_nothing(env):
    key = _job(env, "a", MAIN, ["fxgraph/a"])
    _, key = _restore(env, "b", MAIN)
    _save(env, "b", MAIN, key)
    assert not (env["nfs"] / "main" / "x86_64" / key / "delta-b.tar").exists()


def test_compaction_folds_deltas_and_restore_matches(env):
    key = _job(env, "a", MAIN, ["fxgraph/a"])
    _job(env, "b", MAIN, ["fxgraph/b"])
    _compact(env, MAIN, grace_hours=1)
    key_dir = env["nfs"] / "main" / "x86_64" / key
    assert [p.name for p in ncc.layers(key_dir)] == [sorted(key_dir.glob("base-*.tar"))[-1].name]
    # Folded deltas wait out the grace period for readers that listed them.
    assert len(list(key_dir.glob("delta-*.tar"))) == 2
    _job(env, "c", MAIN, ["fxgraph/c"])
    local, _ = _restore(env, "d", MAIN)
    assert ncc.list_files(local) == {"fxgraph/a", "fxgraph/b", "fxgraph/c"}


def test_gc_after_grace(env):
    key = _job(env, "a", MAIN, ["fxgraph/a"])
    _compact(env, MAIN)
    key_dir = env["nfs"] / "main" / "x86_64" / key
    first = next(key_dir.glob("base-*.tar"))
    _job(env, "b", MAIN, ["fxgraph/b"])
    _compact(env, MAIN)
    old = time.time() - 7 * 3600
    for p in key_dir.iterdir():
        os.utime(p, (old, old))
    _compact(env, MAIN)
    assert not first.exists()
    assert [p.name for p in key_dir.glob("*.tar")] == [ncc.layers(key_dir)[0].name]
    local, _ = _restore(env, "c", MAIN)
    assert ncc.list_files(local) == {"fxgraph/a", "fxgraph/b"}


def test_pr_writes_own_scope_and_reads_main(env):
    _job(env, "main", MAIN, ["fxgraph/m"])
    _job(env, "pr", _pr(7), ["fxgraph/p"])
    assert ncc.list_files(_restore(env, "pr-again", _pr(7))[0]) == {"fxgraph/m", "fxgraph/p"}
    assert ncc.list_files(_restore(env, "other-pr", _pr(8))[0]) == {"fxgraph/m"}
    assert ncc.list_files(_restore(env, "main-again", MAIN)[0]) == {"fxgraph/m"}


def test_merge_queue_reads_main_and_writes_nothing(env):
    _job(env, "main", MAIN, ["fxgraph/m"])
    _job(env, "mq", MQ, ["fxgraph/q"])
    _compact(env, MQ)
    assert ncc.list_files(_restore(env, "after", MAIN)[0]) == {"fxgraph/m"}


def test_new_key_starts_cold(env):
    _job(env, "a", MAIN, ["fxgraph/a"])
    local, _ = _restore(env, "b", MAIN, extras=["image=other"])
    assert ncc.list_files(local) == set()


def test_old_epochs_evicted(env):
    for i in range(3):
        local, key = _restore(env, f"j{i}", MAIN, extras=[f"image={i}"])
        _put(local, f"fxgraph/{i}")
        _save(env, f"j{i}", MAIN, key)
        old = time.time() - (10 - i) * 60
        for p in (env["nfs"] / "main" / "x86_64" / key).iterdir():
            os.utime(p, (old, old))
    _compact(env, MAIN, keep_epochs=2)
    assert len(list((env["nfs"] / "main" / "x86_64").iterdir())) == 2
    assert ncc.list_files(_restore(env, "j0-again", MAIN, extras=["image=0"])[0]) == set()


def test_corrupt_tar_starts_cold(env):
    key = _job(env, "a", MAIN, ["fxgraph/a"])
    (env["nfs"] / "main" / "x86_64" / key / "delta-zz.tar").write_text("not a tar")
    local, _ = _restore(env, "b", MAIN)
    assert ncc.list_files(local) == set()


def test_restore_exports_cache_dir(env, tmp_path):
    genv = tmp_path / "github_env"
    local = tmp_path / "local"
    argv = ["restore", "--nfs-root", str(env["nfs"]), "--local-dir", str(local)]
    argv += [
        "--key-file",
        env["lock"],
        "--manifest",
        str(tmp_path / "m"),
        "--github-env",
        str(genv),
    ]
    ncc.main(argv + _flags(MAIN))
    assert genv.read_text() == f"TORCHINDUCTOR_CACHE_DIR={local}\n"
