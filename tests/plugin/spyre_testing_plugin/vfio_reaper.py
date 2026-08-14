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

"""Free the Spyre accelerator (VFIO) card when a test leaves it claimed.

When a vLLM engine dies abnormally (e.g. an execute_model RPC timeout during a
long Spyre compile), the worker holding the card is orphaned — reparented to
init but still holding the VFIO container (``/dev/vfio/vfio``) and device inode
(``anon_inode:[vfio-device]``) open. The next test's ``torch.spyre.set_device()``
then fails with ``RAS::VFIO::DeviceOpenFail ... "Device or resource busy"``,
cascading through the shard.

So after a failed test we find the holder by fd (a ``/proc/*/fd`` scan, not the
process tree — the holder may be reparented) and SIGKILL it, which frees the
card. The pytest process never opens the device itself, so it is excluded.

A ``/proc`` fd-scan alone is necessary but *not sufficient* as a readiness
signal. vLLM force-kills (SIGKILL) its out-of-process worker at engine shutdown;
the process — and its vfio fds — vanish from ``/proc`` almost immediately, but
the kernel's VFIO device release/reset is **asynchronous**: ``/dev/vfio/<grp>``
keeps returning EBUSY on ``open()`` for a short window (observed ≈0.24 s locally,
up to ≈1.3 s in CI) after the holder is gone. So "no live fd-holder" can report
the card free while it is still resetting, and the next test's
``start_runtime()`` loses the race. ``wait_until_card_free`` therefore also
probes actual openability of the AIU group node(s) to ride out that window.
"""

from __future__ import annotations

import contextlib
import errno
import glob
import os
import signal
import time
from collections.abc import Callable


def spyre_hardware_present() -> bool:
    """True only on a real Spyre host (has /dev/vfio and AIU_WORLD_SIZE set)."""
    if not os.path.isdir("/dev/vfio"):
        return False
    try:
        return int(os.environ.get("AIU_WORLD_SIZE", "0") or 0) > 0
    except ValueError:
        return False


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip() or "<unknown>"
    except OSError:
        return "<unknown>"


def _pids_holding_vfio(exclude_pids: set[int]) -> list[tuple[int, str, str]]:
    """(pid, device, cmdline) for every process holding the Spyre card open,
    found by scanning `/proc/*/fd` for `/dev/vfio/*` or `anon_inode:[vfio-device]`."""
    holders: dict[int, tuple[int, str, str]] = {}
    for fd_path in glob.glob("/proc/[0-9]*/fd/*"):
        pid = int(fd_path.split("/")[2])
        if pid in exclude_pids or pid in holders:
            continue
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue  # fd or process vanished mid-scan — expected race
        if target.startswith("/dev/vfio/") or target == "anon_inode:[vfio-device]":
            holders[pid] = (pid, target, _read_cmdline(pid))
    return list(holders.values())


def _aiu_group_nodes() -> list[str]:
    """VFIO group device nodes (`/dev/vfio/<grp>`) for the AIU card(s) assigned
    to this process.

    Each BDF in ``PCIDEVICE_IBM_COM_AIU_PF`` maps to an IOMMU group via sysfs
    (``/sys/bus/pci/devices/<bdf>/iommu_group`` -> ``/dev/vfio/<grp>``); probing
    those specific nodes keeps the openability check from tripping on an
    unrelated VFIO device on a multi-card host. Falls back to every
    ``/dev/vfio/<n>`` group node when the env var is unset or a BDF can't be
    resolved, so the probe still works on hosts that don't export it."""
    nodes: list[str] = []
    bdfs = os.environ.get("PCIDEVICE_IBM_COM_AIU_PF", "")
    for bdf in (b.strip() for b in bdfs.split(",") if b.strip()):
        try:
            grp = os.path.basename(os.readlink(f"/sys/bus/pci/devices/{bdf}/iommu_group"))
        except OSError:
            continue
        node = f"/dev/vfio/{grp}"
        if os.path.exists(node):
            nodes.append(node)
    if not nodes:
        nodes = sorted(glob.glob("/dev/vfio/[0-9]*"))
    return nodes


def _cards_openable(nodes: list[str]) -> bool:
    """True if every AIU group node can be ``open()``ed right now.

    A group node returns EBUSY while the kernel is still resetting the device
    after its previous holder exited — the async window a bare ``/proc`` fd-scan
    misses because the holder is already reaped. ``open()``+``close()`` of the
    *group* node is side-effect-free: it neither attaches a container nor
    acquires a device fd. Non-EBUSY errors (perms, missing node) mean we can't
    prove the card is busy, so we don't let them block cleanup."""
    for node in nodes:
        try:
            os.close(os.open(node, os.O_RDWR))
        except OSError as e:
            if e.errno == errno.EBUSY:
                return False
    return True


def _self_holds_device(pids: set[int]) -> bool:
    """True if any of `pids` (i.e. the pytest process) holds a live VFIO *device*
    fd (`anon_inode:[vfio-device]`).

    When it does, the card is legitimately in-process-held and will be reused by
    the next in-process test; an openability probe would then spuriously see our
    own card as EBUSY, so callers skip the probe in that regime."""
    for pid in pids:
        for fd_path in glob.glob(f"/proc/{pid}/fd/*"):
            try:
                if os.readlink(fd_path) == "anon_inode:[vfio-device]":
                    return True
            except OSError:
                continue
    return False


def wait_until_card_free(
    exclude_pids: set[int],
    log: Callable[[str], None] = print,
    timeout: float = 10.0,
    poll: float = 0.1,
) -> bool:
    """Poll until the Spyre card is actually free for the next test to open, or
    `timeout` elapses. Returns True once free, False on timeout.

    "Free" means: no process outside `exclude_pids` holds a card fd **and** the
    card is openable again. The second clause is the important one — after vLLM
    force-kills its worker the fd-holder is gone from ``/proc`` while the kernel
    is still resetting the device (EBUSY on ``open()``), so we additionally
    require the AIU group node(s) to open cleanly. That probe is skipped when the
    pytest process itself holds the device fd: there the card is legitimately
    in-process-held for reuse and probing it would only see our own EBUSY.

    Unlike `reap_vfio_holders` this kills nothing — it only waits. Use it as a
    barrier at a test boundary when the previous test's out-of-process engine is
    on its way down but not gone yet.

    A timeout is not fatal: warn and let the caller proceed, so a genuinely
    stuck holder still surfaces as a loud, self-explaining failure in the test
    that actually needs the card rather than aborting the session here."""
    nodes = _aiu_group_nodes()
    start = time.monotonic()
    waited = False
    while True:
        holders = _pids_holding_vfio(exclude_pids)
        if holders:
            reason = ", ".join(f"pid={p} {dev} ({cmd!r})" for p, dev, cmd in holders)
        elif _self_holds_device(exclude_pids) or _cards_openable(nodes):
            if waited:
                log(f"[vfio-reaper] card freed in {time.monotonic() - start:.2f}s")
            return True
        else:
            reason = f"device still resetting (EBUSY on open of {nodes})"
        if time.monotonic() - start >= timeout:
            log(
                f"[vfio-reaper] WARNING: Spyre card still busy after {timeout}s: {reason}; "
                f"later card tests may fail with DeviceOpenFail until it is freed."
            )
            return False
        waited = True
        time.sleep(poll)


def reap_vfio_holders(
    exclude_pids: set[int],
    log: Callable[[str], None] = print,
    timeout: float = 10.0,
    poll: float = 0.1,
) -> None:
    """SIGKILL every process holding a Spyre card fd, then poll until the card is
    free. If it can't be freed, warn and keep going: a best-effort cleanup should
    not abort the whole session (the holder may be an unrelated VFIO device or a
    process we can't kill), and any genuinely card-blocked test still fails loudly
    on its own."""
    holders = _pids_holding_vfio(exclude_pids)
    if not holders:
        return

    for pid, device, cmdline in holders:
        log(f"[vfio-reaper] orphan pid={pid} holding {device} cmd={cmdline!r}; sending SIGKILL")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)

    wait_until_card_free(exclude_pids, log=log, timeout=timeout, poll=poll)
