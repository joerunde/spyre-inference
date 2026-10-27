#!/usr/bin/env bash
# Rewrite /opt/ibm/spyre RPATH entries in a tree unpacked with rpm2cpio so they point
# into the tree instead.
#
#   bash scripts/rewrite-spyre-rpaths.sh PREFIX
#
# DT_RPATH is searched before LD_LIBRARY_PATH, so /opt/ibm/spyre entries would bind
# the system libraries. Rewritten entries no longer match, so running it again is a
# no-op. readelf does the detection, so a run with nothing to rewrite does not need
# patchelf.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 PREFIX" >&2
    exit 2
fi
PREFIX="$1"
BASE_INSTALL_DIR="${PREFIX}/opt/ibm/spyre"

PATCHELF=""
while IFS= read -r -d '' f; do
    [[ "$(head -c4 "$f")" == $'\x7fELF' ]] || continue
    rpath="$(readelf -d "$f" 2>/dev/null | sed -n 's/.*Library r\(un\)\?path: \[\(.*\)\]$/\2/p')"
    [[ ":$rpath" == *:/opt/ibm/spyre* ]] || continue
    if [[ -z "$PATCHELF" ]]; then
        PATCHELF="$(command -v patchelf || uvx --from patchelf bash -c 'command -v patchelf')" || {
            echo "patchelf not found and could not be fetched with uvx" >&2
            exit 1
        }
    fi
    new_rpath=":${rpath}"
    new_rpath="${new_rpath//:\/opt\/ibm\/spyre/:"${BASE_INSTALL_DIR}"}"
    "$PATCHELF" --force-rpath --set-rpath "${new_rpath#:}" "$f"
    echo "rewrote RPATH of ${f#"$PREFIX"/}: ${new_rpath#:}"
done < <(find "$PREFIX" -type f -print0)
