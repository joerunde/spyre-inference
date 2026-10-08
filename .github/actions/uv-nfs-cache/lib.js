// Shared shim for the uv-nfs-cache action's main (restore) and post (save)
// scripts. All real logic lives in ../../scripts/nfs_uv_cache.py; this file
// only parses action inputs, resolves paths, and shells out to it -- so there
// is nothing to bundle and no node_modules.
'use strict';

const { spawnSync } = require('child_process');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');

// Mirror @actions/core's getInput name mangling (spaces -> _, uppercased).
function getInput(name, def = '') {
  const key = 'INPUT_' + name.replace(/ /g, '_').toUpperCase();
  const v = process.env[key];
  return v === undefined || v === '' ? def : v;
}

// Write to GITHUB_STATE with the heredoc syntax the runner expects, so values
// with newlines (the key-files list) round-trip into STATE_* for the post step.
function saveState(name, value) {
  const file = process.env.GITHUB_STATE;
  if (!file) {
    return;
  }
  const delimiter = `ghadelimiter_${crypto.randomUUID()}`;
  fs.appendFileSync(file, `${name}<<${delimiter}\n${value}\n${delimiter}\n`);
}

const getState = (name) => process.env['STATE_' + name] || '';
const warn = (msg) => console.log(`::warning::${msg}`);

const helperPath = () => path.join(__dirname, '..', '..', 'scripts', 'nfs_uv_cache.py');

// Returns the NFS cache root, or null when there is no shared mount to use.
// The /storage-1 mount is shared across torch-spyre org repos, so the cache
// must be repo-scoped. STORAGE_1_DIR is already /storage-1/<repo> by convention,
// but don't trust that: append the GitHub-assigned repo name unless the base
// already ends with it, so a bare-mount STORAGE_1_DIR still can't collide.
function resolveRoot() {
  const override = getInput('storage-root');
  if (override) {
    return override;
  }
  const base = process.env.STORAGE_1_DIR;
  if (!base) {
    return null;
  }
  const repo = (process.env.GITHUB_REPOSITORY || '').split('/').pop();
  const scoped = repo && path.basename(base) !== repo ? path.join(base, repo) : base;
  return path.join(scoped, '.cache', 'uv');
}

const resolveLocal = () =>
  getInput('local-dir') || path.join(process.env.RUNNER_TEMP || '/tmp', 'uv-cache');

// Resolve the lockfiles against the checkout so key computation is independent
// of the step's working directory.
function keyFileArgs(raw) {
  const text = raw !== undefined ? raw : getInput('key-files', 'uv.lock\nspyre-rpms.lock');
  const ws = process.env.GITHUB_WORKSPACE || process.cwd();
  const args = [];
  for (const line of text.split('\n')) {
    const f = line.trim();
    if (f) {
      args.push('--key-file', path.join(ws, f));
    }
  }
  return args;
}

// Run the python helper, streaming its output to the job log. Best-effort: a
// missing interpreter or a non-zero exit warns but never fails the step.
function runHelper(args) {
  const res = spawnSync('python3', ['-I', helperPath(), ...args], {
    stdio: 'inherit',
    env: process.env,
  });
  if (res.error) {
    warn(`uv cache helper could not run: ${res.error.message}`);
  }
}

module.exports = {
  getInput,
  saveState,
  getState,
  warn,
  resolveRoot,
  resolveLocal,
  keyFileArgs,
  runHelper,
};
