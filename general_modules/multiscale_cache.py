"""On-disk cache for per-sample multiscale hierarchies and positional features.

Why this exists
---------------
Building a sample's coarsening hierarchy runs Farthest-Point-Sampling, an
``O(N*k)`` Python loop in :func:`model.coarsening._fps_euclidean` that costs
~1-2 s/sample at 100k nodes. The old design cached each hierarchy in RAM *per
DataLoader worker*, costing ``~12 MB * num_workers * num_jobs`` — which, run as
8 concurrent single-GPU jobs (32 workers) over a 100k-node mesh, exceeds 1 TB
and OOMs the box. The positional-feature (RWPE) cache rode the same per-worker
cap, doubling the blow-up.

This module precomputes every sample's hierarchy *and* positional features
**once** into a single shared HDF5 file. Workers then stream entries from disk
(the OS page cache keeps reads in the ~ms range), so per-worker RAM stays flat
and FPS never re-runs. When several independent training jobs start at once, an
exclusive lock file ensures exactly one builds while the rest wait.

Layout (HDF5)
-------------
    root.attrs['signature']       json — coarsening + positional config + dataset id
    root.attrs['format_version']  int
    root['sample_ids']            int64 [S]  — every cached sample id
    root[str(sid)]
        .attrs['num_levels']      int  — actual hierarchy depth (<= multiscale_levels)
        ['x_pos']                 f32  [N, P]  (only if positional_features > 0)
        ['L{l}_ftc']              int  [N_l]
        ['L{l}_c_ei']             int64 [2, E_l]
        ['L{l}_seeds']            int64 [n_c]      (only if entry has seeds)
        ['L{l}_up_ei']            int64 [2, E_up]  (only if entry has up_ei)
        .attrs['L{l}_n_c']        int
        .attrs['L{l}_mode']       str
        .attrs['L{l}_has_seeds']  bool
        .attrs['L{l}_has_up']     bool
"""

import json
import multiprocessing as mp
import os
import time

import h5py
import numpy as np

from general_modules.multiscale_helpers import build_multiscale_hierarchy

FORMAT_VERSION = 1

# Lock files older than this (seconds) are treated as stale (builder crashed).
_STALE_LOCK_SECONDS = 6 * 3600


# ---------------------------------------------------------------------------
# Signature / paths
# ---------------------------------------------------------------------------

def _coarse_params(config: dict) -> dict:
    """Coarsening parameters that fully determine the hierarchy topology."""
    levels = int(config.get('multiscale_levels', 1))

    raw_ct = config.get('coarsening_type', 'bfs')
    if isinstance(raw_ct, list):
        types = [str(t).strip().lower() for t in raw_ct]
    else:
        types = [str(raw_ct).strip().lower()]
    if len(types) == 1 and levels > 1:
        types = types * levels

    raw_vc = config.get('voronoi_clusters', None)
    if raw_vc is None:
        clusters = [0] * levels
    elif isinstance(raw_vc, list):
        clusters = [int(v) for v in raw_vc]
    else:
        clusters = [int(raw_vc)]
    if len(clusters) == 1 and levels > 1:
        clusters = clusters * levels

    return {
        'levels': levels,
        'types': types,
        'clusters': clusters,
    }


def _pos_params(config: dict) -> dict:
    return {'num': int(config.get('positional_features', 0))}


def _signature(h5_file: str, coarse_params: dict, pos_params: dict) -> dict:
    """Signature invalidating the cache when coarsening/positional config or the
    source dataset file changes."""
    try:
        st = os.stat(h5_file)
        dataset_id = {'size': st.st_size, 'mtime': int(st.st_mtime)}
    except OSError:
        dataset_id = {'size': -1, 'mtime': -1}
    return {
        'format_version': FORMAT_VERSION,
        'dataset': os.path.basename(h5_file),
        'dataset_id': dataset_id,
        'coarse': coarse_params,
        'pos': pos_params,
    }


def cache_path_for(h5_file: str, config: dict, signature: dict) -> str:
    """Derive the cache file path. Different configs hash to different files so
    they coexist. Defaults next to the dataset; override with
    ``hierarchy_cache_dir``."""
    import hashlib
    sig_json = json.dumps(signature, sort_keys=True)
    digest = hashlib.sha1(sig_json.encode()).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(h5_file))[0]
    out_dir = config.get('hierarchy_cache_dir') or os.path.dirname(os.path.abspath(h5_file))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{stem}.mscache.{digest}.h5')


# ---------------------------------------------------------------------------
# HDF5 read/write of a single hierarchy entry
# ---------------------------------------------------------------------------

def _write_entry(root: h5py.Group, sid: int, hierarchy, x_pos) -> None:
    g = root.create_group(str(sid))
    g.attrs['num_levels'] = len(hierarchy)
    if x_pos is not None:
        g.create_dataset('x_pos', data=np.asarray(x_pos, dtype=np.float32))
    for l, entry in enumerate(hierarchy):
        g.create_dataset(f'L{l}_ftc', data=np.asarray(entry['ftc']))
        g.create_dataset(f'L{l}_c_ei', data=np.asarray(entry['c_ei'], dtype=np.int64))
        g.attrs[f'L{l}_n_c'] = int(entry['n_c'])
        g.attrs[f'L{l}_mode'] = str(entry.get('mode', 'centroid'))
        seeds = entry.get('seeds')
        g.attrs[f'L{l}_has_seeds'] = seeds is not None
        if seeds is not None:
            g.create_dataset(f'L{l}_seeds', data=np.asarray(seeds, dtype=np.int64))
        has_up = 'up_ei' in entry
        g.attrs[f'L{l}_has_up'] = has_up
        if has_up:
            g.create_dataset(f'L{l}_up_ei', data=np.asarray(entry['up_ei'], dtype=np.int64))


def _read_entry(g: h5py.Group):
    """Return (hierarchy_list, x_pos_or_None) reconstructed from a sample group."""
    num_levels = int(g.attrs['num_levels'])
    hierarchy = []
    for l in range(num_levels):
        entry = {
            'ftc': g[f'L{l}_ftc'][:],
            'c_ei': g[f'L{l}_c_ei'][:],
            'n_c': int(g.attrs[f'L{l}_n_c']),
            'mode': str(g.attrs[f'L{l}_mode']),
        }
        entry['seeds'] = g[f'L{l}_seeds'][:] if g.attrs.get(f'L{l}_has_seeds', False) else None
        if g.attrs.get(f'L{l}_has_up', False):
            entry['up_ei'] = g[f'L{l}_up_ei'][:]
        hierarchy.append(entry)
    x_pos = g['x_pos'][:] if 'x_pos' in g else None
    return hierarchy, x_pos


# ---------------------------------------------------------------------------
# Build (parallel)
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(h5_file: str, coarse_params: dict, pos_params: dict) -> None:
    _WORKER['h5'] = h5py.File(h5_file, 'r')
    _WORKER['cp'] = coarse_params
    _WORKER['pp'] = pos_params


def _build_one(sid: int):
    """Build one sample's hierarchy (+ positional features). Runs in a pool worker."""
    f = _WORKER['h5']
    cp = _WORKER['cp']
    pp = _WORKER['pp']

    # Deterministic FPS seed -> reproducible cache, identical across jobs/workers.
    np.random.seed(int(sid) & 0x7FFFFFFF)

    mesh_edge = f[f'data/{sid}/mesh_edge'][:]
    edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)
    nodal = f[f'data/{sid}/nodal_data']
    num_nodes = nodal.shape[2]
    ref_pos = np.ascontiguousarray(nodal[:3, 0, :].T).astype(np.float32)  # [N, 3]

    hierarchy = build_multiscale_hierarchy(
        edge_index, num_nodes, ref_pos,
        cp['levels'], cp['types'], cp['clusters'],
    )

    x_pos = None
    if pp['num'] > 0:
        from general_modules.positional_features import compute_positional_features
        x_pos = compute_positional_features(ref_pos, edge_index, pp['num'])

    return int(sid), hierarchy, x_pos


def _resolve_build_workers(config: dict, n_samples: int) -> int:
    requested = config.get('hierarchy_cache_build_workers')
    if requested is not None:
        return max(1, int(requested))
    return max(1, min(32, int(mp.cpu_count() * 0.5), n_samples))


def build_cache(h5_file, sample_ids, signature, cache_path,
                coarse_params, pos_params, config) -> None:
    """Build the full cache to a temp file, then atomically rename into place."""
    sample_ids = [int(s) for s in sample_ids]
    n = len(sample_ids)
    num_workers = _resolve_build_workers(config, n)
    tmp_path = f'{cache_path}.tmp.{os.getpid()}'

    print(f'[mscache] Building hierarchy cache for {n} samples '
          f'({num_workers} workers) -> {cache_path}')
    t0 = time.time()

    try:
        with h5py.File(tmp_path, 'w') as out:
            out.attrs['signature'] = json.dumps(signature, sort_keys=True)
            out.attrs['format_version'] = FORMAT_VERSION
            out.create_dataset('sample_ids', data=np.asarray(sorted(sample_ids), dtype=np.int64))

            def _progress(k):
                if k % max(1, n // 20) == 0 or k == n:
                    el = time.time() - t0
                    rate = k / el if el > 0 else 0.0
                    eta = (n - k) / rate if rate > 0 else 0.0
                    print(f'[mscache]   {k}/{n}  ({el:.0f}s elapsed, ~{eta:.0f}s left)')

            if num_workers <= 1:
                _init_worker(h5_file, coarse_params, pos_params)
                for k, sid in enumerate(sample_ids, 1):
                    _, hier, xp = _build_one(sid)
                    _write_entry(out, sid, hier, xp)
                    _progress(k)
            else:
                ctx = mp.get_context('spawn')
                with ctx.Pool(num_workers, initializer=_init_worker,
                              initargs=(h5_file, coarse_params, pos_params)) as pool:
                    for k, (sid, hier, xp) in enumerate(
                            pool.imap_unordered(_build_one, sample_ids, chunksize=2), 1):
                        _write_entry(out, sid, hier, xp)
                        _progress(k)
        os.replace(tmp_path, cache_path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise

    print(f'[mscache] Done in {time.time() - t0:.0f}s '
          f'({os.path.getsize(cache_path) / 1e9:.1f} GB on disk)')


# ---------------------------------------------------------------------------
# Validation + lock coordination
# ---------------------------------------------------------------------------

def _is_valid(cache_path: str, sig_json: str, sample_ids) -> bool:
    if not os.path.exists(cache_path):
        return False
    try:
        with h5py.File(cache_path, 'r') as f:
            if int(f.attrs.get('format_version', -1)) != FORMAT_VERSION:
                return False
            if f.attrs.get('signature', '') != sig_json:
                return False
            if 'sample_ids' not in f:
                return False
            cached = set(int(s) for s in f['sample_ids'][:])
        return set(int(s) for s in sample_ids).issubset(cached)
    except (OSError, KeyError, ValueError):
        return False


def _lock_is_stale(lock_path: str) -> bool:
    try:
        return (time.time() - os.path.getmtime(lock_path)) > _STALE_LOCK_SECONDS
    except OSError:
        return True


def ensure_cache(h5_file: str, sample_ids, config: dict):
    """Return a path to a valid hierarchy cache, building it if needed.

    Coordinated across concurrent jobs by an exclusive lock file: exactly one
    process builds, the rest poll until the finished file appears.
    """
    coarse_params = _coarse_params(config)
    pos_params = _pos_params(config)
    signature = _signature(h5_file, coarse_params, pos_params)
    sig_json = json.dumps(signature, sort_keys=True)
    cache_path = cache_path_for(h5_file, config, signature)

    if _is_valid(cache_path, sig_json, sample_ids):
        print(f'[mscache] Using existing hierarchy cache: {cache_path}')
        return cache_path

    lock_path = cache_path + '.lock'
    timeout = float(config.get('hierarchy_cache_wait_timeout', 36000))  # 10 h
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Another process is building (or crashed). Wait, then re-check.
            if _is_valid(cache_path, sig_json, sample_ids):
                print(f'[mscache] Cache built by another job: {cache_path}')
                return cache_path
            if _lock_is_stale(lock_path):
                print('[mscache] Removing stale lock and retrying...')
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            time.sleep(3.0)
            continue
        else:
            # We hold the lock.
            try:
                os.write(fd, f'{os.getpid()} {time.time()}'.encode())
                os.close(fd)
                # Double-check in case a builder finished between our checks.
                if _is_valid(cache_path, sig_json, sample_ids):
                    return cache_path
                build_cache(h5_file, sample_ids, signature, cache_path,
                            coarse_params, pos_params, config)
                return cache_path
            finally:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

    raise TimeoutError(
        f'[mscache] Timed out after {timeout:.0f}s waiting for hierarchy cache '
        f'{cache_path}. If a previous build crashed, delete the .lock file.')


# ---------------------------------------------------------------------------
# Reader (one persistent handle per process)
# ---------------------------------------------------------------------------

class HierarchyCacheReader:
    """Streams precomputed hierarchies / positional features from the cache file.

    One handle per process; not picklable (reset to ``None`` in the dataset's
    ``__getstate__`` so DataLoader workers reopen lazily).
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._h = None

    def _handle(self) -> h5py.File:
        if self._h is None:
            self._h = h5py.File(self.cache_path, 'r')
        return self._h

    def has(self, sid: int) -> bool:
        return str(int(sid)) in self._handle()

    def get_hierarchy(self, sid: int):
        hierarchy, _ = _read_entry(self._handle()[str(int(sid))])
        return hierarchy

    def get_pos(self, sid: int):
        g = self._handle()[str(int(sid))]
        return g['x_pos'][:] if 'x_pos' in g else None

    def close(self) -> None:
        if self._h is not None:
            try:
                self._h.close()
            except Exception:
                pass
            self._h = None
