"""Atomic checkpoints, source guards and automatically released writer locks."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def atomic_json(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as handle:
        temporary=Path(handle.name)
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')
    os.replace(temporary,path)


def atomic_npz(path,**arrays):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle:
        temporary=Path(handle.name)
        np.savez_compressed(handle,**arrays)
    os.replace(temporary,path)


@contextmanager
def guarded_run(name,sources):
    OUT.mkdir(exist_ok=True)
    with (OUT/f'.{name}.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another {name} writer is already running.') from exc
        digest={s:hashlib.sha256((HERE/s).read_bytes()).hexdigest()
                for s in [*sources,'provenance.py']}
        path=OUT/f'{name}_source_guard.json'
        if path.exists() and json.loads(path.read_text())!=digest:
            raise RuntimeError('Scientific inputs changed; preserve these records and use a fresh result directory.')
        if not path.exists():atomic_json(path,digest)
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)
