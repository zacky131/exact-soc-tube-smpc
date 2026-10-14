"""Reproducibility record for every result directory (ARIS shared/experiment-integrity.md)."""
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_provenance(outdir, config, script):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import clarabel, cvxpy, numpy, scipy
    root = pathlib.Path(__file__).resolve().parents[1]
    src = {str(p.relative_to(root)): _sha(p) for p in sorted((root / "cxsmpc").glob("*.py"))}
    src[str(pathlib.Path(script).resolve().relative_to(root))] = _sha(script)
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        cpu = next(l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name"))
    except Exception:
        cpu = platform.processor()
    rec = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), command=" ".join(sys.argv),
               python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
               cvxpy=cvxpy.__version__, clarabel=clarabel.__version__, cpu=cpu, os=platform.platform(),
               git_commit=None, source_sha256=src, config=config,
               note="Project is not a git repository; source SHA-256 recorded instead.")
    (outdir / "provenance.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec
