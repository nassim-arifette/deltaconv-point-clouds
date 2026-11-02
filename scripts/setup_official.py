"""Fetch pinned DeltaConv and optionally build its historical Linux environment.

This is a separate environment from the lightweight NumPy audit. Its installation
and the neural benchmark remain unverified in the current restricted workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deltaconv_audit import UPSTREAM_COMMIT, UPSTREAM_URL

MAMBA_URL = "https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/micromamba-linux-64"
MAMBA_SHA256 = "701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT / "work/official")
    parser.add_argument("--install-env", action="store_true")
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    repo = work / "DeltaConv"
    if not repo.exists():
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", UPSTREAM_COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError("existing checkout has a different revision; use a new workdir")
    for extra in ([], ["--cached"]):
        subprocess.run(["git", "-C", str(repo), "diff", *extra, "--exit-code", "--quiet"], check=True)
    subprocess.run(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"], check=True)
    receipt = {"upstream": UPSTREAM_URL, "commit": actual, "environment_created": False,
               "environment_recipe_sha256": hashlib.sha256((repo / "environment.yml").read_bytes()).hexdigest()}
    if args.install_env:
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
            raise RuntimeError("the historical environment installer targets Linux x86_64 / Colab")
        mamba = work / "micromamba"
        if not mamba.exists():
            urllib.request.urlretrieve(MAMBA_URL, mamba)
        if hashlib.sha256(mamba.read_bytes()).hexdigest() != MAMBA_SHA256:
            raise RuntimeError("micromamba checksum mismatch")
        mamba.chmod(0o755)
        prefix = work / "environment"
        env = {**os.environ, "MAMBA_ROOT_PREFIX": str(work / "mamba-root")}
        subprocess.run([str(mamba), "create", "-y", "-p", str(prefix), "-f", str(repo / "environment.yml")], env=env, check=True)
        python = prefix / "bin/python"
        # The upstream evaluation script imports sklearn, absent from its explicit pip list.
        subprocess.run([str(python), "-m", "pip", "install", "scikit-learn==1.0.2"], check=True)
        # Replace the recipe's PyPI package by this exact checkout, including its
        # native FPS extension. CMake and a C++ compiler must be available.
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                        "-e", str(repo)], check=True)
        package_path = subprocess.check_output([str(python), "-c",
            "import deltaconv; print(deltaconv.__file__)"], text=True).strip()
        if not Path(package_path).resolve().is_relative_to(repo):
            raise RuntimeError("environment did not import the pinned DeltaConv checkout")
        frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
        (work / "pip-freeze.txt").write_text(frozen, encoding="utf-8")
        receipt.update(environment_created=True, python=str(python), micromamba_sha256=MAMBA_SHA256,
                       deltaconv_import=package_path)
    (work / "setup.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

