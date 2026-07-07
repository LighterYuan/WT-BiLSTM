from __future__ import annotations

import importlib
import sys


REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pywt": "PyWavelets",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
    "tensorflow": "tensorflow",
    "openpyxl": "openpyxl",
}


def main():
    print(f"Python: {sys.version}")
    missing = []
    for import_name, package_name in REQUIRED.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {package_name}: {version}")
        except Exception as exc:
            print(f"[MISSING] {package_name}: {exc}")
            missing.append(package_name)

    if missing:
        print("\nMissing packages:")
        print("pip install -r requirements.txt")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
