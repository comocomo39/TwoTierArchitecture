# scripts/run.py
import importlib
import sys

USAGE = """\
Uso:
  python -m scripts.run <pipeline>

Pipeline:
  tasb_two_tier | colbert_two_tier | tasb_single_index
"""

def main():
    target = (sys.argv[1] if len(sys.argv) > 1 else "tasb_two_tier").strip().lower()
    mod_map = {
        "tasb_two_tier": "src.tasb_two_tier",
        "colbert_two_tier": "src.colbert_two_tier",
        "tasb_single_index": "src.tasb_single_index",
    }
    if target not in mod_map:
        print(USAGE); sys.exit(1)

    mod_name = mod_map[target]
    mod = importlib.import_module(mod_name)
    # garantisce che un secondo run nella stessa sessione ricominci davvero
    importlib.reload(mod)

if __name__ == "__main__":
    main()
