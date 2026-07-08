import importlib
for m in ["torch", "transformers", "safetensors", "numpy", "matplotlib", "huggingface_hub"]:
    try:
        mod = importlib.import_module(m)
        print("  ", m, getattr(mod, "__version__", "?"))
    except Exception:
        print("  ", m, "MISSING")
