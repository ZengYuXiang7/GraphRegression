# models/__init__.py
import importlib
import pkgutil

for m in pkgutil.iter_modules(__path__):
    if m.ispkg:
        continue
    importlib.import_module(f"{__name__}.{m.name}")