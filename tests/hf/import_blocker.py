"""Make installed packages look missing to a fresh interpreter.

Triton, flash-attn and friends ARE installed on the GPU nodes, so a child
process that must prove it never needs them blocks them instead of relying on
their absence. Children find this module through PYTHONPATH and import it
before anything else, torch and transformers included:

    from import_blocker import block_imports

    attempts = block_imports(("triton", "flash_attn"))
"""

import importlib.abc
import importlib.util
import sys


def block_imports(blocked):
    """Make every module under a top-level name in `blocked` unimportable.

    Returns the list that records each blocked attempt as
    (module name, file of the frame that imported it).
    """
    blocked = frozenset(blocked)
    attempts = []

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] not in blocked:
                return None
            frame = sys._getframe(1)
            while frame is not None and frame.f_code.co_filename.startswith("<frozen"):
                frame = frame.f_back
            attempts.append((name, frame.f_code.co_filename if frame else "?"))
            raise ModuleNotFoundError(f"blocked by import_blocker: {name}", name=name)

    sys.meta_path.insert(0, _Blocker())

    # transformers probes optional packages with importlib.util.find_spec, which
    # would propagate the finder's error and abort `import transformers`. Answer
    # those probes the way a machine without the package does.
    real_find_spec = importlib.util.find_spec

    def _find_spec(name, package=None):
        if name.split(".")[0] in blocked:
            return None
        return real_find_spec(name, package)

    importlib.util.find_spec = _find_spec
    return attempts
