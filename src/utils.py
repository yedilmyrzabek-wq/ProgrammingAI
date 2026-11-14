import json
import time
import tracemalloc


def measure(operation, *args, **kwargs):
    """Return the result, wall time and peak traced Python allocations in MiB."""
    already_tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        start = time.perf_counter()
        result = operation(*args, **kwargs)
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    return result, seconds, max(0, peak - before) / 1024**2


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return path
