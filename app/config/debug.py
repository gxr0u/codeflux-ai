from __future__ import annotations


DEBUG = True


def debug_print(*args, **kwargs) -> None:
    if DEBUG:
        print(*args, **kwargs)
