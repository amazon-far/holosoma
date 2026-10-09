from holosoma.utils.path import resolve_path


def get_holosoma_root() -> str:
    """Return the installed holosoma package root."""
    return resolve_path("@holosoma")
