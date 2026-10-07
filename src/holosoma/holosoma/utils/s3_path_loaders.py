"""Built-in singular and plural S3 path loaders."""

from __future__ import annotations

from collections.abc import Iterable
from glob import has_magic

from holosoma.utils.s3_glob import list_s3_uris


def _download(uri: str) -> str:
    from holosoma.utils.file_cache import get_cached_file_path

    return get_cached_file_path(uri)


class S3PathLoader:
    """Localize one S3 URI or lazily localize an S3 glob."""

    def __init__(self, *, multiple: bool) -> None:
        self.multiple = multiple

    def matches(self, value: str) -> bool:
        return value.startswith("s3://")

    def load(self, value: str) -> str | Iterable[str]:
        if not self.multiple:
            return _download(value)
        if has_magic(value):
            uris = list_s3_uris(value)
        else:
            uris = [value]
        return (_download(uri) for uri in uris)


s3_path_loader = S3PathLoader(multiple=False)
s3_paths_loader = S3PathLoader(multiple=True)
