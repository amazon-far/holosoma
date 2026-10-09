"""Unit tests for generic one-to-one and one-to-many path loaders."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files  # type: ignore[import-not-found]

import boto3  # type: ignore[import-untyped]
import pytest
from botocore.stub import Stubber  # type: ignore[import-untyped]

from holosoma.utils import path as path_utils
from holosoma.utils.core_path_loaders import local_path_loader
from holosoma.utils.package_path_loader import PackagePathLoader
from holosoma.utils.path import resolve_asset_path, resolve_data_file_path, resolve_path, resolve_paths
from holosoma.utils.s3_glob import _literal_prefix, list_s3_uris
from holosoma.utils.s3_path_loaders import s3_path_loader

PKG = str(files("holosoma"))


@pytest.fixture(autouse=True)
def _clear_entrypoint_loader_cache():
    path_utils._entrypoint_loaders.cache_clear()
    yield
    path_utils._entrypoint_loaders.cache_clear()


# ----- core singular loader -----


def test_package_path_resolves_under_holosoma():
    assert (
        resolve_path("holosoma/data/scene_objects/boxes/small_box.urdf")
        == f"{PKG}/data/scene_objects/boxes/small_box.urdf"
    )


def test_at_holosoma_alias_resolves_identically():
    plain = resolve_path("holosoma/data/scene_objects/boxes/small_box.urdf")
    alias = resolve_path("@holosoma/data/scene_objects/boxes/small_box.urdf")
    assert alias == plain == f"{PKG}/data/scene_objects/boxes/small_box.urdf"


def test_at_holosoma_root_resolves_to_package_dir():
    assert resolve_path("@holosoma") == PKG


def test_package_root_with_trailing_slash_resolves_to_package_dir():
    assert resolve_path("holosoma/") == PKG


def test_bare_holosoma_remains_a_plain_local_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert resolve_path("holosoma") == str(tmp_path / "holosoma")


def test_absolute_path_returned_as_is():
    assert resolve_path("/home/user/custom.npz") == "/home/user/custom.npz"


def test_relative_path_resolved_against_cwd():
    assert resolve_data_file_path("my_data/custom.npz") == str(Path.cwd() / "my_data/custom.npz")


def test_user_path_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_path("~/custom.npz") == str(tmp_path / "custom.npz")


def test_s3_path_is_downloaded_to_local_cache(monkeypatch, tmp_path):
    local_path = tmp_path / "box.usd"
    uri = "s3://bucket/path/to/box.usd"
    monkeypatch.setattr(
        "holosoma.utils.file_cache.get_cached_file_path",
        lambda value: str(local_path) if value == uri else None,
    )

    assert resolve_path(uri) == str(local_path)


# ----- core plural loader -----


def test_local_glob_returns_sorted_absolute_paths(tmp_path):
    (tmp_path / "b.npz").touch()
    (tmp_path / "a.npz").touch()
    (tmp_path / "ignore.txt").touch()

    assert list(resolve_paths(str(tmp_path / "*.npz"))) == [
        str(tmp_path / "a.npz"),
        str(tmp_path / "b.npz"),
    ]


def test_holosoma_prefix_glob_resolves_package_files():
    assert list(resolve_paths("@holosoma/data/scene_objects/boxes/*.urdf")) == [
        f"{PKG}/data/scene_objects/boxes/large_box.urdf",
        f"{PKG}/data/scene_objects/boxes/small_box.urdf",
    ]


def test_package_loader_supports_extension_prefix_and_standard_glob():
    path_loader = PackagePathLoader("holosoma", "@my_ext", multiple=False)
    paths_loader = PackagePathLoader("holosoma", "@my_ext", multiple=True)

    assert path_loader.matches("@my_ext")
    assert path_loader.matches("@my_ext/data")
    assert not path_loader.matches("@my_extension/data")
    assert path_loader.load("@my_ext/data/scene_objects/boxes/small_box.urdf") == (
        f"{PKG}/data/scene_objects/boxes/small_box.urdf"
    )
    assert list(paths_loader.load("@my_ext/data/scene_objects/**/small_box.urdf")) == [
        f"{PKG}/data/scene_objects/boxes/small_box.urdf"
    ]


def test_package_loader_recursive_glob_allows_zero_directories(monkeypatch, tmp_path):
    match = tmp_path / "a" / "b" / "c.npz"
    match.parent.mkdir(parents=True)
    match.touch()
    monkeypatch.setattr("holosoma.utils.package_path_loader.files", lambda _package: tmp_path)
    loader = PackagePathLoader("my_ext", "@my_ext", multiple=True)

    assert list(loader.load("@my_ext/a/**/b/**/c.npz")) == [str(match)]


def test_non_glob_path_is_a_singleton_even_if_missing(tmp_path):
    path = str(tmp_path / "missing.npz")
    assert list(resolve_paths(path)) == [path]


def test_exact_s3_path_plural_loader_downloads_one_file(monkeypatch, tmp_path):
    local_path = tmp_path / "motion.npz"
    uri = "s3://bucket/motion.npz"
    downloaded = []

    def get_cached_file_path(value):
        downloaded.append(value)
        return str(local_path)

    monkeypatch.setattr("holosoma.utils.file_cache.get_cached_file_path", get_cached_file_path)

    paths = resolve_paths(uri)
    assert downloaded == []
    assert list(paths) == [str(local_path)]
    assert downloaded == [uri]


def _stub_s3_listing(monkeypatch, keys, *, prefix):
    """Use botocore's documented Stubber to emulate paginated ListObjectsV2."""
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # noqa: S106 - inert Stubber credentials
    )
    stubber = Stubber(client)
    listed = [{"Key": key} for key in keys if key.startswith(prefix)]
    first_page = {"IsTruncated": len(listed) > 1, "Contents": listed[:1]}
    if len(listed) > 1:
        first_page["NextContinuationToken"] = "page-2"
    stubber.add_response(
        "list_objects_v2",
        first_page,
        {"Bucket": "bucket", "Prefix": prefix},
    )
    if len(listed) > 1:
        stubber.add_response(
            "list_objects_v2",
            {"IsTruncated": False, "Contents": listed[1:]},
            {"Bucket": "bucket", "Prefix": prefix, "ContinuationToken": "page-2"},
        )
    stubber.activate()
    monkeypatch.setattr("holosoma.utils.s3_glob._get_s3_client", lambda: client)
    return stubber


def _fake_s3_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "s3-cache"
    downloaded = []

    def get_cached_file_path(uri):
        downloaded.append(uri)
        relative = uri[len("s3://bucket/") :]
        local_path = cache_root / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.touch()
        return str(local_path)

    monkeypatch.setattr("holosoma.utils.file_cache.get_cached_file_path", get_cached_file_path)
    return cache_root, downloaded


def test_s3_glob_lists_scoped_prefix_and_returns_sorted_local_paths(monkeypatch, tmp_path):
    stubber = _stub_s3_listing(
        monkeypatch,
        [
            "motions/clip2.npz",
            "motions/clip1.npz",
            "motions/notes.md",
            "motions/.hidden.npz",
            "motions/take2/deep.npz",
            "other/elsewhere.npz",
        ],
        prefix="motions/",
    )
    cache_root, downloaded = _fake_s3_cache(monkeypatch, tmp_path)

    paths = resolve_paths("s3://bucket/motions/*.npz")
    stubber.assert_no_pending_responses()
    assert downloaded == []

    assert next(paths) == str(cache_root / "motions/clip1.npz")
    assert downloaded == ["s3://bucket/motions/clip1.npz"]
    assert list(paths) == [str(cache_root / "motions/clip2.npz")]
    assert downloaded == ["s3://bucket/motions/clip1.npz", "s3://bucket/motions/clip2.npz"]


def test_each_s3_double_star_can_match_zero_components(monkeypatch, tmp_path):
    keys = [
        "a/b/c.npz",
        "a/x/b/y/c.npz",
    ]
    stubber = _stub_s3_listing(monkeypatch, keys, prefix="a/")
    cache_root, _ = _fake_s3_cache(monkeypatch, tmp_path)

    assert list(resolve_paths("s3://bucket/a/**/b/**/c.npz")) == [str(cache_root / key) for key in keys]
    stubber.assert_no_pending_responses()


def test_s3_glob_is_not_motion_format_specific(monkeypatch, tmp_path):
    stubber = _stub_s3_listing(
        monkeypatch,
        ["motions/clip.npz", "motions/notes.md"],
        prefix="motions/",
    )
    cache_root, _ = _fake_s3_cache(monkeypatch, tmp_path)

    assert list(resolve_paths("s3://bucket/motions/*")) == [
        str(cache_root / "motions/clip.npz"),
        str(cache_root / "motions/notes.md"),
    ]
    stubber.assert_no_pending_responses()


def test_s3_glob_matching_nothing_returns_empty_without_downloading(monkeypatch):
    stubber = _stub_s3_listing(monkeypatch, ["motions/notes.md"], prefix="motions/")

    def fail_if_downloaded(uri):
        raise AssertionError(f"unexpected download: {uri}")

    monkeypatch.setattr("holosoma.utils.file_cache.get_cached_file_path", fail_if_downloaded)

    assert list(resolve_paths("s3://bucket/motions/*.npz")) == []
    stubber.assert_no_pending_responses()


def test_s3_glob_handles_listing_page_without_contents(monkeypatch):
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # noqa: S106 - inert Stubber credentials
    )
    stubber = Stubber(client)
    stubber.add_response(
        "list_objects_v2",
        {"IsTruncated": False},
        {"Bucket": "bucket", "Prefix": "motions/"},
    )
    stubber.activate()
    monkeypatch.setattr("holosoma.utils.s3_glob._get_s3_client", lambda: client)

    assert list(resolve_paths("s3://bucket/motions/*.npz")) == []
    stubber.assert_no_pending_responses()


@pytest.mark.parametrize(
    "pattern",
    [
        "not-s3",
        "s3://",
        "s3://bucket",
        "s3://bucket/",
        "s3://buck*/motion.npz",
    ],
)
def test_invalid_s3_glob_is_rejected_before_listing(monkeypatch, pattern):
    def fail_if_listed():
        raise AssertionError("invalid S3 pattern reached the listing boundary")

    monkeypatch.setattr("holosoma.utils.s3_glob._get_s3_client", fail_if_listed)

    with pytest.raises(ValueError, match=r"Invalid S3 pattern|bucket names|object-key pattern"):
        list_s3_uris(pattern)


@pytest.mark.parametrize(
    "pattern",
    [
        "a/**/b/**/c.npz",
        "motions/*.npz",
        "motions/clip?.npz",
        "motions/clip[12].npz",
        "motions/**/*.npz",
        "motions/**/.hidden.npz",
        "motions/.hidden/*.npz",
        "**/**/clip*.npz",
        "literal/star[*].npz",
        "literal/question[?].npz",
        "literal/[[]bracket].npz",
        "classes/[!0-9].npz",
    ],
)
def test_s3_glob_selection_matches_python_stdlib(monkeypatch, tmp_path, pattern):
    keys = [
        "a/b/c.npz",
        "a/x/b/y/c.npz",
        "motions/clip1.npz",
        "motions/clip2.npz",
        "motions/clip10.npz",
        "motions/sub/clip3.npz",
        "motions/sub/.hidden.npz",
        "motions/.hidden/inside.npz",
        "other/clip4.npz",
        "literal/star*.npz",
        "literal/question?.npz",
        "literal/[bracket].npz",
        "classes/a.npz",
        "classes/1.npz",
    ]
    local_root = tmp_path / "local"
    for key in keys:
        local_path = local_root / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.touch()

    expected = sorted(
        str(Path(match).relative_to(local_root))
        for match in glob.glob(str(local_root / pattern), recursive=True)  # noqa: PTH207
        if Path(match).is_file()
    )
    stubber = _stub_s3_listing(monkeypatch, keys, prefix=_literal_prefix(pattern))
    _, downloaded = _fake_s3_cache(monkeypatch, tmp_path)

    list(resolve_paths(f"s3://bucket/{pattern}"))

    stubber.assert_no_pending_responses()
    assert [uri[len("s3://bucket/") :] for uri in downloaded] == expected


# ----- extension entry points -----


class _EntryPoint:
    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader


class _Loader:
    def __init__(self, matches, load):
        self.matches = matches
        self.load = load


def _install_fake_entry_points(monkeypatch, groups):
    monkeypatch.setattr(path_utils, "entry_points", lambda *, group: groups.get(group, []))


def test_entrypoint_name_does_not_dispatch_singular_loader(monkeypatch, tmp_path):
    asset = tmp_path / "assets" / "robot.urdf"

    loader = _Loader(
        lambda path: path.startswith("@my_ext/"),
        lambda path: tmp_path / path[len("@my_ext/") :],
    )

    _install_fake_entry_points(
        monkeypatch,
        {
            path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [
                _EntryPoint("unrelated-label", "my_ext.paths:path_loader", loader),
                _EntryPoint("local", "holosoma.utils.core_path_loaders:local_path_loader", local_path_loader),
            ]
        },
    )

    assert resolve_path("@my_ext/assets/robot.urdf") == str(asset)
    assert resolve_asset_path("@my_ext/assets/robot.urdf", "/unused/root") == str(asset)
    assert resolve_asset_path("robot.urdf", "@my_ext/assets") == str(asset)


def test_package_loader_resolves_relative_asset_file_under_symbolic_root(monkeypatch):
    loader = PackagePathLoader("holosoma", "@my_ext", multiple=False)
    _install_fake_entry_points(
        monkeypatch,
        {
            path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [
                _EntryPoint("not-the-prefix", "my_ext.paths:path_loader", loader),
                _EntryPoint("local", "holosoma.utils.core_path_loaders:local_path_loader", local_path_loader),
            ]
        },
    )

    assert resolve_asset_path("small_box.urdf", "@my_ext/data/scene_objects/boxes") == (
        f"{PKG}/data/scene_objects/boxes/small_box.urdf"
    )


def test_plural_resolution_does_not_compose_singular_loader(monkeypatch, tmp_path):
    loader = _Loader(
        lambda path: path.startswith("@my_ext/"),
        lambda path: tmp_path / path[len("@my_ext/") :],
    )

    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("anything", "my_ext.paths:path_loader", loader)]},
    )

    with pytest.raises(ValueError, match="No paths loader could resolve"):
        resolve_paths("@my_ext/assets/*.urdf")


def test_generic_plural_loader_results_are_normalized(monkeypatch, tmp_path):
    calls = []

    loader = _Loader(
        lambda path: path == "@my_ext/assets/*.urdf",
        lambda path: calls.append(path) or [tmp_path / "a.urdf", tmp_path / "b.urdf"],
    )

    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATHS_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("not-a-prefix", "my_ext.paths:paths_loader", loader)]},
    )

    assert list(resolve_paths("@my_ext/assets/*.urdf")) == [
        str(tmp_path / "a.urdf"),
        str(tmp_path / "b.urdf"),
    ]
    assert calls == ["@my_ext/assets/*.urdf"]


def test_plural_loader_iterable_is_consumed_lazily(monkeypatch, tmp_path):
    yielded = []

    def load_paths(path):
        def generate():
            yielded.append("first")
            yield tmp_path / "first.npz"
            yielded.append("second")
            yield tmp_path / "second.npz"

        return generate()

    loader = _Loader(lambda path: path == "@my_ext/*.npz", load_paths)
    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATHS_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("lazy", "my_ext.paths:paths_loader", loader)]},
    )

    paths = resolve_paths("@my_ext/*.npz")
    assert yielded == []
    assert next(paths) == str(tmp_path / "first.npz")
    assert yielded == ["first"]
    assert list(paths) == [str(tmp_path / "second.npz")]
    assert yielded == ["first", "second"]


def test_singular_loader_must_return_local_path(monkeypatch):
    loader = _Loader(
        lambda path: path == "@bad/file.npz",
        lambda _path: "s3://bucket/file.npz",
    )

    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("bad-output", "bad.paths:path_loader", loader)]},
    )

    with pytest.raises(ValueError, match="returned non-local path"):
        resolve_path("@bad/file.npz")


def test_singular_loader_rejects_bytes(monkeypatch):
    loader = _Loader(
        lambda path: path == "@bad/file.npz",
        lambda _path: b"/tmp/file.npz",
    )
    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("bytes", "bad.paths:path_loader", loader)]},
    )

    with pytest.raises(TypeError, match="returned bytes"):
        resolve_path("@bad/file.npz")


def test_plural_loader_must_return_local_paths(monkeypatch):
    loader = _Loader(
        lambda path: path == "@bad/*.npz",
        lambda _path: ["s3://bucket/file.npz"],
    )

    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATHS_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("bad-output", "bad.paths:paths_loader", loader)]},
    )

    with pytest.raises(ValueError, match="returned non-local path"):
        list(resolve_paths("@bad/*.npz"))


def test_plural_loader_must_return_an_iterable(monkeypatch):
    loader = _Loader(lambda _path: True, lambda _path: 7)
    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATHS_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("not-iterable", "bad.paths:paths_loader", loader)]},
    )

    with pytest.raises(TypeError, match="did not return an iterable"):
        resolve_paths("@bad/*.npz")


def test_unmatched_loader_is_not_loaded(monkeypatch, tmp_path):
    def fail_if_loaded(path):
        raise AssertionError(f"unmatched loader was invoked for {path}")

    skipped = _Loader(lambda _path: False, fail_if_loaded)
    selected = _Loader(lambda path: path == "@selected/file", lambda _path: tmp_path / "file")
    _install_fake_entry_points(
        monkeypatch,
        {
            path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [
                _EntryPoint("first", "first.paths:path_loader", skipped),
                _EntryPoint("second", "second.paths:path_loader", selected),
            ]
        },
    )

    assert resolve_path("@selected/file") == str(tmp_path / "file")


def test_invalid_entrypoint_loader_is_skipped_without_hiding_valid_loader(monkeypatch, tmp_path):
    selected = _Loader(lambda path: path == "@selected/file", lambda _path: tmp_path / "file")
    _install_fake_entry_points(
        monkeypatch,
        {
            path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [
                _EntryPoint("broken", "broken.paths:path_loader", object()),
                _EntryPoint("valid", "valid.paths:path_loader", selected),
            ]
        },
    )

    assert resolve_path("@selected/file") == str(tmp_path / "file")


def test_loader_matcher_must_return_bool(monkeypatch):
    loader = _Loader(lambda _path: 1, lambda path: path)
    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("bad-matcher", "bad.paths:path_loader", loader)]},
    )

    with pytest.raises(TypeError, match=r"matches\(\); expected bool"):
        resolve_path("@bad/file")


def test_loader_matcher_failure_reports_loader_source(monkeypatch):
    def fail_matching(_path):
        raise RuntimeError("matcher broke")

    loader = _Loader(fail_matching, lambda path: path)
    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("broken", "broken.paths:path_loader", loader)]},
    )

    with pytest.raises(RuntimeError, match=r"broken\.paths:path_loader") as exc_info:
        resolve_path("@broken/file")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_plural_loader_cannot_return_one_string(monkeypatch):
    loader = _Loader(lambda _path: True, lambda _path: "/tmp/one.npz")
    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATHS_LOADER_ENTRYPOINT_GROUP: [_EntryPoint("bad-plural", "bad.paths:paths_loader", loader)]},
    )

    with pytest.raises(TypeError, match="returned one path instead of an iterable"):
        resolve_paths("@bad/*.npz")


@pytest.mark.parametrize("reverse_entrypoints", [False, True])
def test_overlapping_singular_loaders_are_ambiguous_before_loading(monkeypatch, tmp_path, reverse_entrypoints):
    load_calls = []

    def load_first(path):
        load_calls.append("first")
        return tmp_path / "first"

    def load_second(path):
        load_calls.append("second")
        return tmp_path / "second"

    entries = [
        _EntryPoint("arbitrary-a", "first.paths:path_loader", _Loader(lambda _path: True, load_first)),
        _EntryPoint("arbitrary-b", "second.paths:path_loader", _Loader(lambda _path: True, load_second)),
    ]
    if reverse_entrypoints:
        entries.reverse()
    _install_fake_entry_points(
        monkeypatch,
        {path_utils.PATH_LOADER_ENTRYPOINT_GROUP: entries},
    )

    with pytest.raises(ValueError, match="Multiple path loaders matched") as exc_info:
        resolve_path("@shared/file")

    assert "first.paths:path_loader" in str(exc_info.value)
    assert "second.paths:path_loader" in str(exc_info.value)
    assert load_calls == []


def test_overlapping_plural_loaders_are_ambiguous_before_loading(monkeypatch):
    load_calls = []

    def load_paths(path):
        load_calls.append(path)
        return []

    _install_fake_entry_points(
        monkeypatch,
        {
            path_utils.PATHS_LOADER_ENTRYPOINT_GROUP: [
                _EntryPoint("a", "first.paths:paths_loader", _Loader(lambda _path: True, load_paths)),
                _EntryPoint("b", "second.paths:paths_loader", _Loader(lambda _path: True, load_paths)),
            ]
        },
    )

    with pytest.raises(ValueError, match="Multiple paths loaders matched"):
        resolve_paths("@shared/*.npz")

    assert load_calls == []


def test_extension_overlapping_builtin_s3_loader_is_ambiguous_before_downloading(monkeypatch, tmp_path):
    load_calls = []

    def load_s3(path):
        load_calls.append(path)
        return tmp_path / "extension-copy.npz"

    _install_fake_entry_points(
        monkeypatch,
        {
            path_utils.PATH_LOADER_ENTRYPOINT_GROUP: [
                _EntryPoint(
                    "s3",
                    "holosoma.utils.s3_path_loaders:s3_path_loader",
                    s3_path_loader,
                ),
                _EntryPoint(
                    "unrelated-name",
                    "my_ext.paths:s3_loader",
                    _Loader(lambda path: path.startswith("s3://"), load_s3),
                ),
            ]
        },
    )

    with pytest.raises(ValueError, match="Multiple path loaders matched") as exc_info:
        resolve_path("s3://bucket/motion.npz")

    assert "my_ext.paths:s3_loader" in str(exc_info.value)
    assert "holosoma.utils.s3_path_loaders:s3_path_loader" in str(exc_info.value)
    assert load_calls == []


def test_unknown_at_prefix_fails_loudly():
    with pytest.raises(ValueError, match="No path loader could resolve"):
        resolve_path("@missing_ext/data/file.npz")


def test_unknown_uri_scheme_fails_loudly():
    with pytest.raises(ValueError, match="No path loader could resolve"):
        resolve_path("ftp://example.com/data/file.npz")


# ----- resolve_asset_path: asset_root applies only to plain relative paths -----


@pytest.mark.parametrize(
    "asset_file",
    [
        "holosoma/data/scene_objects/boxes/small_box.urdf",
        "@holosoma/data/scene_objects/boxes/small_box.urdf",
        "/abs/box.urdf",
    ],
)
def test_self_locating_paths_ignore_asset_root(asset_file):
    assert resolve_asset_path(asset_file, asset_root="/some/root") == resolve_path(asset_file)


def test_s3_asset_ignores_root_and_is_downloaded(monkeypatch, tmp_path):
    local_path = tmp_path / "box.usd"
    uri = "s3://bucket/box.usd"
    monkeypatch.setattr(
        "holosoma.utils.file_cache.get_cached_file_path",
        lambda value: str(local_path) if value == uri else None,
    )

    assert resolve_asset_path(uri, asset_root="/some/root") == str(local_path)


def test_relative_path_joined_onto_asset_root():
    assert resolve_asset_path("box.urdf", asset_root="/some/root") == "/some/root/box.urdf"


def test_relative_path_strips_trailing_slash_on_root():
    assert resolve_asset_path("box.urdf", asset_root="/some/root/") == "/some/root/box.urdf"


def test_relative_path_without_root_falls_back_to_cwd():
    assert resolve_asset_path("box.urdf", asset_root=None) == str(Path.cwd() / "box.urdf")
