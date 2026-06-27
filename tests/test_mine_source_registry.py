"""RFC 002 source-adapter CLI seam: ``mine --source NAME`` routes through the
registry and files adapter-authored ``DrawerRecord``s via ``PalaceContext``.

This covers the seam wired into ``cmd_mine`` — the registry/ingest-loop path is
opt-in and leaves the legacy ``--mode`` dispatch untouched.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Iterator

import pytest

from mempalace.sources.base import (
    AdapterSchema,
    BaseSourceAdapter,
    DrawerRecord,
    FieldSpec,
    IngestResult,
    SourceItemMetadata,
    SourceRef,
)
from mempalace.sources.registry import register, reset_adapters, unregister


class _FakeAdapter(BaseSourceAdapter):
    name = "fake-test"
    adapter_version = "0.1.0"
    capabilities = frozenset({"byte_preserving"})
    supported_modes = frozenset({"whole_record"})
    declared_transformations = frozenset()
    default_privacy_class = "internal"

    def ingest(self, *, source: SourceRef, palace) -> Iterator[IngestResult]:
        yield SourceItemMetadata(source_file="fake://item/1", version="v1")
        yield DrawerRecord(
            content="the operator said the verb leads",
            source_file="fake://item/1",
            metadata={"wing": "wing_test", "room": "general", "lar_demo": "1"},
        )

    def describe_schema(self) -> AdapterSchema:
        return AdapterSchema(
            version="1.0",
            fields={"lar_demo": FieldSpec(type="string", required=False, description="demo field")},
        )


@pytest.fixture
def _fake_adapter():
    register("fake-test", _FakeAdapter)
    yield
    reset_adapters()
    unregister("fake-test")


def test_mine_source_files_pre_annotated_record(tmp_dir, palace_path, _fake_adapter):
    from mempalace.cli import _mine_via_source_adapter
    from mempalace.palace import get_collection

    args = Namespace(source="fake-test", dir=tmp_dir, wing=None, dry_run=False)
    _mine_via_source_adapter(args, palace_path)

    col = get_collection(palace_path, create=True)
    got = col.get(where={"source_file": "fake://item/1"})
    assert got["documents"] == ["the operator said the verb leads"]
    meta = got["metadatas"][0]
    assert meta["lar_demo"] == "1"  # adapter-authored metadata lands verbatim
    assert meta["adapter_name"] == "fake-test"  # stamped by PalaceContext (RFC 002 §5.1)
    assert meta["adapter_version"] == "0.1.0"


def test_mine_source_dry_run_files_nothing(tmp_dir, palace_path, _fake_adapter):
    from mempalace.cli import _mine_via_source_adapter
    from mempalace.palace import get_collection

    args = Namespace(source="fake-test", dir=tmp_dir, wing=None, dry_run=True)
    _mine_via_source_adapter(args, palace_path)

    col = get_collection(palace_path, create=True)
    assert col.get(where={"source_file": "fake://item/1"})["ids"] == []


def test_mine_source_unknown_adapter_exits(tmp_dir, palace_path):
    from mempalace.cli import _mine_via_source_adapter

    args = Namespace(source="does-not-exist", dir=tmp_dir, wing=None, dry_run=False)
    with pytest.raises(SystemExit):
        _mine_via_source_adapter(args, palace_path)


def test_run_mine_source_adapter_via_daemon_path(tmp_dir, palace_path, _fake_adapter):
    """The daemon path: ``run_mine`` routes a ``source_adapter`` payload through the registry
    (the same core the CLI runs), so ``mine --source NAME --daemon`` files verbatim through the
    write-daemon's single palace handle."""
    from mempalace.palace import get_collection
    from mempalace.service import run_mine

    res = run_mine(
        {
            "source_adapter": "fake-test",
            "source": tmp_dir,
            "palace_path": palace_path,
        }
    )
    assert res["success"] is True
    assert res["filed"] == 1
    assert res["source"] == "fake-test"

    col = get_collection(palace_path, create=True)
    got = col.get(where={"source_file": "fake://item/1"})
    assert got["documents"] == ["the operator said the verb leads"]
    assert got["metadatas"][0]["lar_demo"] == "1"  # adapter-authored metadata lands verbatim


def test_run_mine_source_adapter_dry_run_files_nothing(tmp_dir, palace_path, _fake_adapter):
    from mempalace.palace import get_collection
    from mempalace.service import run_mine

    res = run_mine(
        {
            "source_adapter": "fake-test",
            "source": tmp_dir,
            "dry_run": True,
            "palace_path": palace_path,
        }
    )
    assert res["success"] is True

    col = get_collection(palace_path, create=True)
    assert col.get(where={"source_file": "fake://item/1"})["ids"] == []
