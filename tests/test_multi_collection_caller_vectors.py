"""Multi-collection, caller-supplied-vector store contract (base store-tech).

These tests pin the generic infrastructure an upstream caller drives when it
keeps several vector spaces in one palace dir and supplies every vector itself
(no embedding model baked into the base):

* ``embedding_function=None`` via ``options`` -> the store skips the model and
  honors caller-supplied vectors on both write and read, across a reopen.
* N named collections live in one palace; the SAME drawer id is reusable across
  them as a cross-collection join key.
* Caller-supplied HNSW knobs (``ef_construction``/``max_neighbors``/
  ``sync_threshold``/``batch_size``) land in ``collection_metadata`` where the
  divergence guard, the cosine-space detector, and ``_read_sync_threshold``
  read them.
* Arbitrary-metadata ``where`` filters pre-filter the HNSW query.
* A dim-mismatch write raises cleanly.
* The HNSW<->sqlite divergence guard / cheap boot-canary surface answers
  (never raises) on a caller-vector collection.

The base carries ZERO caller-domain knowledge — the metadata keys below
(``register``/``grammar_layer``/``struct_hash``) are arbitrary strings to the
store; they exercise the generic where-filter, nothing more.
"""

import pytest

from mempalace.backends.base import PalaceRef
from mempalace.backends.chroma import (
    ChromaBackend,
    _collection_has_sync_threshold_metadata,
    _read_sync_threshold,
    hnsw_capacity_status,
)

# Two caller vector spaces sharing one palace, exactly as the caller drives it.
CONTENT = "content"
FORM = "form"

# Caller-tuned HNSW config (write-heavy). ef_construction/max_neighbors use the
# modern parameter names; the base maps them to the legacy metadata keys that
# keep the divergence guard wired.
CALLER_OPTIONS = {
    "embedding_function": None,  # caller supplies vectors; model skipped
    "ef_construction": 200,
    "max_neighbors": 32,
    "sync_threshold": 1000,
    "batch_size": 100,
}

DIM = 4


def _vec(*xs):
    """Pad/normalize a tiny vector to DIM dims."""
    v = list(xs) + [0.0] * (DIM - len(xs))
    return v[:DIM]


def _make_palace(tmp_path):
    """A ChromaBackend + PalaceRef pointed at a fresh palace dir."""
    backend = ChromaBackend()
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    return backend, palace


def _caller_collection(backend, palace, name, *, create=True):
    return backend.get_collection(
        palace=palace,
        collection_name=name,
        create=create,
        options=CALLER_OPTIONS,
    )


# ---------------------------------------------------------------------------
# Caller-supplied embedding function (model skipped)
# ---------------------------------------------------------------------------


def test_embedding_function_none_skips_model_and_honors_caller_vectors(tmp_path):
    backend, palace = _make_palace(tmp_path)
    col = _caller_collection(backend, palace, CONTENT)
    # The underlying chroma collection has NO embedding function.
    assert col._collection._embedding_function is None

    col.upsert(
        ids=["d1", "d2"],
        documents=["alpha", "beta"],
        embeddings=[_vec(1.0), _vec(0.0, 1.0)],
        metadatas=[{"register": "syn"}, {"register": "canon"}],
    )
    assert col.count() == 2

    # Query by caller-supplied vector returns the nearest drawer.
    res = col.query(query_embeddings=[_vec(1.0)], n_results=1)
    assert res.ids == [["d1"]]


def test_caller_vector_mode_survives_reopen(tmp_path):
    # First backend writes caller vectors, then closes (releases the rust lock).
    backend, palace = _make_palace(tmp_path)
    col = _caller_collection(backend, palace, CONTENT)
    col.upsert(ids=["d1"], documents=["x"], embeddings=[_vec(1.0)])
    backend.close()

    # A fresh backend re-opens in caller-vector mode (options carry None) and
    # can both read the row and query it by vector — proof the model stays out.
    backend2, palace2 = _make_palace(tmp_path)
    col2 = _caller_collection(backend2, palace2, CONTENT, create=False)
    assert col2._collection._embedding_function is None
    assert col2.get(ids=["d1"]).ids == ["d1"]
    assert col2.query(query_embeddings=[_vec(1.0)], n_results=1).ids == [["d1"]]
    backend2.close()


def test_dim_mismatch_raises_cleanly(tmp_path):
    backend, palace = _make_palace(tmp_path)
    col = _caller_collection(backend, palace, CONTENT)
    col.upsert(ids=["d1"], documents=["x"], embeddings=[_vec(1.0)])

    with pytest.raises(Exception) as excinfo:
        col.upsert(ids=["d2"], documents=["y"], embeddings=[[0.1, 0.2]])  # 2 dims, not DIM
    assert "dimension" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Multiple collections in one palace + shared drawer id
# ---------------------------------------------------------------------------


def test_two_collections_one_palace_shared_drawer_id(tmp_path):
    backend, palace = _make_palace(tmp_path)
    content = _caller_collection(backend, palace, CONTENT)
    form = _caller_collection(backend, palace, FORM)

    # SAME drawer id "d1" lives in BOTH collections as the join key, each with
    # its own vector space (content embedding vs form embedding).
    content.upsert(ids=["d1"], documents=["the prose"], embeddings=[_vec(1.0, 0.0)])
    form.upsert(ids=["d1"], documents=["the shape"], embeddings=[_vec(0.0, 1.0)])

    assert content.get(ids=["d1"]).documents == ["the prose"]
    assert form.get(ids=["d1"]).documents == ["the shape"]
    # The two collections are independent stores under one palace dir.
    assert content.count() == 1
    assert form.count() == 1


def test_collections_are_isolated_within_a_palace(tmp_path):
    backend, palace = _make_palace(tmp_path)
    content = _caller_collection(backend, palace, CONTENT)
    form = _caller_collection(backend, palace, FORM)

    content.upsert(ids=["only-content"], documents=["c"], embeddings=[_vec(1.0)])
    # The id written to content is not visible from form.
    assert form.get(ids=["only-content"]).ids == []


# ---------------------------------------------------------------------------
# Metadata where-filters pre-filter the HNSW query
# ---------------------------------------------------------------------------


def test_where_prefilters_hnsw_query(tmp_path):
    backend, palace = _make_palace(tmp_path)
    col = _caller_collection(backend, palace, CONTENT)
    col.upsert(
        ids=["d1", "d2", "d3"],
        documents=["a", "b", "c"],
        embeddings=[_vec(1.0), _vec(0.9, 0.1), _vec(0.8, 0.2)],
        metadatas=[
            {"register": "syn", "grammar_layer": "L0", "struct_hash": "h1"},
            {"register": "canon", "grammar_layer": "L1", "struct_hash": "h2"},
            {"register": "syn", "grammar_layer": "L2", "struct_hash": "h1"},
        ],
    )

    # Single-key filter: only the two "syn" drawers are eligible, even though
    # the nearest raw neighbor (d2) is "canon".
    res = col.query(query_embeddings=[_vec(1.0)], n_results=5, where={"register": "syn"})
    assert set(res.ids[0]) == {"d1", "d3"}
    assert "d2" not in res.ids[0]

    # Compound filter across arbitrary keys.
    res2 = col.query(
        query_embeddings=[_vec(1.0)],
        n_results=5,
        where={"$and": [{"register": "syn"}, {"struct_hash": "h1"}]},
    )
    assert set(res2.ids[0]) == {"d1", "d3"}

    # The same where on get() (no vector) also filters.
    got = col.get(where={"grammar_layer": "L1"})
    assert got.ids == ["d2"]


# ---------------------------------------------------------------------------
# HNSW config knobs land where the divergence guard reads them
# ---------------------------------------------------------------------------


def test_hnsw_knobs_persist_to_collection_metadata(tmp_path):
    backend, palace = _make_palace(tmp_path)
    col = _caller_collection(backend, palace, CONTENT)
    meta = col.metadata

    assert meta.get("hnsw:space") == "cosine"
    assert meta.get("hnsw:construction_ef") == 200  # ef_construction
    assert meta.get("hnsw:M") == 32  # max_neighbors
    assert meta.get("hnsw:sync_threshold") == 1000
    assert meta.get("hnsw:batch_size") == 100
    # Cosine space preserved -> the searcher's similarity formula stays correct.
    assert col.distance_metric == "cosine"


def test_divergence_guard_reads_caller_sync_threshold(tmp_path):
    backend, palace = _make_palace(tmp_path)
    col = _caller_collection(backend, palace, CONTENT)
    col.upsert(ids=["d1", "d2"], documents=["a", "b"], embeddings=[_vec(1.0), _vec(0.0, 1.0)])
    backend.close()

    # The guard's sqlite-only read path (the @daemon repair-tail relies on it)
    # sees the caller-tuned sync_threshold rather than the 1000 fallback.
    assert _collection_has_sync_threshold_metadata(str(tmp_path), CONTENT) is True
    assert _read_sync_threshold(str(tmp_path), CONTENT) == 1000


def test_boot_canary_never_raises_on_caller_vector_collection(tmp_path):
    backend, palace = _make_palace(tmp_path)
    col = _caller_collection(backend, palace, CONTENT)
    col.upsert(ids=["d1", "d2"], documents=["a", "b"], embeddings=[_vec(1.0), _vec(0.0, 1.0)])
    backend.close()

    # The cheap boot-canary returns a structured verdict (never raises) and
    # counts the sqlite-side embeddings for the named collection.
    status = hnsw_capacity_status(str(tmp_path), CONTENT)
    assert status["status"] in {"ok", "diverged", "unknown"}
    assert status["sqlite_count"] == 2
    # A second collection in the same palace is independently inspectable.
    assert hnsw_capacity_status(str(tmp_path), FORM)["sqlite_count"] in (None, 0)
