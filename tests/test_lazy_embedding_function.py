"""A store can bind an embedding function without paying for one.

A store process must hand chromadb an embedding function — 1.x does not persist it, so a reader that
omits it silently gets the library default and its queries stop matching the writer's vectors. But
"must bind" and "must load a model" name different requirements, and the eager factory fuses
them: opening one content palace imports eleven onnxruntime modules and builds a MiniLM session that
nothing calls. A host standing several palaces pays that once per store process.

`MEMPALACE_LAZY_EMBEDDER=1` binds without building. It stays OPT-IN because chroma's
embedder-mismatch detection behaves differently against a proxy, and that guard protects against
querying an incomparable vector space — worth more than the model costs. A caller opts in only when
it supplies every vector itself.
"""

import os
import subprocess
import sys

from mempalace.backends.base import PalaceRef
from mempalace.backends.chroma import ChromaBackend
from mempalace.embedding import LazyEmbeddingFunction, get_lazy_embedding_function

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = """
import os, sys, tempfile
sys.path.insert(0, {repo!r})
os.environ["MEMPALACE_LAZY_EMBEDDER"] = {flag!r}
from mempalace.backends.base import PalaceRef
from mempalace.backends.chroma import ChromaBackend
d = tempfile.mkdtemp()
ChromaBackend().get_collection(palace=PalaceRef(id=d, local_path=d),
                               collection_name="probe_collection", create=True)
print("onnxruntime" in sys.modules)
"""


def _opened_with_a_model(flag: str) -> bool:
    """Open a collection in a FRESH process and report whether a model loaded.

    A subprocess rather than an in-process check, because "no model loaded" describes a process, not
    a call: any earlier test that embedded leaves onnxruntime in `sys.modules`, and an in-process
    assertion would then pass or fail on test ordering rather than on the code under it.
    """
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=_REPO, flag=flag)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, f"probe failed: {out.stderr[-400:]}"
    return out.stdout.strip().splitlines()[-1] == "True"


def test_opting_in_opens_a_collection_without_loading_a_model():
    """THE GUARD, carrying its own control.

    Both arms run, so a green here cannot come from a probe that quietly stopped measuring: the
    opt-in arm must load nothing, and the default arm must still load a model.
    """
    assert _opened_with_a_model("1") is False, (
        "opting in still built an embedding model — a store that supplies its own vectors needs the "
        "function bound, never constructed"
    )
    assert _opened_with_a_model("0") is True, (
        "the control arm loaded no model either, so this probe proves nothing about the opt-in"
    )


def test_the_model_still_builds_when_something_actually_embeds():
    """Deferral must not become refusal — a caller that embeds pays the usual price, one call later."""
    lazy = get_lazy_embedding_function()
    assert isinstance(lazy, LazyEmbeddingFunction)
    assert lazy.built is False
    out = lazy(["a line of text"])
    assert lazy.built is True
    assert len(out) == 1 and len(out[0]) > 0


def test_the_attributes_chroma_reads_answer_without_building():
    """chroma reads both at collection creation; answering either by building defeats the deferral."""
    lazy = get_lazy_embedding_function()
    assert lazy.is_legacy is False
    assert lazy.name() == LazyEmbeddingFunction.CHROMA_EF_NAME
    assert lazy.built is False


def test_a_bound_store_round_trips_a_caller_vector(tmp_path):
    """The binding stays real: writes land and read back at the caller's width."""
    os.environ["MEMPALACE_LAZY_EMBEDDER"] = "1"
    try:
        ref = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
        col = ChromaBackend().get_collection(
            palace=ref, collection_name="lazy_round_trip", create=True
        )
        col.upsert(
            ids=["a"],
            documents=["body"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"wing": "w", "room": "r"}],
        )
        got = col.get(ids=["a"], include=["embeddings"])
        assert len(got["embeddings"][0]) == 3
    finally:
        os.environ.pop("MEMPALACE_LAZY_EMBEDDER", None)
