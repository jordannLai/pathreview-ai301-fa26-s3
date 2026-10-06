"""Tests for VectorStore.delete_profile_chunks (issue #32)."""

from uuid import uuid4

import pytest
from chromadb.errors import NotFoundError

from rag.retriever.vector_store import VectorStore


@pytest.mark.unit
class TestDeleteProfileChunks:
    """Test suite for profile-scoped vector-store deletion."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a VectorStore backed by a throwaway Chroma directory."""
        return VectorStore(persist_dir=str(tmp_path / "chroma"))

    def _seed(self, store, profile_id: str, count: int) -> None:
        """Store chunks the way the ingestion pipeline does: per-profile
        collection, profile_id in every chunk's metadata."""
        collection = store.get_collection(f"profile_{profile_id}")
        collection.add(
            ids=[f"resume_{profile_id}_chunk_{i}" for i in range(count)],
            embeddings=[[0.1 * (i + 1), 0.2, 0.3] for i in range(count)],
            documents=[f"chunk {i} for {profile_id}" for i in range(count)],
            metadatas=[
                {"profile_id": profile_id, "source_type": "resume", "chunk_index": i}
                for i in range(count)
            ],
        )

    def _collection_names(self, store) -> list[str]:
        return [c.name if hasattr(c, "name") else c for c in store.client.list_collections()]

    def test_delete_profile_chunks_removes_profile_collection(self, store):
        """The profile's collection and every chunk tagged with its id are gone."""
        profile_id = str(uuid4())
        self._seed(store, profile_id, count=2)

        deleted = store.delete_profile_chunks(profile_id)

        assert deleted == 2
        assert f"profile_{profile_id}" not in self._collection_names(store)
        with pytest.raises(NotFoundError):
            store.client.get_collection(name=f"profile_{profile_id}")
        for name in self._collection_names(store):
            leftover = store.client.get_collection(name=name).get(where={"profile_id": profile_id})
            assert leftover["ids"] == []

    def test_delete_profile_chunks_leaves_other_profiles(self, store):
        """Deleting profile A leaves profile B's chunks untouched."""
        profile_a, profile_b = str(uuid4()), str(uuid4())
        self._seed(store, profile_a, count=2)
        self._seed(store, profile_b, count=3)

        store.delete_profile_chunks(profile_a)

        assert f"profile_{profile_b}" in self._collection_names(store)
        assert store.client.get_collection(name=f"profile_{profile_b}").count() == 3

    def test_delete_profile_chunks_missing_collection_is_noop(self, store):
        """A profile that never had chunks returns 0, raises nothing, and
        does not leave an empty collection behind."""
        profile_id = str(uuid4())

        deleted = store.delete_profile_chunks(profile_id)

        assert deleted == 0
        assert f"profile_{profile_id}" not in self._collection_names(store)
