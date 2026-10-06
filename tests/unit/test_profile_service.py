"""Tests for profile_service.delete_profile vector-store cleanup (issue #32)."""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from core.services.profile_service import delete_profile


@pytest.mark.unit
class TestDeleteProfile:
    """Test suite for delete_profile."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock async database session with no related rows."""
        session = AsyncMock()
        session.delete = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        result = Mock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)
        return session

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        store = Mock()
        store.delete_profile_chunks = Mock(return_value=1)
        return store

    @pytest.mark.asyncio
    async def test_delete_profile_cleans_vector_store_before_commit(
        self, mock_db_session, mock_vector_store
    ):
        """The profile's chunks are deleted exactly once, before db.commit."""
        profile_id = uuid4()
        calls = Mock()
        calls.attach_mock(mock_vector_store.delete_profile_chunks, "cleanup")
        calls.attach_mock(mock_db_session.commit, "commit")

        with patch(
            "core.services.profile_service.get_profile",
            AsyncMock(return_value=Mock(id=profile_id)),
        ):
            deleted = await delete_profile(
                mock_db_session, profile_id, uuid4(), vector_store=mock_vector_store
            )

        assert deleted is True
        mock_vector_store.delete_profile_chunks.assert_called_once_with(str(profile_id))
        mock_db_session.commit.assert_awaited_once()
        call_names = [c[0] for c in calls.mock_calls]
        assert call_names.index("cleanup") < call_names.index("commit")

    @pytest.mark.asyncio
    async def test_delete_profile_cleanup_failure_rolls_back(
        self, mock_db_session, mock_vector_store
    ):
        """If the vector-store cleanup fails, the DB delete is rolled back,
        never committed, and the error propagates."""
        mock_vector_store.delete_profile_chunks.side_effect = RuntimeError("chroma down")

        with (
            patch(
                "core.services.profile_service.get_profile",
                AsyncMock(return_value=Mock()),
            ),
            pytest.raises(RuntimeError, match="chroma down"),
        ):
            await delete_profile(mock_db_session, uuid4(), uuid4(), vector_store=mock_vector_store)

        mock_db_session.rollback.assert_awaited_once()
        mock_db_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_profile_not_found_skips_vector_store(
        self, mock_db_session, mock_vector_store
    ):
        """A missing or not-owned profile returns False and never touches
        the vector store."""
        with patch(
            "core.services.profile_service.get_profile",
            AsyncMock(return_value=None),
        ):
            deleted = await delete_profile(
                mock_db_session, uuid4(), uuid4(), vector_store=mock_vector_store
            )

        assert deleted is False
        mock_vector_store.delete_profile_chunks.assert_not_called()
        mock_db_session.commit.assert_not_awaited()
