"""Tests for the Redis-backed job tracker."""

import pytest

from proxy.services.job_tracker import JobTracker


@pytest.fixture
async def tracker(fake_redis):
    """JobTracker backed by fakeredis."""
    t = JobTracker.__new__(JobTracker)
    t.redis = fake_redis
    return t


class TestJobTracker:
    async def test_create_job(self, tracker):
        job = await tracker.create_job("job-1", "test-collection", filename="doc.pdf")
        assert job["job_id"] == "job-1"
        assert job["status"] == "pending"
        assert job["collection"] == "test-collection"
        assert job["filename"] == "doc.pdf"
        assert job["progress"] == "0"

    async def test_get_status_after_create(self, tracker):
        await tracker.create_job("job-2", "col")
        status = await tracker.get_status("job-2")
        assert status is not None
        assert status["job_id"] == "job-2"
        assert status["status"] == "pending"
        assert status["progress"] == 0.0

    async def test_get_status_nonexistent(self, tracker):
        status = await tracker.get_status("nonexistent")
        assert status is None

    async def test_update_progress(self, tracker):
        await tracker.create_job("job-3", "col")
        await tracker.update_progress("job-3", 5, 10)

        status = await tracker.get_status("job-3")
        assert status["status"] == "processing"
        assert status["processed_pages"] == 5
        assert status["total_pages"] == 10
        assert status["progress"] == 0.5

    async def test_complete_job(self, tracker):
        await tracker.create_job("job-4", "col")
        await tracker.complete_job("job-4", total_pages=10, indexed_pages=10)

        status = await tracker.get_status("job-4")
        assert status["status"] == "completed"
        assert status["progress"] == 1.0

    async def test_complete_job_partial(self, tracker):
        await tracker.create_job("job-5", "col")
        await tracker.complete_job("job-5", total_pages=10, indexed_pages=7)

        status = await tracker.get_status("job-5")
        assert status["status"] == "partial"

    async def test_fail_job(self, tracker):
        await tracker.create_job("job-6", "col")
        await tracker.fail_job("job-6", "Some error happened")

        status = await tracker.get_status("job-6")
        assert status["status"] == "failed"
        assert "Some error" in status["error"]

    async def test_fail_job_truncates_error(self, tracker):
        await tracker.create_job("job-7", "col")
        long_error = "x" * 2000
        await tracker.fail_job("job-7", long_error)

        status = await tracker.get_status("job-7")
        assert len(status["error"]) <= 1000

    async def test_ping(self, tracker):
        result = await tracker.ping()
        assert result is True

    async def test_key_format(self, tracker):
        assert tracker._key("abc") == "vrag:job:abc"

    async def test_progress_zero_division(self, tracker):
        await tracker.create_job("job-8", "col")
        # total_pages=0 should not raise
        await tracker.update_progress("job-8", 0, 0)
        status = await tracker.get_status("job-8")
        assert status["progress"] == 0.0
