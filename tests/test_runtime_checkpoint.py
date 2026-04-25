from __future__ import annotations

import asyncio

from searchagent.runtime.checkpoint import CheckpointStore


def test_checkpoint_store_records_completed_sample(tmp_path) -> None:
    async def run_case() -> None:
        record_path = tmp_path / "000000.json"
        record_path.write_text("{}", encoding="utf-8")
        store = CheckpointStore.from_output_dir(tmp_path)

        await store.start_run(run_id="run-1", output_dir=tmp_path)
        await store.mark_pending(
            sample_id="000000",
            index=0,
            trace_id="trace-1",
            record_path=record_path,
            prompt="query",
        )
        await store.mark_started(
            sample_id="000000",
            run_id="run-1",
            trace_id="trace-1",
            started_at="2026-04-25T00:00:00",
        )
        await store.mark_completed(
            sample_id="000000",
            stats={"turns": 2, "tool_calls": 1},
            record_path=record_path,
            ended_at="2026-04-25T00:00:01",
            elapsed=1.0,
        )

        assert store.path.exists()
        assert store.sample_state("000000")["status"] == "completed"
        assert store.is_completed("000000", record_path)

    asyncio.run(run_case())


def test_checkpoint_store_identifies_resume_candidates(tmp_path) -> None:
    async def run_case() -> None:
        store = CheckpointStore.from_output_dir(tmp_path)
        await store.mark_pending(
            sample_id="000001",
            index=1,
            trace_id="trace-2",
            record_path=tmp_path / "000001.json",
            prompt="query",
        )
        await store.mark_started(
            sample_id="000001",
            run_id="run-1",
            trace_id="trace-2",
            started_at="2026-04-25T00:00:00",
        )

        resumed_store = CheckpointStore.from_output_dir(tmp_path)

        assert resumed_store.is_resume_candidate("000001")
        assert resumed_store.sample_state("000001")["attempts"] == 1

    asyncio.run(run_case())
