import asyncio

from rcs.batching.dynamic import DynamicBatcher


def test_batcher_returns_every_result() -> None:
    async def run(items: list[int]) -> list[int]:
        return [item * 2 for item in items]

    async def scenario() -> list[int]:
        batcher = DynamicBatcher(run, max_batch=4, wait_ms=1)
        await batcher.start()
        values = await asyncio.gather(*(batcher.submit(i) for i in range(8)))
        batcher.accepting = False
        batcher.task.cancel()
        return values

    assert asyncio.run(scenario()) == list(range(0, 16, 2))
