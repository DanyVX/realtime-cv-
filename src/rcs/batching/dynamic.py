"""Bounded, deadline-aware dynamic batching."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Item(Generic[T, R]):
    value: T
    deadline: float
    future: asyncio.Future[R]


class DynamicBatcher(Generic[T, R]):
    def __init__(
        self,
        run: Callable[[list[T]], Awaitable[list[R]]],
        capacity: int = 128,
        max_batch: int = 8,
        wait_ms: int = 5,
    ):
        self.run, self.queue, self.max_batch, self.wait = (
            run,
            asyncio.Queue[Item[T, R]](capacity),
            max_batch,
            wait_ms / 1000,
        )
        self.task: asyncio.Task[None] | None = None
        self.accepting = True

    async def start(self) -> None:
        self.task = asyncio.create_task(self._worker())

    async def submit(self, value: T, timeout_s: float = 5) -> R:
        if not self.accepting:
            raise RuntimeError("service draining")
        if self.queue.full():
            raise OverflowError("queue saturated")
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        await self.queue.put(Item(value, monotonic() + timeout_s, future))
        return await future

    async def _worker(self) -> None:
        while self.accepting or not self.queue.empty():
            first = await self.queue.get()
            batch = [first]
            end = monotonic() + self.wait
            while len(batch) < self.max_batch and monotonic() < end:
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), end - monotonic()))
                except TimeoutError:
                    break
            alive = [
                item
                for item in batch
                if item.deadline > monotonic() and not item.future.cancelled()
            ]
            for item in batch:
                if item not in alive and not item.future.done():
                    item.future.set_exception(TimeoutError("deadline exceeded"))
            if alive:
                try:
                    for item, result in zip(
                        alive, await self.run([i.value for i in alive]), strict=True
                    ):
                        if not item.future.done():
                            item.future.set_result(result)
                except Exception as error:
                    for item in alive:
                        if not item.future.done():
                            item.future.set_exception(error)

    async def close(self) -> None:
        self.accepting = False
        if self.task:
            await self.task
