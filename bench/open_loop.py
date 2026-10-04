"""Open-loop HTTP load generator; measures latency from intended send time."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx


async def run(url: str, image: Path, rate: float, count: int) -> list[float]:
    data = image.read_bytes()
    started = time.perf_counter()
    delays = []
    async with httpx.AsyncClient(timeout=20) as client:
        for index in range(count):
            intended = started + index / rate
            await asyncio.sleep(max(0, intended - time.perf_counter()))
            response = await client.post(url, files={"image": ("input.jpg", data, "image/jpeg")})
            delays.append(time.perf_counter() - intended)
            response.raise_for_status()
    return delays


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/detect")
    parser.add_argument("--rate", type=float, default=1)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    print(asyncio.run(run(args.url, args.image, args.rate, args.count)))
