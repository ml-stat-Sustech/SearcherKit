from __future__ import annotations

import argparse
import asyncio
import os
import multiprocessing
from dataclasses import asdict, dataclass
from typing import Any, Mapping, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.cli_args import FrontendArgs


@dataclass(frozen=True)
class VllmServerConfig:
    engine_args: "AsyncEngineArgs"
    frontend_args: "FrontendArgs"
    uvicorn_args: Mapping[str, Any] | None = None
    env: Mapping[str, str] | None = None
    startup_timeout_s: float = 60.0


class VllmServer:
    def __init__(self, config: VllmServerConfig) -> None:
        self.config = config
        self._process: multiprocessing.Process | None = None

    @property
    def port(self) -> int:
        return self.config.frontend_args.port
    
    @property
    def host(self) -> str:
        return self.config.frontend_args.host or "127.0.0.1" # consistency with vllm

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    async def start(self) -> None:
        if self._process and self._process.is_alive():
            return

        self._process = _spawn_vllm_server_process(
            server_args=_build_server_args(
                engine_args=self.config.engine_args,
                frontend_args=self.config.frontend_args,
            ),
            uvicorn_args=self.config.uvicorn_args,
            env=self.config.env,
        )
        self._process.start()

        try:
            await _wait_for_port(
                host="127.0.0.1",
                port=self.port,
                timeout_s=self.config.startup_timeout_s,
            )
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        if not self._process:
            return
        if self._process.is_alive():
            self._process.terminate()
            await asyncio.to_thread(self._process.join, timeout=10)
            if self._process.is_alive():
                self._process.kill()
                await asyncio.to_thread(self._process.join, timeout=10)
        self._process = None

    async def __aenter__(self) -> "VllmServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()


async def _wait_for_port(*, host: str, port: int, timeout_s: float) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"vLLM server did not start on {host}:{port} within {timeout_s}s"
                )
            await asyncio.sleep(0.2)

def _spawn_vllm_server_process(
    *,
    server_args: argparse.Namespace,
    uvicorn_args: Mapping[str, Any] | None,
    env: Mapping[str, str] | None,
) -> multiprocessing.Process:
    def _entry() -> None:
        if env:
            os.environ.update(env)
        from vllm.entrypoints.openai.api_server import run_server
        import uvloop # we specially ensure uvloop here, consistency with vllm

        uvloop.run(run_server(server_args, **(uvicorn_args or {})))

    return multiprocessing.Process(target=_entry, daemon=True)


def _build_server_args(
    *,
    engine_args: "AsyncEngineArgs",
    frontend_args: "FrontendArgs",
) -> argparse.Namespace:
    engine_map = asdict(engine_args)
    frontend_map = asdict(frontend_args)

    merged = {**frontend_map, **engine_map}
    return argparse.Namespace(**merged)


async def serve_vllm_server(config: VllmServerConfig) -> VllmServer:
    server = VllmServer(config)
    await server.start()
    return server
