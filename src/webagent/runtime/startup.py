from __future__ import annotations

import asyncio
import atexit
import shutil
import subprocess
import sys

from omegaconf import DictConfig

from webagent.log import get_logger

logger = get_logger(__name__)

_es_started_by_us: bool = False
_mcp_proc: subprocess.Popen[bytes] | None = None


async def check_and_start(cfg: DictConfig) -> None:
    if not cfg.get("auto_startup", {}).get("enabled", False):
        return

    auto_cfg = cfg.auto_startup

    if auto_cfg.get("elasticsearch", {}).get("enabled", False):
        await _start_and_check_elasticsearch(auto_cfg.elasticsearch)

    if auto_cfg.get("mcp_server", {}).get("enabled", False):
        global _mcp_proc
        _mcp_proc = await _check_and_start_mcp_server(auto_cfg.mcp_server)

    atexit.register(shutdown)


def shutdown() -> None:
    global _es_started_by_us, _mcp_proc

    if _mcp_proc is not None:
        logger.info("Stopping MCP server (pid=%d)", _mcp_proc.pid)
        _mcp_proc.terminate()
        try:
            _mcp_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _mcp_proc.kill()
            _mcp_proc.wait(timeout=5)
        _mcp_proc = None

    if _es_started_by_us:
        logger.info("Stopping Elasticsearch")
        es_bin = shutil.which("elasticsearch")
        if es_bin is None:
            logger.warning("elasticsearch binary not found, cannot stop")
            return
        try:
            subprocess.run(
                ["pkill", "-f", "org.elasticsearch.bootstrap.Elasticsearch"],
                timeout=15,
            )
            logger.info("Elasticsearch stopped")
        except subprocess.TimeoutExpired:
            logger.warning("Elasticsearch did not stop gracefully, sending SIGKILL")
            subprocess.run(
                ["pkill", "-9", "-f", "org.elasticsearch.bootstrap.Elasticsearch"],
                timeout=10,
            )
        _es_started_by_us = False


async def _start_and_check_elasticsearch(es_cfg: DictConfig) -> None:
    global _es_started_by_us
    host: str = es_cfg.host
    index_name: str | None = es_cfg.get("index_name", None)

    es_bin = shutil.which("elasticsearch")
    if es_bin is None:
        logger.error("elasticsearch binary not found in PATH")
        sys.exit(1)

    from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError

    try:
        es = Elasticsearch(host, request_timeout=10)
        if es.ping():
            info = es.info()
            logger.info("Elasticsearch already running: version=%s", info["version"]["number"])
            await _check_es_index(es, index_name, es_cfg)
            return
    except ESConnectionError:
        pass

    logger.info("Starting Elasticsearch in daemon mode")
    es_opts: dict[str, str] = dict(es_cfg.get("options", {}))
    cmd = [es_bin, "-d"]
    for k, v in es_opts.items():
        cmd.extend([f"-E{k}={v}"])

    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        logger.error("elasticsearch -d exited with code %d", proc.returncode)
        sys.exit(1)

    startup_timeout_s: float = es_cfg.get("startup_timeout_s", 60.0)
    host_parsed = host.replace("http://", "").replace("https://", "")
    es_host, _, es_port_str = host_parsed.partition(":")
    es_port = int(es_port_str) if es_port_str else 9200

    try:
        await _wait_for_port(host=es_host, port=es_port, timeout_s=startup_timeout_s)
    except TimeoutError:
        logger.error("Elasticsearch did not start within %.1fs", startup_timeout_s)
        sys.exit(1)

    _es_started_by_us = True
    logger.info("Elasticsearch started on %s", host)

    es = Elasticsearch(host, request_timeout=10)
    await _check_es_index(es, index_name, es_cfg)


async def _check_es_index(es, index_name: str | None, es_cfg: DictConfig) -> None:
    if index_name is None:
        logger.info("No index_name configured, skipping index check")
        return

    if es.indices.exists(index=index_name):
        count = es.count(index=index_name)["count"]
        logger.info("Elasticsearch index '%s' exists with %d documents", index_name, count)
        return

    logger.warning("Elasticsearch index '%s' does not exist", index_name)
    answer = input(f"Create index '{index_name}'? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        logger.error("Index '%s' not created, exiting", index_name)
        sys.exit(1)

    from webagent.local_wiki.eswiki.wiki2index_links import create_index

    embedding_dim: int = es_cfg.get("embedding_dim", 384)
    include_vector: bool = es_cfg.get("include_vector", False)
    create_index(es, index_name, embedding_dim, include_vector=include_vector)


async def _check_and_start_mcp_server(mcp_cfg: DictConfig) -> subprocess.Popen[bytes] | None:
    host: str = mcp_cfg.host
    port: int = mcp_cfg.port
    workers: int = mcp_cfg.get("workers", 8)
    timeout_s: float = mcp_cfg.get("startup_timeout_s", 30.0)
    env_overrides: dict[str, str] = dict(mcp_cfg.get("env", {}))

    if await _port_open(host, port):
        logger.info("MCP server already running on %s:%s", host, port)
        return None

    if not shutil.which("uvicorn"):
        logger.error("uvicorn not found, cannot start MCP server")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.local_wiki.mcp:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers),
    ]

    import os
    env = os.environ.copy()
    env.update(env_overrides)

    logger.info("Starting MCP server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)

    try:
        await _wait_for_port(host=host, port=port, timeout_s=timeout_s)
        logger.info("MCP server started on %s:%s (pid=%d)", host, port, proc.pid)
    except TimeoutError:
        proc.terminate()
        proc.wait(timeout=10)
        logger.error("MCP server failed to start within %.1fs", timeout_s)
        sys.exit(1)

    return proc


async def _port_open(host: str, port: int) -> bool:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=3.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except OSError:
        return False


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
                    f"Server did not start on {host}:{port} within {timeout_s}s"
                )
            await asyncio.sleep(0.5)
