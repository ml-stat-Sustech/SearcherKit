from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from omegaconf import DictConfig

from searchagent.log import get_logger

logger = get_logger(__name__)

_es_started_by_us: bool = False


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    if is_dataclass(cfg):
        return asdict(cfg).get(key, default)
    return getattr(cfg, key, default)


async def check_and_start(cfg: Any) -> None:
    auto_cfg = _cfg_get(cfg, "auto_startup", {})
    if not _cfg_get(auto_cfg, "enabled", False):
        return

    elasticsearch_cfg = _cfg_get(auto_cfg, "elasticsearch", {})
    if _cfg_get(elasticsearch_cfg, "enabled", False):
        await _start_and_check_elasticsearch(elasticsearch_cfg)

async def shutdown() -> None:
    def _shutdown():
        global _es_started_by_us

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

    await asyncio.to_thread(_shutdown)

async def _start_and_check_elasticsearch(es_cfg: Any) -> None:
    global _es_started_by_us
    host: str = _cfg_get(es_cfg, "host")
    index_name: str | None = _cfg_get(es_cfg, "index_name", None)

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

    es_bin = shutil.which("elasticsearch")
    if es_bin is None:
        logger.error("elasticsearch binary not found in PATH")
        sys.exit(1)

    es_opts: dict[str, str] = dict(_cfg_get(es_cfg, "options", {}))
    cmd = [es_bin, "-d"]
    for k, v in es_opts.items():
        cmd.extend([f"-E{k}={v}"])

    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        logger.error("elasticsearch -d exited with code %d", proc.returncode)
        sys.exit(1)

    startup_timeout_s: float = _cfg_get(es_cfg, "startup_timeout_s", 60.0)
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


async def _check_es_index(es: Any, index_name: str | None, es_cfg: Any) -> None:
    if index_name is None:
        logger.info("No index_name configured, skipping index check")
        return

    if es.indices.exists(index=index_name):
        count = es.count(index=index_name)["count"]
        logger.info("Elasticsearch index '%s' exists with %d documents", index_name, count)
        return

    corpus_type: str = _cfg_get(es_cfg, "corpus_type", "wiki")

    if corpus_type == "wiki":
        wiki_dump_path: str | None = _cfg_get(es_cfg, "wiki_dump_path", None)
        if wiki_dump_path is None:
            logger.error(
                "Elasticsearch index '%s' does not exist and wiki_dump_path is not configured. "
                "Either set wiki_dump_path to build the index, or create it manually.",
                index_name,
            )
            sys.exit(1)
        source_desc = wiki_dump_path
    elif corpus_type == "bcp":
        dataset_path: str = _cfg_get(es_cfg, "bcp_dataset_path", "Tevatron/browsecomp-plus-corpus")
        source_desc = dataset_path
    else:
        logger.error("Unknown corpus_type '%s', expected 'wiki' or 'bcp'", corpus_type)
        sys.exit(1)

    logger.warning("Elasticsearch index '%s' does not exist", index_name)
    answer = input(f"Build index '{index_name}' from '{source_desc}' (corpus_type={corpus_type})? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        logger.error("Index '%s' not created, exiting", index_name)
        sys.exit(1)

    host: str = _cfg_get(es_cfg, "host")
    include_vector: bool = _cfg_get(es_cfg, "include_vector", False)
    embedding_dim: int = _cfg_get(es_cfg, "embedding_dim", 1024)
    model_name: str = _cfg_get(es_cfg, "model_name", "")
    prompt_strategy: str = _cfg_get(es_cfg, "prompt_strategy", "none")
    cpu_batch_size: int = _cfg_get(es_cfg, "cpu_batch_size", 200)
    gpu_batch_size: int = _cfg_get(es_cfg, "gpu_batch_size", 20)

    from pathlib import Path
    base = Path(__file__).resolve().parent.parent / "plugins" / "local_wiki"

    if corpus_type == "wiki":
        script_path = base / "deploy_elasticsearch.py"
        cmd = [
            sys.executable, str(script_path),
            "--es_host", host,
            "--index_name", index_name,
            "--wiki_dump_path", wiki_dump_path,
            "--model_name", model_name,
            "--embedding_dim", str(embedding_dim),
            "--prompt_strategy", prompt_strategy,
            "--cpu_batch_size", str(cpu_batch_size),
            "--gpu_batch_size", str(gpu_batch_size),
        ]
        if include_vector:
            cmd.append("--dense-vector")
    else:
        script_path = Path(__file__).resolve().parent.parent / "plugins" / "browsecomp_plus" / "deploy_elasticsearch.py"
        cmd = [
            sys.executable, str(script_path),
            "--es_host", host,
            "--index_name", index_name,
            "--dataset_path", dataset_path,
            "--model_name", model_name,
            "--embedding_dim", str(embedding_dim),
            "--prompt_strategy", prompt_strategy,
            "--cpu_batch_size", str(cpu_batch_size),
            "--gpu_batch_size", str(gpu_batch_size),
        ]
        if include_vector:
            cmd.append("--dense-vector")

    import os
    env = os.environ.copy()
    env_overrides: dict[str, str] = dict(_cfg_get(es_cfg, "env", {}))
    env.update(env_overrides)

    logger.info("Building index (%s): %s", corpus_type, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    returncode = await proc.wait()
    if returncode != 0:
        logger.error("Index build script exited with code %d", returncode)
        sys.exit(1)
    logger.info("Index '%s' built successfully", index_name)


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
