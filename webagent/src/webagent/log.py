from __future__ import annotations

import logging
import os

DEFAULT_LOGGER_NAME = "webagent"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_ENV_VARS = ("WEBAGENT_LOG_LEVEL", "LOG_LEVEL")
LOG_FORMAT = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONSOLE_HANDLER_NAME = "webagent.console"


def _resolve_log_level(level: str | int | None = None) -> int:
    if isinstance(level, int):
        return level

    level_name = level
    if level_name is None:
        for env_var in LOG_LEVEL_ENV_VARS:
            env_value = os.getenv(env_var)
            if env_value:
                level_name = env_value
                break
    level_name = (level_name or DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    level: str | int | None = None,
) -> logging.Logger:
    resolved_level = _resolve_log_level(level)
    base_logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    base_logger.setLevel(resolved_level)
    base_logger.propagate = False

    handler = None
    for existing_handler in base_logger.handlers:
        if existing_handler.get_name() == _CONSOLE_HANDLER_NAME:
            handler = existing_handler
            break

    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_CONSOLE_HANDLER_NAME)
        base_logger.addHandler(handler)

    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logger = logging.getLogger(name)
    if name.startswith(f"{DEFAULT_LOGGER_NAME}."):
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
    else:
        logger.setLevel(resolved_level)
        logger.propagate = False
        if logger is not base_logger and not logger.handlers:
            logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    target_name = name or DEFAULT_LOGGER_NAME
    return setup_logger(target_name)


logger = get_logger()

