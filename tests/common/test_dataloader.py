import json
from collections.abc import Callable
from pathlib import Path

import pytest

from searcherkit.common.dataloader import DataConfig, GenericDataLoader


FIXTURE_PATH = Path("tests/fixtures/datasets/openseeker.jsonl")
PARQUET_FIXTURE_PATH = Path("tests/fixtures/datasets/openseeker.parquet")


def _direct_loader(
    path: Path,
    fmt: str,
    *,
    max_items: int | None = None,
) -> GenericDataLoader:
    return GenericDataLoader(
        str(path),
        fmt,  # type: ignore[arg-type]
        input_key="question",
        answer_key="answer",
        max_items=max_items,
    )


def _config_loader(
    path: Path,
    fmt: str,
    *,
    max_items: int | None = None,
) -> GenericDataLoader:
    return GenericDataLoader(
        config=DataConfig(
            source=str(path),
            fmt=fmt,
            input_key="question",
            answer_key="answer",
            max_items=max_items,
        )
    )


@pytest.mark.parametrize("loader_factory", [_direct_loader, _config_loader])
def test_yield_inputs_jsonl(loader_factory: Callable[..., GenericDataLoader]) -> None:
    loader = loader_factory(FIXTURE_PATH, "jsonl")

    item = next(loader.yield_inputs())

    assert item[0].startswith("Identify the individual satisfying all constraints")
    assert item[1] is None
    assert item[2] == "Carl von Ossietzky"


@pytest.mark.parametrize("loader_factory", [_direct_loader, _config_loader])
def test_yield_inputs_parquet(loader_factory: Callable[..., GenericDataLoader]) -> None:
    records = [
        {"question": record["question"], "answer": record["answer"]}
        for record in _read_jsonl(FIXTURE_PATH)
    ]
    loader = loader_factory(PARQUET_FIXTURE_PATH, "parquet")

    items = list(loader.yield_inputs())

    assert items == [(record["question"], None, record["answer"]) for record in records]


@pytest.mark.parametrize("loader_factory", [_direct_loader, _config_loader])
def test_max_items_limits_yielded_inputs(loader_factory: Callable[..., GenericDataLoader]) -> None:
    loader = loader_factory(FIXTURE_PATH, "jsonl", max_items=1)

    iterator = loader.yield_inputs()
    items = [next(iterator)]

    assert len(items) == 1
    assert items[0][2] == "Carl von Ossietzky"
    with pytest.raises(StopIteration):
        next(iterator)


def test_errors(tmp_path: Path) -> None:
    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid json at line 1"):
        list(
            GenericDataLoader(
                str(invalid_jsonl),
                "jsonl",
                input_key="question",
                answer_key="answer",
            ).yield_inputs()
        )

    missing_input = tmp_path / "missing-input.jsonl"
    missing_input.write_text('{"answer": "missing question"}\n', encoding="utf-8")
    with pytest.raises(KeyError, match="missing input_key 'question'"):
        list(
            GenericDataLoader(
                str(missing_input),
                "jsonl",
                input_key="question",
                answer_key="answer",
            ).yield_inputs()
        )

    with pytest.raises(ValueError, match="unsupported fmt"):
        list(GenericDataLoader(str(FIXTURE_PATH), "csv").yield_inputs())  # type: ignore


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
