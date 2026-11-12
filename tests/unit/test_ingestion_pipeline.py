import json
from types import SimpleNamespace

from backend.ingestion.pipeline import collection_name, token_summary
from backend.retrieval.vector_store import index_is_current


class FakeClient:
    def __init__(self, point_count: int):
        self.point_count = point_count

    def collection_exists(self, collection: str) -> bool:
        return collection == "manufacturing_intermediate_c1_m1"

    def count(self, collection: str, exact: bool):
        return SimpleNamespace(count=self.point_count)


def test_collection_name_uses_selected_settings():
    assert collection_name("manufacturing_intermediate", "c1", "m1") == (
        "manufacturing_intermediate_c1_m1"
    )


def test_token_summary_handles_even_count():
    assert token_summary([4, 1, 3, 2]) == {
        "min": 1,
        "median": 2.5,
        "max": 4,
    }


def test_index_is_current_requires_matching_record_and_point_count(tmp_path):
    record = {
        "collection": "manufacturing_intermediate_c1_m1",
        "chunk_count": 2,
    }
    record_path = tmp_path / "index_m1.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert index_is_current(
        FakeClient(point_count=2),
        record["collection"],
        record_path,
        record,
    )
    assert not index_is_current(
        FakeClient(point_count=1),
        record["collection"],
        record_path,
        record,
    )
