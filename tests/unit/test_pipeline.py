"""RAGPipeline의 검색 후보 선택 동작을 검사한다."""

from contextlib import contextmanager

import pytest

from backend import pipeline as pipeline_module


class FakeEmbedder:
    model_code = "m1"

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeObservation:
    def __init__(self):
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuse:
    def __init__(self, events=None):
        self.observations = []
        self.events = events if events is not None else []

    def create_trace_id(self, seed):
        return f"trace-{seed}"

    def start_as_current_observation(self, **kwargs):
        self.events.append(kwargs["name"])
        observation = FakeObservation()
        observation.start_kwargs = kwargs
        self.observations.append(observation)
        return observation


def test_retrieve_uses_final_k_and_keeps_all_candidates(monkeypatch):
    candidates = [
        {
            "chunk_id": f"c{index}",
            "book_id": "MB01",
            "text": f"청크 {index}",
            "score": 1.0 / index,
        }
        for index in range(1, 6)
    ]
    captured = {}

    def fake_search(client, collection, query_vector, limit, exact):
        captured["collection"] = collection
        return candidates

    def fake_fill_context(selected, counter, token_budget):
        captured["selected"] = selected
        captured["token_budget"] = token_budget
        return selected, 1200

    monkeypatch.setattr(
        pipeline_module,
        "search",
        fake_search,
    )
    monkeypatch.setattr(
        pipeline_module,
        "fill_context",
        fake_fill_context,
    )

    pipeline = pipeline_module.RAGPipeline.__new__(
        pipeline_module.RAGPipeline
    )
    pipeline.config = {
        "dataset": "manufacturing_basic",
        "chunk_version": "c3",
        "final_k": 3,
        "context_token_budget": 2373,
    }
    pipeline.retrieval_configs = {
        "naive": {
            "retrieval_version": "r1",
            "method": "dense",
            "candidate_count": 20,
            "exact": True,
        }
    }
    pipeline.embedder = FakeEmbedder()
    pipeline.client = object()
    pipeline.counter = object()

    result = pipeline.retrieve(
        "질문",
        "naive",
    )

    assert [
        candidate["chunk_id"]
        for candidate in captured["selected"]
    ] == ["c1", "c2", "c3"]
    assert captured["token_budget"] == 2373
    assert captured["collection"] == "manufacturing_basic_c3_m1"

    assert result["final_k"] == 3
    assert len(result["candidates"]) == 5
    assert len(result["context"]) == 3
    assert result["context_token_count"] == 1200


def test_query_connects_service_user_and_session_to_langfuse(monkeypatch):
    attributes = {}
    events = []

    @contextmanager
    def fake_attributes(**kwargs):
        events.append("attributes")
        attributes.update(kwargs)
        yield

    monkeypatch.setattr(pipeline_module, "propagate_attributes", fake_attributes)
    monkeypatch.setattr(pipeline_module, "CallbackHandler", lambda: object())

    pipeline = pipeline_module.RAGPipeline.__new__(pipeline_module.RAGPipeline)
    pipeline.config = {
        "dataset": "manufacturing_basic",
        "chunk_version": "c3",
        "default_retrieval_mode": "hybrid",
    }
    pipeline.retrieval_configs = {
        "hybrid": {"retrieval_version": "r2"},
    }
    pipeline.embedder = FakeEmbedder()
    pipeline.langfuse = FakeLangfuse(events)
    pipeline.generator = type(
        "FakeGenerator",
        (),
        {
            "prompt_version": "p1",
            "generate": lambda self, question, chunks, callbacks: {
                "status": "answered",
                "answer": "답변",
                "reason": "근거 확인",
                "citations": [],
                "latency_seconds": 0.3,
                "attempts": 1,
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "error": None,
            },
        },
    )()
    candidate = {
        "chunk_id": "MB01_c3_0002",
        "book_id": "MB01",
        "file_name": "MB01.pdf",
        "page_start": 2,
        "page_end": 2,
        "text": "근거 내용",
        "score": 0.9,
    }
    pipeline.retrieve = lambda question, retrieval_mode, chunk_version: {
        "retrieval_version": "r2",
        "chunk_version": "c3",
        "collection": "manufacturing_basic_c3_m1",
        "score_type": "rrf",
        "final_k": 3,
        "candidates": [candidate],
        "context": [candidate],
        "context_token_count": 100,
    }

    result = pipeline.query(
        "질문",
        request_id="request-1",
        source="service",
        user_id="demo_field",
        session_id="session-1",
    )

    assert attributes["user_id"] == "demo_field"
    assert attributes["session_id"] == "session-1"
    assert attributes["metadata"]["source"] == "service"
    assert attributes["metadata"]["user_id"] == "demo_field"
    assert attributes["metadata"]["session_id"] == "session-1"
    assert events[:2] == ["rag_query", "attributes"]
    root_update = pipeline.langfuse.observations[0].updates[0]
    assert root_update["output"]["status"] == "answered"
    assert (
        root_update["output"]["retrieval_results"][0]["chunk_id"]
        == candidate["chunk_id"]
    )
    assert root_update["usage_details"] == {"input": 100, "output": 20, "total": 120}
    assert result["trace_id"] == "trace-request-1"


def test_query_marks_root_trace_as_error_when_retrieval_raises(monkeypatch):
    @contextmanager
    def fake_attributes(**kwargs):
        yield

    monkeypatch.setattr(pipeline_module, "propagate_attributes", fake_attributes)
    pipeline = pipeline_module.RAGPipeline.__new__(pipeline_module.RAGPipeline)
    pipeline.config = {
        "dataset": "manufacturing_basic",
        "chunk_version": "c3",
        "default_retrieval_mode": "hybrid",
    }
    pipeline.retrieval_configs = {"hybrid": {"retrieval_version": "r2"}}
    pipeline.embedder = FakeEmbedder()
    pipeline.generator = SimpleGenerator()
    pipeline.langfuse = FakeLangfuse()

    def fail_retrieval(*args, **kwargs):
        raise RuntimeError("internal details")

    pipeline.retrieve = fail_retrieval

    with pytest.raises(RuntimeError, match="internal details"):
        pipeline.query(
            "질문",
            request_id="request-error",
            source="service",
            user_id="demo_field",
            session_id="session-1",
        )

    root_update = pipeline.langfuse.observations[0].updates[0]
    assert root_update["output"]["status"] == "error"
    assert root_update["metadata"]["status"] == "error"
    assert root_update["status_message"] == "pipeline_error:RuntimeError"
    assert "internal details" not in str(root_update)


class SimpleGenerator:
    prompt_version = "p1"
