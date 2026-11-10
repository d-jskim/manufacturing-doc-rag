"""검색 후보를 순위대로 문맥에 담는다. 다음 청크가 한도를 넘기면 거기서 멈춘다(건너뛰지 않음)."""


def fill_context(candidates: list[dict], counter, token_budget: int) -> tuple[list[dict], int]:
    context, used = [], 0
    for chunk in candidates:
        tokens = counter.count(chunk["text"])
        if used + tokens > token_budget:
            break
        context.append(chunk)
        used += tokens
    return context, used