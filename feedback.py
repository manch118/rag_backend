from models import FeedbackEntry, StatsResponse


_store: list[FeedbackEntry] = []


def post_feedback(entry: FeedbackEntry) -> dict:
    _store.append(entry)
    return {"status": "ok", "message": "Feedback recorded"}


def get_stats() -> StatsResponse:
    if not _store:
        return StatsResponse(
            total_feedback=0,
            avg_relevance=0.0,
            avg_completeness=0.0,
            avg_specificity=0.0,
            avg_fluency=0.0,
            avg_overall=0.0,
            recent_entries=[]
        )

    n = len(_store)
    avg_rel = sum(e.relevance_score for e in _store) / n
    avg_comp = sum(e.completeness_score for e in _store) / n
    avg_spec = sum(e.specificity_score for e in _store) / n
    avg_flu = sum(e.fluency_score for e in _store) / n
    avg_overall = (avg_rel + avg_comp + avg_spec + avg_flu) / 4.0

    recent = _store[-5:] if len(_store) > 5 else _store[:]

    return StatsResponse(
        total_feedback=n,
        avg_relevance=round(avg_rel, 2),
        avg_completeness=round(avg_comp, 2),
        avg_specificity=round(avg_spec, 2),
        avg_fluency=round(avg_flu, 2),
        avg_overall=round(avg_overall, 2),
        recent_entries=recent
    )