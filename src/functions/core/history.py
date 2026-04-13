# functions/core/history.py
from __future__ import annotations

"""
functions.core.history

History summarization (deterministic, no LLM).

This module builds a short HistorySummaryText used by HyDE prompts and
also provides a utility to extract "seen" feed_ids for exclude-seen policy.

Key uses
--------
1) Batch / Refresh pipelines (HyDE prompt context):
   - Summarize user interaction history into a short, bounded text block.

2) Online retrieval pipeline (exclude-seen):
   - Extract recently seen feed_ids deterministically to filter candidates.

Design goals
------------
- Deterministic output:
  - stable ordering, no randomness
  - stable truncation rules
- Bounded size:
  - explicit max K, explicit max_chars truncation
- Best-effort robustness:
  - safe behavior when timestamps or metadata are missing
  - do not crash on common input issues (but do stay deterministic)

Important production note
-------------------------
In production, theme inference and "recent feeds text" should be driven by
feed metadata (tags/categories) from the content store (and/or logs), not by
feed_id prefix heuristics. This heuristic is POC-only.

Data expectations
-----------------
For summarization, the input user_events is expected to have:
- feed_id (required)
- event_type (recommended)
- dwell_ms (optional)
- ts (optional, but required for deterministic recency blocks and windowing)
"""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd


# -----------------------------------------------------------------------------
# Event scoring (deterministic)
# -----------------------------------------------------------------------------
# Small deterministic weights to separate event types.
# NOTE: These are heuristics; production should calibrate via offline analysis.
EVENT_WEIGHT: Dict[str, float] = {
    "view": 1.0,
    "click": 1.2,
    "like": 1.5,
    "share": 1.6,
    "comment": 1.6,
}


def _dwell_boost(dwell_ms: int) -> float:
    """
    Deterministic dwell-time boost.

    We intentionally keep this coarse so it:
    - remains stable across minor logging differences
    - does not dominate event_type weighting

    Args:
        dwell_ms: dwell time in milliseconds (>=0)
    """
    if dwell_ms >= 60000:
        return 0.6
    if dwell_ms >= 30000:
        return 0.4
    if dwell_ms >= 10000:
        return 0.2
    return 0.0


def _safe_int(x: Any, default: int = 0) -> int:
    """Convert to int safely (no exceptions)."""
    try:
        return int(x)
    except Exception:
        return default


def _infer_theme_from_feed_id(feed_id: str) -> str:
    """
    POC heuristic: infer a high-level theme from feed_id prefix.

    Replace later with:
    - feed metadata tags (preferred), or
    - a join to content taxonomy table, or
    - a learned theme classifier.

    Returns:
        Theme string key (stable vocabulary).
    """
    fid = (feed_id or "").upper()
    if fid.startswith("TH_F") or fid.startswith("EN_F"):
        return "data_career"
    if fid.startswith("TH_BIO"):
        return "biotech"
    if fid.startswith("TH_SCH"):
        return "scholarship"
    if fid.startswith("TH_UNI"):
        return "university"
    return "other"


def _clean_text(s: Any) -> str:
    """
    Normalize whitespace deterministically.

    - Replace newlines with spaces
    - Collapse repeated spaces
    - Strip leading/trailing whitespace
    """
    if s is None:
        return ""
    t = str(s).replace("\n", " ").replace("\r", " ").strip()
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def _truncate(s: str, max_chars: int) -> str:
    """
    Hard truncate to a stable max char budget.

    Uses an ellipsis (…).
    """
    if max_chars <= 0:
        return ""
    s = s or ""
    if len(s) <= max_chars:
        return s
    # Keep last char as ellipsis, preserving deterministic truncation boundary.
    return s[: max_chars - 1].rstrip() + "…"


def _build_recent_feeds_block(
    df_events: pd.DataFrame,
    preferred_language: str,
    feeds_lookup: Dict[str, Dict[str, Any]],
    recent_k: int,
    feed_text_max_chars: int,
) -> str:
    """
    Build a bounded, deterministic "recent feeds" context block for HyDE.

    Rules
    -----
    - Requires "ts" column to define deterministic recency ordering.
    - Sort by ts desc (most recent first).
    - Select last K UNIQUE feed_ids that exist in feeds_lookup.
    - For each feed, include title + (summary or content or title) truncated.

    Args:
        df_events: events dataframe (must have feed_id and ts)
        preferred_language: "th" or "en" (controls header language)
        feeds_lookup: mapping feed_id -> feed metadata (title/summary/content)
        recent_k: maximum unique feeds to include
        feed_text_max_chars: per-feed snippet max length
    """
    if recent_k <= 0:
        return ""

    if "ts" not in df_events.columns:
        # Cannot determine recency deterministically without timestamps.
        return ""

    ts = pd.to_datetime(df_events["ts"], errors="coerce", utc=True)
    df2 = df_events.copy()
    df2["_ts"] = ts
    df2 = df2.dropna(subset=["_ts"])
    if df2.empty:
        return ""

    # Deterministic ordering: recency desc
    df2 = df2.sort_values("_ts", ascending=False)

    seen: Set[str] = set()
    chosen: List[str] = []
    for fid in df2["feed_id"].astype(str).tolist():
        fid = fid.strip()
        if not fid or fid in seen:
            continue
        # Only include if we can actually show something from metadata.
        if fid in feeds_lookup:
            chosen.append(fid)
            seen.add(fid)
        if len(chosen) >= recent_k:
            break

    if not chosen:
        return ""

    header = "Recent interacted feeds (truncated):" if preferred_language == "en" else "ฟีดล่าสุดที่มีปฏิสัมพันธ์ (ตัดสั้น):"
    lines: List[str] = [header]

    for i, fid in enumerate(chosen, start=1):
        feed = feeds_lookup.get(fid, {}) or {}
        title = _clean_text(feed.get("title", ""))
        summary = _clean_text(feed.get("summary", ""))
        content = _clean_text(feed.get("content", ""))

        body = summary or content or title
        body = _truncate(body, feed_text_max_chars)

        # Stable, parse-friendly format.
        lines.append(f"- ({i}) {fid} | {title} | {body}")

    return "\n".join(lines).strip()


# -----------------------------------------------------------------------------
# Public API: summarization
# -----------------------------------------------------------------------------
def build_history_summary(
    user_events: pd.DataFrame,
    preferred_language: str = "th",
    window_days: int = 30,
    top_themes: int = 4,
    include_recent_feeds: bool = True,
    recent_k: int = 5,
    feeds_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    feed_text_max_chars: int = 240,
) -> str:
    """
    Build a stable short text summary from interaction events.

    Input
    -----
    user_events dataframe with (typical) columns:
      - feed_id (required)
      - event_type (optional; defaults to "view")
      - dwell_ms (optional; defaults to 0)
      - ts (optional; only needed for recency block and windowing)

    Output
    ------
    HistorySummaryText in TH/EN.

    Notes
    -----
    - This does not currently filter by window_days using ts. The `window_days` is
      a descriptive label used in the summary header.
      If you want true windowing here, do it upstream by filtering user_events
      before calling this function (keeps this module deterministic/simple).
    - include_recent_feeds requires feeds_lookup and ts.
    """
    if user_events is None or len(user_events) == 0:
        return ""

    df = user_events.copy()

    # Normalize required columns defensively.
    if "event_type" not in df.columns:
        df["event_type"] = "view"
    if "dwell_ms" not in df.columns:
        df["dwell_ms"] = 0
    if "feed_id" not in df.columns:
        return ""

    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    df["feed_id"] = df["feed_id"].astype(str).str.strip()

    # Score each event deterministically.
    scores: List[float] = []
    for _, r in df.iterrows():
        et = str(r.get("event_type", "view") or "view")
        base = EVENT_WEIGHT.get(et, 1.0)
        dwell = _safe_int(r.get("dwell_ms", 0), 0)
        scores.append(base + _dwell_boost(dwell))
    df["score"] = scores

    # POC theme inference.
    df["theme"] = df["feed_id"].apply(_infer_theme_from_feed_id)

    # Aggregate by theme; all deterministic aggregations.
    theme_agg = (
        df.groupby("theme", dropna=False)
        .agg(
            total_events=("theme", "count"),
            total_score=("score", "sum"),
            num_views=("event_type", lambda s: int((s == "view").sum())),
            num_clicks=("event_type", lambda s: int((s == "click").sum())),
            num_likes=("event_type", lambda s: int((s == "like").sum())),
            max_dwell=("dwell_ms", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max())),
        )
        .reset_index()
    )

    # Sort by score desc then events desc to keep ordering stable.
    theme_agg = theme_agg.sort_values(["total_score", "total_events", "theme"], ascending=[False, False, True])
    top = theme_agg.head(int(max(0, top_themes)))
    total_events = int(len(df))

    # Identify "weak engagement" themes (heuristic, deterministic).
    weak_themes: List[str] = []
    for _, r in theme_agg.iterrows():
        if int(r["total_events"]) >= 2 and float(r["total_score"]) <= 2.1 and int(r["max_dwell"]) < 8000:
            weak_themes.append(str(r["theme"]))
    weak_themes = weak_themes[:2]

    preferred_language = (preferred_language or "th").strip().lower()
    preferred_language = "en" if preferred_language == "en" else "th"

    if preferred_language == "en":
        lines: List[str] = [f"Summary of last {window_days} days (total {total_events} events):"]
        if not top.empty:
            lines.append("Highest engagement themes:")
            for _, r in top.iterrows():
                lines.append(
                    f"- {r['theme']}: score={float(r['total_score']):.1f}, events={int(r['total_events'])}, "
                    f"likes={int(r['num_likes'])}, clicks={int(r['num_clicks'])}, max_dwell_ms={int(r['max_dwell'])}"
                )
        if weak_themes:
            lines.append(f"Lower engagement themes: {', '.join(weak_themes)}")

        if include_recent_feeds and feeds_lookup:
            block = _build_recent_feeds_block(
                df_events=df,
                preferred_language=preferred_language,
                feeds_lookup=feeds_lookup,
                recent_k=int(max(0, recent_k)),
                feed_text_max_chars=int(max(0, feed_text_max_chars)),
            )
            if block:
                lines.append(block)

        return "\n".join(lines).strip()

    # TH default
    lines = [f"สรุปพฤติกรรม {window_days} วันล่าสุด (รวม {total_events} เหตุการณ์):"]
    if not top.empty:
        lines.append("ธีมที่มีการมีส่วนร่วมสูง:")
        for _, r in top.iterrows():
            lines.append(
                f"- {r['theme']}: คะแนน={float(r['total_score']):.1f}, เหตุการณ์={int(r['total_events'])}, "
                f"ไลก์={int(r['num_likes'])}, คลิก={int(r['num_clicks'])}, dwell สูงสุด={int(r['max_dwell'])}ms"
            )
    if weak_themes:
        lines.append(f"ธีมที่มีการมีส่วนร่วมน้อย: {', '.join(weak_themes)}")

    if include_recent_feeds and feeds_lookup:
        block = _build_recent_feeds_block(
            df_events=df,
            preferred_language=preferred_language,
            feeds_lookup=feeds_lookup,
            recent_k=int(max(0, recent_k)),
            feed_text_max_chars=int(max(0, feed_text_max_chars)),
        )
        if block:
            lines.append(block)

    return "\n".join(lines).strip()


# -----------------------------------------------------------------------------
# Public API: seen-feed extraction for exclude-seen policy
# -----------------------------------------------------------------------------
def extract_seen_feed_ids(
    user_events: pd.DataFrame,
    *,
    event_types: Optional[Iterable[str]] = None,
    now_utc: Optional[datetime] = None,
    window_days: Optional[int] = None,
    max_unique: Optional[int] = None,
) -> Set[str]:
    """
    Extract unique feed_ids the user already interacted with.

    This is used in pipeline_3_online_retrieval for exclude-seen filtering.

    Args:
        user_events:
            User interactions dataframe with at least a `feed_id` column.
        event_types:
            Optional allow-list of event types (e.g., ["view","click"]).
            If provided and `event_type` column exists, only these are counted.
        now_utc:
            Reference time (UTC). Defaults to datetime.now(timezone.utc).
        window_days:
            If provided and `ts` exists, only include events in the last N days.
        max_unique:
            If provided and `ts` exists, cap the number of unique feeds using
            deterministic recency ordering (most recent first).

    Returns:
        Set[str] of feed_id values.
    """
    if user_events is None or len(user_events) == 0:
        return set()
    if "feed_id" not in user_events.columns:
        return set()

    df = user_events.copy()
    df["feed_id"] = df["feed_id"].astype(str).str.strip()
    df = df[df["feed_id"].astype(bool)]

    # Filter event types if possible.
    if event_types is not None and "event_type" in df.columns:
        allow = {str(x).lower().strip() for x in event_types if str(x).strip()}
        df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
        df = df[df["event_type"].isin(allow)]

    ts_col = "ts" if "ts" in df.columns else None

    # Optional windowing by time (only if ts exists).
    if window_days is not None and int(window_days) > 0 and ts_col is not None:
        now_utc = now_utc or datetime.now(timezone.utc)
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.assign(_ts=ts).dropna(subset=["_ts"])
        if not df.empty:
            cutoff = now_utc - pd.Timedelta(days=int(window_days))
            df = df[df["_ts"] >= cutoff]

    # Optional deterministic cap by most-recent unique feeds.
    if max_unique is not None and int(max_unique) > 0 and ts_col is not None:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts", ascending=False)

        seen: Set[str] = set()
        for fid in df["feed_id"].tolist():
            if fid in seen:
                continue
            seen.add(fid)
            if len(seen) >= int(max_unique):
                break
        return seen

    return set(df["feed_id"].tolist())


__all__ = [
    "EVENT_WEIGHT",
    "build_history_summary",
    "extract_seen_feed_ids",
]
