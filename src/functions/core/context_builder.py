# functions/core/context_builder.py
from __future__ import annotations

"""
User context construction utilities.

This module builds a **deterministic user context** from a single row in
students.csv. The output is used by HyDE query generation (LLM-side)
and MUST remain:

- Stable across runs
- Compact (prompt-efficient)
- Free of business logic or ranking assumptions

Design principles
-----------------
- Deterministic string formatting (no random order)
- Minimal normalization (avoid over-cleaning user intent)
- Language-aware but schema-stable
- No LLM usage
"""

from dataclasses import dataclass
from typing import Any, Dict

verbose = 0
# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class UserContextArtifacts:
    """
    Container for user context artifacts.

    Attributes
    ----------
    user_context_json:
        Structured representation for programmatic use and auditing.
    user_context_text:
        Compact, human-readable summary used as HyDE prompt input.
    """

    user_context_json: Dict[str, Any]
    user_context_text: str


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------
def build_user_context(student_row: Dict[str, Any]) -> UserContextArtifacts:
    print(f"Position : context_builder.py/context_builder.py") if verbose else None
    """
    Build deterministic user context artifacts from a students.csv row.

    Parameters
    ----------
    student_row:
        Dict-like row from students.csv (already parsed).

    Returns
    -------
    UserContextArtifacts
        JSON + text representations of the user context.

    Notes
    -----
    - This function MUST stay deterministic.
    - Do not add inferred fields or heuristics here.
    - Any enrichment belongs in later pipeline stages.
    """
    print(f"student_row: {student_row}") if verbose else None

    # ------------------------------------------------------------------
    # Required / basic fields
    # ------------------------------------------------------------------
    student_id = student_row["student_id"]

    preferred_language = (student_row.get("preferred_language") or "th").strip()
    preferred_language = preferred_language if preferred_language in ("th", "en") else "th"

    current_status = student_row.get("current_status") or "unknown"
    edu_level = student_row.get("student_year") or "unknown"
    edu_major = student_row.get("faculty_name")+" "+student_row.get("curriculum_name")+" "+student_row.get("university_name") or ""

    # ------------------------------------------------------------------
    # Optional profile fields
    # ------------------------------------------------------------------
    target_roles_raw = (student_row.get("target_roles") or "").strip()
    skills_raw = (student_row.get("skill") or "").strip()
    interests_raw = (student_row.get("interests") or "").strip()

    onboard_grp = student_row.get("onboard_grp") or "NA"
    onboard_desc = student_row.get("onboard_grp_description") or ""

    # ------------------------------------------------------------------
    # Structured JSON (stable, sanitized)
    # ------------------------------------------------------------------
    user_context_json: Dict[str, Any] = {
        "student_id": student_id,
        "preferred_language": preferred_language,
        "current_status": current_status,
        "education": {
            "level": edu_level,
            "major": edu_major,
        },
        "target_roles": [
            {
                "role_id": r.strip().lower().replace(" ", "_"),
                "role_name": r.strip(),
                "priority": i + 1,
            }
            for i, r in enumerate([x for x in target_roles_raw.split("|") if x.strip()])
        ],
        "skills": [
            {
                "skill_id": s.split(":")[0].strip().lower().replace(" ", "_"),
                "skill_name": s.split(":")[0].strip(),
                "proficiency": (s.split(":")[1].strip() if ":" in s else "unknown"),
            }
            for s in [x for x in skills_raw.split(";") if x.strip()]
        ],
        "interests": [x.strip() for x in interests_raw.split(";") if x.strip()],
        "onboard_grp": onboard_grp,
        "onboard_grp_description": onboard_desc,
    }

    # ------------------------------------------------------------------
    # Text summary (compact, HyDE-friendly)
    # ------------------------------------------------------------------
    roles_text = ", ".join([x.strip() for x in target_roles_raw.split("|") if x.strip()]) or "ไม่ระบุ"
    skills_text = skills_raw or "ไม่ระบุ"
    interests_text = interests_raw.replace(";", ", ") if interests_raw else "ไม่ระบุ"

    if preferred_language == "th":
        user_context_text = (
            f"นักศึกษา: {edu_major}\n"
            f"เป้าหมายอาชีพ: {roles_text}\n"
            f"ทักษะ: {skills_text}\n"
            # f"ความสนใจ: {interests_text}\n"
            f"กลุ่มผู้ใช้: {onboard_grp} ({onboard_desc})\n"
            f"ภาษา: ไทย"
        )
    else:
        user_context_text = (
            f"Student major: {edu_major}.\n"
            f"Target roles: {roles_text}.\n"
            f"Skills: {skills_text}.\n"
            # f"Interests: {interests_text}.\n"
            f"Onboard group: {onboard_grp} ({onboard_desc}).\n"
            f"Language: English."
        )

    return UserContextArtifacts(
        user_context_json=user_context_json,
        user_context_text=user_context_text,
    )


__all__ = [
    "UserContextArtifacts",
    "build_user_context",
]
