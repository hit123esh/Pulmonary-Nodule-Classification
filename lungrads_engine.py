"""
Lung-RADS v2022 Decision Engine.

Deterministic rule-based classification for lung cancer screening LDCT cases.
Implements the Lung-RADS v2022 category assignment logic with size thresholds,
baseline/new/growing status, and 4X suspicious feature detection.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def classify(features: dict) -> dict:
    """
    Classify a case using the Lung-RADS v2022 guidelines.

    Args:
        features: dict from nlp_extractor.extract_features()

    Returns:
        dict with keys: risk_category, nodule_type, nodule_size,
                        nodule_count, category_guideline,
                        management_recommendation, reasoning
    """
    reasoning_parts = []
    size = features.get("primary_size")
    nodule_type = features.get("primary_type", "Solid")
    status = features.get("status", "Baseline")
    count = features.get("count_label", "Single")
    solid_comp = features.get("solid_component_size")

    # ------------------------------------------------------------------
    # Handle no-nodule case (Category 1)
    # ------------------------------------------------------------------
    nodule_count_raw = features.get("nodule_count", 0)
    if nodule_count_raw == 0 or size is None:
        if nodule_count_raw == 0:
            reasoning_parts.append(
                "No pulmonary nodules identified on this screening LDCT."
            )
            cat_num = "1"
            cat_name = "Negative"
            mgmt = "Continue annual screening with 12-month LDCT."
        else:
            reasoning_parts.append(
                "WARNING: Nodule size could not be extracted from report. "
                "Manual review required."
            )
            cat_num = "0"
            cat_name = "Incomplete"
            mgmt = "Additional imaging or comparison needed."

        return _build_result(
            nodule_type=nodule_type if nodule_count_raw > 0 else "N/A",
            nodule_size=size,
            nodule_count=count if nodule_count_raw > 0 else "None",
            category_guideline=f"Category {cat_num} – {cat_name}",
            management=mgmt,
            reasoning=" ".join(reasoning_parts),
        )

    reasoning_parts.append(
        f"Nodule type: {nodule_type}. Size: {size} mm. Status: {status}. Count: {count}."
    )

    # ------------------------------------------------------------------
    # Classify by nodule type
    # ------------------------------------------------------------------
    if nodule_type == "Solid":
        cat_num, cat_name = _classify_solid(size, status, reasoning_parts)
    elif nodule_type == "GGN":
        cat_num, cat_name = _classify_ggn(size, status, reasoning_parts)
    elif nodule_type == "Part-Solid":
        cat_num, cat_name = _classify_part_solid(
            size, status, solid_comp, reasoning_parts
        )
    else:
        cat_num, cat_name = _classify_solid(size, status, reasoning_parts)

    # ------------------------------------------------------------------
    # Check for 4X upgrade (suspicious features)
    # ------------------------------------------------------------------
    cat_num, cat_name = _check_4x_upgrade(
        features, cat_num, cat_name, reasoning_parts
    )

    # ------------------------------------------------------------------
    # Check S modifier
    # ------------------------------------------------------------------
    s_modifier = _check_s_modifier(features, reasoning_parts)

    # ------------------------------------------------------------------
    # Determine management
    # ------------------------------------------------------------------
    mgmt = _get_management(cat_num)
    s_label = "S" if s_modifier else ""
    category_str = f"Category {cat_num}{s_label} – {cat_name}"

    return _build_result(
        nodule_type=nodule_type,
        nodule_size=size,
        nodule_count=count,
        category_guideline=category_str,
        management=mgmt,
        reasoning=" ".join(reasoning_parts),
    )


# ======================================================================
# Solid Nodule Classification
# ======================================================================

def _classify_solid(size: float, status: str, reasoning: list) -> tuple[str, str]:
    """Classify solid nodule by size and status."""

    if status == "Growing":
        # Growing solid nodule
        if size < 8:
            reasoning.append(
                f"Growing solid nodule <8 mm → Category 4A (Suspicious)."
            )
            return "4A", "Suspicious"
        else:
            reasoning.append(
                f"Growing or new solid nodule ≥8 mm → Category 4B (Very Suspicious)."
            )
            return "4B", "Very Suspicious"

    if status == "New":
        if size < 4:
            reasoning.append(f"New solid nodule <4 mm → Category 2 (Benign).")
            return "2", "Benign Appearance"
        elif size < 6:
            reasoning.append(
                f"New solid nodule 4 to <6 mm → Category 3 (Probably Benign)."
            )
            return "3", "Probably Benign"
        elif size < 8:
            reasoning.append(
                f"New solid nodule 6 to <8 mm → Category 4A (Suspicious)."
            )
            return "4A", "Suspicious"
        else:
            reasoning.append(
                f"New solid nodule ≥8 mm → Category 4B (Very Suspicious)."
            )
            return "4B", "Very Suspicious"

    # Baseline or Stable
    if size < 6:
        reasoning.append(
            f"Baseline solid nodule <6 mm → Category 2 (Benign Appearance)."
        )
        return "2", "Benign Appearance"
    elif size < 8:
        reasoning.append(
            f"Baseline solid nodule 6 to <8 mm → Category 3 (Probably Benign)."
        )
        return "3", "Probably Benign"
    elif size < 15:
        reasoning.append(
            f"Baseline solid nodule 8 to <15 mm → Category 4A (Suspicious)."
        )
        return "4A", "Suspicious"
    else:
        reasoning.append(
            f"Baseline solid nodule ≥15 mm → Category 4B (Very Suspicious)."
        )
        return "4B", "Very Suspicious"


# ======================================================================
# Ground-Glass Nodule (GGN) Classification
# ======================================================================

def _classify_ggn(size: float, status: str, reasoning: list) -> tuple[str, str]:
    """Classify non-solid (GGN) nodule."""

    if size < 30:
        reasoning.append(
            f"Non-solid (GGN) nodule <30 mm → Category 2 (Benign Appearance)."
        )
        return "2", "Benign Appearance"
    else:
        # ≥30 mm at baseline or new
        if status in ("Stable",):
            reasoning.append(
                f"Non-solid (GGN) nodule ≥30 mm, stable or slowly growing → Category 2."
            )
            return "2", "Benign Appearance"
        else:
            reasoning.append(
                f"Non-solid (GGN) nodule ≥30 mm at baseline or new → Category 3 (Probably Benign)."
            )
            return "3", "Probably Benign"


# ======================================================================
# Part-Solid Nodule Classification
# ======================================================================

def _classify_part_solid(
    size: float, status: str, solid_comp: float | None,
    reasoning: list
) -> tuple[str, str]:
    """Classify part-solid nodule by solid component size."""

    # Use total size if solid component unknown
    sc = solid_comp if solid_comp is not None else None

    if size < 6 and status == "Baseline":
        reasoning.append(
            f"Baseline part-solid nodule <6 mm total → Category 2 (Benign Appearance)."
        )
        return "2", "Benign Appearance"

    if status == "New" and size < 6:
        reasoning.append(
            f"New part-solid nodule <6 mm total → Category 3 (Probably Benign)."
        )
        return "3", "Probably Benign"

    # ≥6 mm total — classify by solid component
    if sc is not None:
        if status in ("New", "Growing") and sc >= 4:
            reasoning.append(
                f"New or growing solid component ≥4 mm → Category 4B (Very Suspicious)."
            )
            return "4B", "Very Suspicious"
        if sc < 6:
            reasoning.append(
                f"Part-solid nodule ≥6 mm total with solid component <6 mm → Category 3 (Probably Benign)."
            )
            return "3", "Probably Benign"
        elif sc < 8:
            reasoning.append(
                f"Part-solid nodule with solid component 6 to <8 mm → Category 4A (Suspicious)."
            )
            return "4A", "Suspicious"
        else:
            reasoning.append(
                f"Part-solid nodule with solid component ≥8 mm → Category 4B (Very Suspicious)."
            )
            return "4B", "Very Suspicious"
    else:
        # No solid component size available — use total size as approximation
        reasoning.append(
            "Solid component size not extractable; classifying by total nodule size."
        )
        if size < 6:
            return "2", "Benign Appearance"
        elif size < 8:
            return "3", "Probably Benign"
        elif size < 15:
            return "4A", "Suspicious"
        else:
            return "4B", "Very Suspicious"


# ======================================================================
# 4X Upgrade Check
# ======================================================================

def _check_4x_upgrade(
    features: dict, cat_num: str, cat_name: str, reasoning: list
) -> tuple[str, str]:
    """Check if case should be upgraded to 4X based on suspicious features."""
    # 4X applies to Category 3 or 4 nodules with additional suspicious features
    if cat_num not in ("3", "4A", "4B"):
        return cat_num, cat_name

    suspicious = []
    if features.get("spiculated"):
        suspicious.append("spiculated margins")
    if features.get("lymphadenopathy"):
        suspicious.append("lymphadenopathy")

    if suspicious:
        reasoning.append(
            f"UPGRADE to Category 4X due to additional suspicious features: "
            f"{', '.join(suspicious)}."
        )
        return "4X", "Very Suspicious (Additional Features)"

    return cat_num, cat_name


# ======================================================================
# S Modifier
# ======================================================================

def _check_s_modifier(features: dict, reasoning: list) -> bool:
    """Check for significant non-lung findings (S modifier)."""
    # For simplicity, flag emphysema as a potentially significant finding
    # In full implementation, would check coronary calcification, etc.
    if features.get("emphysema"):
        reasoning.append("Note: Emphysema identified — S modifier may apply.")
        return True
    return False


# ======================================================================
# Management Lookup
# ======================================================================

def _get_management(cat_num: str) -> str:
    """Return management recommendation for a Lung-RADS category."""
    mgmt_table = {
        "0": "Comparison to prior CT or additional imaging needed.",
        "1": "Continue annual screening with 12-month LDCT.",
        "2": "Continue annual screening with 12-month LDCT.",
        "3": "6-month LDCT recommended.",
        "4A": "3-month LDCT recommended; PET/CT may be considered if ≥8 mm.",
        "4B": (
            "Diagnostic chest CT with or without contrast, PET/CT, "
            "tissue sampling, and/or referral for clinical evaluation."
        ),
        "4X": (
            "Diagnostic chest CT with or without contrast, PET/CT, "
            "tissue sampling, and/or referral for clinical evaluation."
        ),
    }
    return mgmt_table.get(cat_num, "Clinical judgment required.")


# ======================================================================
# Helpers
# ======================================================================

def _build_result(nodule_type, nodule_size, nodule_count,
                  category_guideline, management, reasoning):
    """Build standardized result dict for Lung-RADS cases."""
    return {
        "risk_category": "",  # Lung-RADS doesn't use risk category like Fleischner
        "nodule_type": nodule_type,
        "nodule_size": str(nodule_size) if nodule_size is not None else "",
        "nodule_count": nodule_count,
        "category_guideline": category_guideline,
        "management_recommendation": management,
        "reasoning": reasoning,
    }
