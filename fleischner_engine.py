"""
Fleischner Society 2017 Decision Engine.

Deterministic rule-based classification for incidental pulmonary nodules.
Implements the full Fleischner 2017 guideline logic with risk stratification,
size thresholds, and subsolid nodule handling.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def classify(features: dict) -> dict:
    """
    Classify a case using the Fleischner 2017 guidelines.

    Args:
        features: dict from nlp_extractor.extract_features()

    Returns:
        dict with keys: risk_category, nodule_type, nodule_size,
                        nodule_count, category_guideline,
                        management_recommendation, reasoning
    """
    reasoning_parts = []

    # ------------------------------------------------------------------
    # Step 1: Check exclusion criteria
    # ------------------------------------------------------------------
    exclusions = _check_exclusions(features, reasoning_parts)
    if exclusions:
        return _build_result(
            risk_category="N/A",
            nodule_type=features["primary_type"],
            nodule_size=features["primary_size"],
            nodule_count=features["count_label"],
            category_guideline="Fleischner Exclusion Applies",
            management="Clinical judgment required — " + exclusions,
            reasoning=" ".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    # Step 2: Risk stratification
    # ------------------------------------------------------------------
    risk = _stratify_risk(features, reasoning_parts)

    # ------------------------------------------------------------------
    # Step 3: Perifissural nodule check
    # ------------------------------------------------------------------
    if features.get("perifissural"):
        reasoning_parts.append(
            "Nodule is perifissural with smooth/ovoid morphology, "
            "consistent with intrapulmonary lymph node."
        )
        return _build_result(
            risk_category=risk,
            nodule_type="Solid",
            nodule_size=features["primary_size"],
            nodule_count=features["count_label"],
            category_guideline="Perifissural Nodule (Likely Lymph Node)",
            management="No follow-up required.",
            reasoning=" ".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    # Step 4: Route by nodule type
    # ------------------------------------------------------------------
    nodule_type = features["primary_type"]
    size = features["primary_size"]
    count = features["count_label"]

    if size is None:
        reasoning_parts.append(
            "WARNING: Nodule size could not be extracted. "
            "Cannot apply size-based guideline."
        )
        return _build_result(
            risk_category=risk,
            nodule_type=nodule_type,
            nodule_size=None,
            nodule_count=count,
            category_guideline="Size Not Determined",
            management="Cannot classify — size not extracted. Manual review required.",
            reasoning=" ".join(reasoning_parts),
        )

    reasoning_parts.append(
        f"Nodule type: {nodule_type}. Size: {size} mm. Count: {count}."
    )

    if nodule_type == "Solid":
        return _classify_solid(features, risk, size, count, reasoning_parts)
    elif nodule_type == "GGN":
        return _classify_ggn(features, risk, size, count, reasoning_parts)
    elif nodule_type == "Part-Solid":
        return _classify_part_solid(features, risk, size, count, reasoning_parts)
    else:
        return _classify_solid(features, risk, size, count, reasoning_parts)


# ======================================================================
# Exclusion Criteria
# ======================================================================

def _check_exclusions(features: dict, reasoning: list) -> str | None:
    """Check Fleischner exclusion criteria. Returns reason string if excluded."""
    issues = []
    age = features.get("age")
    if age is not None and age < 35:
        issues.append(f"Patient age ({age}) is below 35")
    if features.get("known_cancer"):
        issues.append("Patient has known primary cancer")
    if features.get("immunocompromised"):
        issues.append("Patient is immunocompromised")
    # Note: is_screening is already handled by the router (Type column)
    # but double-check anyway
    if features.get("is_screening"):
        issues.append("CT is part of lung cancer screening")

    if issues:
        reasoning.append(
            "Fleischner exclusion criteria met: " + "; ".join(issues) + "."
        )
        return "; ".join(issues)
    reasoning.append("No Fleischner exclusion criteria identified.")
    return None


# ======================================================================
# Risk Stratification
# ======================================================================

def _stratify_risk(features: dict, reasoning: list) -> str:
    """Classify patient as Low Risk or High Risk."""
    risk_factors = []

    if features.get("smoking"):
        risk_factors.append("smoking history")
        py = features.get("pack_years")
        if py:
            risk_factors[-1] += f" ({int(py)} pack-years)"

    age = features.get("age")
    if age and age > 55:
        risk_factors.append(f"age {age} (>55)")

    if features.get("upper_lobe"):
        risk_factors.append("upper lobe location")

    if features.get("spiculated"):
        risk_factors.append("spiculated margins")

    if features.get("emphysema"):
        risk_factors.append("emphysema")

    if features.get("occupational_exposure"):
        risk_factors.append("occupational exposure")

    if features.get("family_history"):
        risk_factors.append("family history of lung cancer")

    if risk_factors:
        risk = "High Risk"
        reasoning.append(
            f"Risk stratification: HIGH RISK based on: {', '.join(risk_factors)}."
        )
    else:
        risk = "Low Risk"
        reasoning.append(
            "Risk stratification: LOW RISK — no significant risk factors identified."
        )

    return risk


# ======================================================================
# Solid Nodule Classification
# ======================================================================

def _classify_solid(features, risk, size, count, reasoning):
    """Apply Fleischner solid nodule recommendations."""

    if count == "Single":
        if size < 6:
            if risk == "Low Risk":
                cat = "Single Solid Nodule <6 mm, Low Risk"
                mgmt = "No routine follow-up required."
                reasoning.append(
                    "Solid nodule <6 mm in a low-risk patient has <1% malignancy risk. "
                    "No follow-up is recommended per Fleischner 2017."
                )
            else:
                cat = "Single Solid Nodule <6 mm, High Risk"
                mgmt = "Optional CT at 12 months."
                reasoning.append(
                    "Solid nodule <6 mm in a high-risk patient. Malignancy risk is <1%, "
                    "but optional 12-month CT may be considered given risk factors."
                )
        elif size <= 8:
            if risk == "Low Risk":
                cat = "Single Solid Nodule 6-8 mm, Low Risk"
                mgmt = "CT at 6-12 months. If stable, no further follow-up generally required."
                reasoning.append(
                    "Solid nodule 6-8 mm in a low-risk patient. CT at 6-12 months recommended "
                    "to assess stability per Fleischner 2017."
                )
            else:
                cat = "Single Solid Nodule 6-8 mm, High Risk"
                mgmt = "CT at 6-12 months, then CT at 18-24 months to confirm stability."
                reasoning.append(
                    "Solid nodule 6-8 mm in a high-risk patient. Initial CT at 6-12 months "
                    "followed by a second CT at 18-24 months to confirm stability per Fleischner 2017."
                )
        else:  # > 8 mm
            if risk == "Low Risk":
                cat = "Single Solid Nodule >8 mm, Low Risk"
                mgmt = "CT at 3 months, PET/CT, and/or tissue biopsy."
                reasoning.append(
                    "Solid nodule >8 mm has ~10-20% malignancy risk. Prompt evaluation with "
                    "CT at 3 months, PET/CT, or biopsy is recommended per Fleischner 2017."
                )
            else:
                cat = "Single Solid Nodule >8 mm, High Risk"
                mgmt = (
                    "CT at 3 months, PET/CT, and/or tissue biopsy. "
                    "Lower threshold for PET/CT or biopsy given high-risk status."
                )
                reasoning.append(
                    "Solid nodule >8 mm in a high-risk patient has significant malignancy risk. "
                    "Prompt evaluation with CT at 3 months, PET/CT, or biopsy with a lower intervention "
                    "threshold per Fleischner 2017."
                )
    else:  # Multiple
        if size < 6:
            if risk == "Low Risk":
                cat = "Multiple Solid Nodules <6 mm, Low Risk"
                mgmt = "No routine follow-up required."
                reasoning.append(
                    "Multiple solid nodules all <6 mm in a low-risk patient. "
                    "No follow-up needed per Fleischner 2017."
                )
            else:
                cat = "Multiple Solid Nodules <6 mm, High Risk"
                mgmt = "Optional CT at 12 months."
                reasoning.append(
                    "Multiple solid nodules <6 mm in a high-risk patient. "
                    "Optional CT at 12 months per Fleischner 2017."
                )
        elif size <= 8:
            if risk == "Low Risk":
                cat = "Multiple Solid Nodules 6-8 mm, Low Risk"
                mgmt = "CT at 3-6 months, then optional CT at 18-24 months."
                reasoning.append(
                    "Multiple solid nodules (largest 6-8 mm) in a low-risk patient. "
                    "CT at 3-6 months based on most suspicious nodule, then optional "
                    "follow-up at 18-24 months per Fleischner 2017."
                )
            else:
                cat = "Multiple Solid Nodules 6-8 mm, High Risk"
                mgmt = "CT at 3-6 months, then CT at 18-24 months."
                reasoning.append(
                    "Multiple solid nodules (largest 6-8 mm) in a high-risk patient. "
                    "CT at 3-6 months based on most suspicious nodule, then CT at "
                    "18-24 months per Fleischner 2017."
                )
        else:  # > 8 mm
            if risk == "Low Risk":
                cat = "Multiple Solid Nodules >8 mm, Low Risk"
                mgmt = "CT at 3-6 months, then CT at 18-24 months if needed."
                reasoning.append(
                    "Multiple solid nodules (largest >8 mm) in a low-risk patient. "
                    "CT at 3-6 months to assess growth, then 18-24 months if inconclusive "
                    "per Fleischner 2017."
                )
            else:
                cat = "Multiple Solid Nodules >8 mm, High Risk"
                mgmt = "CT at 3-6 months, then CT at 18-24 months."
                reasoning.append(
                    "Multiple solid nodules (largest >8 mm) in a high-risk patient. "
                    "CT at 3-6 months, then CT at 18-24 months per Fleischner 2017."
                )

    return _build_result(
        risk_category=risk,
        nodule_type="Solid",
        nodule_size=size,
        nodule_count=count,
        category_guideline=cat,
        management=mgmt,
        reasoning=" ".join(reasoning),
    )


# ======================================================================
# Pure Ground-Glass Nodule (GGN) Classification
# ======================================================================

def _classify_ggn(features, risk, size, count, reasoning):
    """Apply Fleischner GGN recommendations."""

    if count == "Single":
        if size < 6:
            cat = "Single GGN <6 mm"
            mgmt = "No follow-up required."
            reasoning.append(
                "Pure ground-glass nodule <6 mm. No follow-up required per Fleischner 2017."
            )
        else:
            cat = "Single GGN ≥6 mm"
            mgmt = (
                "CT at 6-12 months to confirm persistence. "
                "If persistent, CT every 2 years for up to 5 years."
            )
            reasoning.append(
                f"Pure ground-glass nodule {size} mm (≥6 mm). CT at 6-12 months to confirm "
                "persistence, then surveillance CT every 2 years for 5 years. "
                "If growth or solid component develops, consider resection per Fleischner 2017."
            )
    else:  # Multiple
        if size < 6:
            cat = "Multiple Subsolid Nodules <6 mm"
            if risk == "High Risk":
                mgmt = "CT at 3-6 months. If persistent, consider CT at 2 and 4 years."
                reasoning.append(
                    "Multiple subsolid nodules all <6 mm in a high-risk patient. "
                    "CT at 3-6 months; if persistent, consider CT at 2 and 4 years per Fleischner 2017."
                )
            else:
                mgmt = "CT at 3-6 months. If persistent, no further routine follow-up."
                reasoning.append(
                    "Multiple subsolid nodules all <6 mm in a low-risk patient. "
                    "CT at 3-6 months; if persistent and no risk factors, no further routine "
                    "follow-up per Fleischner 2017."
                )
        else:
            cat = "Multiple Subsolid Nodules ≥6 mm"
            mgmt = (
                "CT at 3-6 months. Subsequent management guided by most suspicious nodule. "
                "Periodic imaging up to 5 years."
            )
            reasoning.append(
                f"Multiple subsolid nodules (largest {size} mm, ≥6 mm). "
                "CT at 3-6 months, management guided by most suspicious nodule, "
                "with periodic imaging up to 5 years per Fleischner 2017."
            )

    return _build_result(
        risk_category=risk,
        nodule_type="GGN",
        nodule_size=size,
        nodule_count=count,
        category_guideline=cat,
        management=mgmt,
        reasoning=" ".join(reasoning),
    )


# ======================================================================
# Part-Solid Nodule Classification
# ======================================================================

def _classify_part_solid(features, risk, size, count, reasoning):
    """Apply Fleischner part-solid nodule recommendations."""
    solid_comp = features.get("solid_component_size")

    if count == "Single":
        if size < 6:
            cat = "Single Part-Solid Nodule <6 mm"
            mgmt = "No routine follow-up."
            reasoning.append(
                "Part-solid nodule <6 mm. Very small part-solid nodules are difficult to "
                "characterize. No routine follow-up per Fleischner 2017."
            )
        else:
            if solid_comp and solid_comp >= 6:
                cat = "Single Part-Solid Nodule ≥6 mm with Solid Component ≥6 mm"
                mgmt = (
                    "HIGHLY SUSPICIOUS — recommend PET/CT, biopsy, or surgical evaluation."
                )
                reasoning.append(
                    f"Part-solid nodule {size} mm with solid component {solid_comp} mm (≥6 mm). "
                    "This is highly suspicious for invasive cancer. PET/CT, biopsy, or "
                    "surgical evaluation is recommended per Fleischner 2017."
                )
            else:
                cat = "Single Part-Solid Nodule ≥6 mm"
                mgmt = (
                    "CT at 3-6 months to confirm persistence. "
                    "If persistent and solid component <6 mm: annual CT for 5 years. "
                    "If solid component grows to ≥6 mm: PET/CT, biopsy, or surgery."
                )
                if solid_comp:
                    reasoning.append(
                        f"Part-solid nodule {size} mm with solid component {solid_comp} mm (<6 mm). "
                        "CT at 3-6 months to confirm persistence, then annual CT for 5 years "
                        "if solid component remains <6 mm per Fleischner 2017."
                    )
                else:
                    reasoning.append(
                        f"Part-solid nodule {size} mm (≥6 mm). CT at 3-6 months to confirm "
                        "persistence. Annual CT for 5 years if solid component <6 mm, with "
                        "escalation to PET/biopsy if solid component reaches ≥6 mm per Fleischner 2017."
                    )
    else:  # Multiple
        if size < 6:
            cat = "Multiple Subsolid Nodules <6 mm"
            mgmt = "CT at 3-6 months. If persistent, manage based on risk profile."
            reasoning.append(
                "Multiple subsolid (part-solid) nodules all <6 mm. "
                "CT at 3-6 months per Fleischner 2017."
            )
        else:
            cat = "Multiple Subsolid Nodules ≥6 mm"
            mgmt = (
                "CT at 3-6 months. Management guided by most suspicious nodule. "
                "Periodic imaging up to 5 years."
            )
            reasoning.append(
                f"Multiple subsolid nodules (largest {size} mm, ≥6 mm). "
                "CT at 3-6 months, guided by the most suspicious nodule, "
                "with periodic imaging up to 5 years per Fleischner 2017."
            )

    return _build_result(
        risk_category=risk,
        nodule_type="Part-Solid",
        nodule_size=size,
        nodule_count=count,
        category_guideline=cat,
        management=mgmt,
        reasoning=" ".join(reasoning),
    )


# ======================================================================
# Helpers
# ======================================================================

def _build_result(risk_category, nodule_type, nodule_size, nodule_count,
                  category_guideline, management, reasoning):
    """Build the standardized result dict."""
    return {
        "risk_category": risk_category,
        "nodule_type": nodule_type,
        "nodule_size": str(nodule_size) if nodule_size is not None else "",
        "nodule_count": nodule_count,
        "category_guideline": category_guideline,
        "management_recommendation": management,
        "reasoning": reasoning,
    }
