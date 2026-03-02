"""
NLP Extraction Layer for Pulmonary Nodule Classification System.

Extracts structured clinical features from free-text radiology reports using
regex-based pattern detection. No ML models are used.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


def extract_features(report: str) -> dict:
    """
    Extract all clinical features from a free-text radiology report.

    Returns a dict with keys:
        age, smoking, pack_years, nodule_sizes, nodule_types, nodule_count,
        upper_lobe, spiculated, emphysema, family_history, occupational_exposure,
        immunocompromised, known_cancer, is_screening, status, solid_component_size,
        perifissural, lymphadenopathy, warnings
    """
    text = report.strip()
    text_lower = text.lower()
    warnings = []

    features = {
        "age": _extract_age(text, warnings),
        "smoking": _extract_smoking(text_lower),
        "pack_years": _extract_pack_years(text_lower),
        "nodule_sizes": _extract_sizes(text, warnings),
        "nodule_types": _extract_nodule_types(text_lower),
        "nodule_count": _extract_nodule_count(text_lower),
        "upper_lobe": _detect_upper_lobe(text_lower),
        "spiculated": _detect_spiculation(text_lower),
        "emphysema": _detect_emphysema(text_lower),
        "family_history": _detect_family_history(text_lower),
        "occupational_exposure": _detect_occupational_exposure(text_lower),
        "immunocompromised": _detect_immunocompromised(text_lower),
        "known_cancer": _detect_known_cancer(text_lower),
        "is_screening": _detect_screening(text_lower),
        "status": _extract_status(text_lower),
        "solid_component_size": _extract_solid_component(text, warnings),
        "perifissural": _detect_perifissural(text_lower),
        "lymphadenopathy": _detect_lymphadenopathy(text_lower),
        "warnings": warnings,
    }

    # Derive primary nodule size (largest)
    if features["nodule_sizes"]:
        features["primary_size"] = max(features["nodule_sizes"])
    else:
        features["primary_size"] = None
        warnings.append("Could not extract nodule size from report.")

    # Derive primary nodule type
    features["primary_type"] = _derive_primary_type(features)

    # Derive count label
    if features["nodule_count"] > 1:
        features["count_label"] = "Multiple"
    else:
        features["count_label"] = "Single"

    return features


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------

def _extract_age(text: str, warnings: list) -> int | None:
    """Extract patient age from report text."""
    patterns = [
        r'(\d{1,3})\s*[-–]\s*year\s*[-–]?\s*old',
        r'age\s*(?:of\s+)?(\d{1,3})',
        r'(\d{1,3})\s*y/?o\b',
        r'(\d{1,3})\s*year\s+old',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                return age
    warnings.append("Could not extract age from report.")
    return None


# ---------------------------------------------------------------------------
# Smoking
# ---------------------------------------------------------------------------

def _extract_smoking(text_lower: str) -> bool:
    """Detect if patient has smoking history."""
    positive = [
        r'smok(?:er|ing|ed)',
        r'tobacco',
        r'pack[\s-]*year',
        r'former\s+smoker',
        r'current\s+smoker',
        r'heavy\s+smoker',
    ]
    negative = [
        r'never\s+smok',
        r'non[\s-]*smok',
        r'no\s+smoking',
        r'no\s+tobacco',
        r'no\s+history\s+of\s+smoking',
    ]
    for pat in negative:
        if re.search(pat, text_lower):
            return False
    for pat in positive:
        if re.search(pat, text_lower):
            return True
    return False


def _extract_pack_years(text_lower: str) -> float | None:
    """Extract pack-year history."""
    m = re.search(r'(\d+)\s*pack[\s-]*year', text_lower)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Nodule sizes
# ---------------------------------------------------------------------------

def _extract_sizes(text: str, warnings: list) -> list[float]:
    """
    Extract all NODULE sizes in mm from report.
    Handles: "7 mm", "7 x 5 mm" (→ mean), "average diameter 6 mm",
    "measuring 7 mm", "measures 8 mm", etc.
    Excludes sizes that belong to lymph nodes or non-nodule structures.
    """
    sizes = []

    # Pattern: N x M mm → mean
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm', text, re.IGNORECASE):
        # Skip if this is describing a lymph node
        context_start = max(0, m.start() - 40)
        context = text[context_start:m.end() + 40].lower()
        if re.search(r'lymph\s*node', context):
            continue
        d1, d2 = float(m.group(1)), float(m.group(2))
        sizes.append(round((d1 + d2) / 2, 1))

    # Pattern: "average diameter N mm"
    for m in re.finditer(r'average\s+diameter\s+(\d+(?:\.\d+)?)\s*mm', text, re.IGNORECASE):
        sizes.append(float(m.group(1)))

    # Pattern: simple "N mm" not already captured by AxB
    for m in re.finditer(r'(?<![x×\d])\b(\d+(?:\.\d+)?)\s*mm\b', text, re.IGNORECASE):
        val = float(m.group(1))
        # Skip if preceded by "x N" pattern (AxB)
        start = m.start()
        prefix = text[max(0, start - 20):start]
        if re.search(r'\d+\s*[x×]\s*$', prefix):
            continue
        # Skip if this size belongs to a lymph node, not a nodule
        context_start = max(0, start - 40)
        context_end = min(len(text), m.end() + 40)
        context = text[context_start:context_end].lower()
        if re.search(r'lymph\s*node', context):
            continue
        if 1 <= val <= 100:  # reasonable nodule sizes
            sizes.append(val)

    return sizes


# ---------------------------------------------------------------------------
# Solid component size
# ---------------------------------------------------------------------------

def _extract_solid_component(text: str, warnings: list) -> float | None:
    """Extract the solid component size for part-solid nodules.
    
    Returns the CURRENT (most recent) solid component size.
    """
    patterns = [
        r'solid\s+component\s+(?:now\s+)?(?:measuring\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*mm',
        r'(\d+(?:\.\d+)?)\s*mm\s+solid\s+component',
        r'solid\s+(?:portion|part)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*mm',
        r'solid\s+component\s+(?:of|is|was|measures?)\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*mm',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Nodule type
# ---------------------------------------------------------------------------

def _extract_nodule_types(text_lower: str) -> list[str]:
    """Detect nodule types present in the report."""
    types = []
    if re.search(r'part[\s-]*solid|mixed\s+(solid|density)', text_lower):
        types.append("Part-Solid")
    if re.search(r'ground[\s-]*glass|ggo|ggn|non[\s-]*solid', text_lower):
        # Only add GGN if part-solid was not already detected, or if
        # there's a separate pure GGN mention
        if "Part-Solid" not in types or re.search(r'pure\s+ground[\s-]*glass|pure\s+gg[no]', text_lower):
            types.append("GGN")
    if re.search(r'\bsolid\b', text_lower):
        # Only add Solid if no subsolid type was found, or if "solid nodule" is explicitly mentioned
        if not types or re.search(r'solid\s+nodule|solid\s+pulmonary', text_lower):
            types.append("Solid")

    # If part-solid detected, remove standalone Solid to avoid confusion
    if "Part-Solid" in types and "Solid" in types:
        # Keep Solid only if there are multiple nodules with separate solid ones
        if not re.search(r'solid\s+nodule', text_lower):
            types.remove("Solid")

    if not types:
        types.append("Solid")  # default
    return types


def _derive_primary_type(features: dict) -> str:
    """Derive the primary (most suspicious) nodule type."""
    types = features["nodule_types"]
    # Priority: Part-Solid > GGN > Solid (Part-Solid is most suspicious)
    if "Part-Solid" in types:
        return "Part-Solid"
    if "GGN" in types:
        return "GGN"
    return "Solid"


# ---------------------------------------------------------------------------
# Nodule count
# ---------------------------------------------------------------------------

def _extract_nodule_count(text_lower: str) -> int:
    """Estimate the number of nodules described."""
    if re.search(r'multiple\s+(solid\s+|subsolid\s+|pulmonary\s+)?nodules', text_lower):
        return 3  # placeholder for "multiple"
    # Only count as 0 if the report says there are no nodules at all
    # (not "no other nodules" which implies at least one exists)
    if re.search(r'no\s+(pulmonary\s+)?nodules?\s+identified', text_lower):
        # But not if qualified by "no other" — that means there IS a nodule
        if not re.search(r'no\s+other\s+(pulmonary\s+)?nodules?', text_lower):
            return 0

    count = 0
    # Count ordinal/quantifier mentions
    quantifiers = {
        r'(single|solitary|one)\s+(solid\s+|ground[\s-]*glass\s+|part[\s-]*solid\s+)?nodule': 1,
        r'(two|2)\s+(solid\s+|pulmonary\s+)?nodule': 2,
        r'(three|3)\s+(solid\s+|pulmonary\s+)?nodule': 3,
        r'\bsecond\s+(smaller\s+)?\d+': 2,
        r'\banother\s+': 2,
        r'\badditional\s+': 2,
        r'previously\s+noted\s+\d+\s*mm': 2,  # implies a second nodule from prior
    }

    # Count distinct nodule descriptions ("N mm ... nodule" patterns)
    size_matches = re.findall(
        r'(\d+(?:\.\d+)?)\s*(?:[x×]\s*\d+(?:\.\d+)?\s*)?mm\s+(?:solid\s+|ground[\s-]*glass\s+|part[\s-]*solid\s+)?nodule',
        text_lower
    )
    if size_matches:
        count = max(count, len(set(size_matches)))

    for pat, val in quantifiers.items():
        if re.search(pat, text_lower):
            count = max(count, val)

    # Look for "largest" which implies multiple
    if re.search(r'largest|dominant', text_lower) and count < 2:
        count = 2

    return max(count, 1)  # at least 1 unless explicitly 0


# ---------------------------------------------------------------------------
# Location and morphology
# ---------------------------------------------------------------------------

def _detect_upper_lobe(text_lower: str) -> bool:
    return bool(re.search(r'upper\s+lobe', text_lower))


def _detect_spiculation(text_lower: str) -> bool:
    """Detect spiculation, handling negation."""
    if re.search(r'no\s+spicul|without\s+spicul|non[\s-]*spicul', text_lower):
        stripped = re.sub(r'no\s+spicul\w*|without\s+spicul\w*|non[\s-]*spicul\w*', '', text_lower)
        return bool(re.search(r'spicul', stripped))
    return bool(re.search(r'spicul', text_lower))


def _detect_emphysema(text_lower: str) -> bool:
    """Detect emphysema, handling negation (e.g., 'No emphysema')."""
    if re.search(r'no\s+emphysema|without\s+emphysema', text_lower):
        # Check if there's also a positive mention elsewhere
        stripped = re.sub(r'no\s+emphysema|without\s+emphysema', '', text_lower)
        return bool(re.search(r'emphysema', stripped))
    return bool(re.search(r'emphysema', text_lower))


def _detect_family_history(text_lower: str) -> bool:
    """Detect family history of cancer, handling negation."""
    negated = r'no\s+family\s+history|without\s+family\s+history|no\s+known\s+family\s+history'
    positive = r'family\s+history\s+of\s+(lung\s+)?cancer|father\s+had\s+(lung\s+)?cancer|mother\s+had\s+(lung\s+)?cancer|first[\s-]*degree\s+relative'
    if re.search(negated, text_lower):
        stripped = re.sub(negated + r'[^.]*', '', text_lower)
        return bool(re.search(positive, stripped))
    return bool(re.search(positive, text_lower))


def _detect_occupational_exposure(text_lower: str) -> bool:
    """Detect occupational exposure, handling negation."""
    negated = r'no\s+occupational\s+exposure|without\s+occupational\s+exposure|no\s+known\s+exposure'
    positive = r'asbestos|radon|uranium|occupational\s+exposure|carcinogen'
    if re.search(negated, text_lower):
        stripped = re.sub(negated + r'\w*', '', text_lower)
        return bool(re.search(positive, stripped))
    return bool(re.search(positive, text_lower))


def _detect_immunocompromised(text_lower: str) -> bool:
    return bool(re.search(r'immunocompromis|immunosuppress|hiv|transplant|chemotherapy', text_lower))


def _detect_known_cancer(text_lower: str) -> bool:
    return bool(re.search(r'known\s+(primary\s+)?cancer|history\s+of\s+cancer|metasta', text_lower))


def _detect_screening(text_lower: str) -> bool:
    return bool(re.search(r'screening|ldct|low[\s-]*dose\s+ct', text_lower))


def _detect_perifissural(text_lower: str) -> bool:
    return bool(re.search(r'perifissural|along\s+the\s+.{0,20}fissure|intrapulmonary\s+lymph\s+node', text_lower))


def _detect_lymphadenopathy(text_lower: str) -> bool:
    """Detect presence (not absence) of lymphadenopathy."""
    # Match all common negated forms including "No mediastinal or hilar lymphadenopathy"
    negated_patterns = [
        r'no\s+(?:mediastinal\s+(?:or\s+)?)?(?:hilar\s+)?lymphadenopathy',
        r'no\s+(?:hilar\s+(?:or\s+)?)?(?:mediastinal\s+)?lymphadenopathy',
        r'without\s+(?:mediastinal\s+|hilar\s+)*lymphadenopathy',
        r'no\s+lymphadenopathy',
    ]
    negated_re = '|'.join(negated_patterns)
    if re.search(negated_re, text_lower):
        # Strip negated forms and check for positive mentions elsewhere
        stripped = re.sub(negated_re, '', text_lower)
        if re.search(r'lymphadenopathy|lymph\s+node.{0,30}(prominent|enlarged|suspicious|borderline)', stripped):
            return True
        return False
    return bool(re.search(r'lymphadenopathy|lymph\s+node.{0,30}(prominent|enlarged|suspicious|borderline)', text_lower))


# ---------------------------------------------------------------------------
# Status (baseline / new / growing)
# ---------------------------------------------------------------------------

def _extract_status(text_lower: str) -> str:
    """Determine if the nodule is baseline, new, or growing.

    Priority: Growing > New > Stable > Baseline.
    Handles negation of 'new' (e.g., 'no new nodules').
    When multiple nodules have different statuses (e.g., one new, one stable),
    return the MOST SUSPICIOUS status since guidelines classify by the most
    suspicious finding.
    """
    # Growing takes highest priority
    if re.search(r'grow(ing|th|n)|increas(ed|ing)\s+in\s+size|interval\s+(increase|growth|enlargement)|now\s+measures?\s+\d+.*previously\s+\d+|previously\s+\d+.*now\s+\d+', text_lower):
        return "Growing"

    # New — but handle negation like 'no new nodules'
    new_pattern = r'\bnew\b|interval\s+development|not\s+seen\s+on\s+prior|not\s+(present|seen|identified)\s+(on|in)\s+prior|newly\s+(developed|identified)'
    negated_new = r'no\s+new\s+nodule|without\s+new|no\s+new\s+finding'
    has_new = bool(re.search(new_pattern, text_lower))
    has_negated_new = bool(re.search(negated_new, text_lower))

    # If "new" is present but negated, strip the negated form and re-check
    if has_new and has_negated_new:
        stripped = re.sub(negated_new, '', text_lower)
        has_new = bool(re.search(new_pattern, stripped))

    # New takes priority over Stable — per guidelines, classify by the
    # MOST SUSPICIOUS finding when multiple nodules have different statuses
    if has_new:
        return "New"

    if re.search(r'stable|unchanged|no\s+interval\s+change', text_lower):
        return "Stable"

    if re.search(r'baseline|initial\s+screen|first\s+screen|no\s+prior\s+(imaging|comparison|ct|exam)', text_lower):
        return "Baseline"

    return "Baseline"  # default
