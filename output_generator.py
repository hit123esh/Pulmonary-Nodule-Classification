"""
Output Generation Layer for Pulmonary Nodule Classification System.

Handles reading input CSV, merging classification results, and writing
the completed output CSV.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Required output columns
OUTPUT_COLUMNS = [
    "Case_ID",
    "Type",
    "Report",
    "Your_Risk_Category",
    "Your_Nodule_Type",
    "Your_Nodule_Size",
    "Your_Nodule_Count",
    "Your_Category_or_Guideline",
    "Your_Management_Recommendation",
    "Your_Reasoning",
]


def read_cases(csv_path: str) -> list[dict]:
    """Read the assignment CSV and return list of case dicts."""
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize Type column
            case_type = row.get("Type", "").strip()
            if case_type.lower() in ("fleischner", "fleischner society", "fleischner 2017"):
                row["Type"] = "Fleischner"
            elif case_type.lower() in ("lung-rads", "lungrads", "lung rads", "lung-rads v2022"):
                row["Type"] = "Lung-RADS"
            cases.append(row)
    logger.info(f"Loaded {len(cases)} cases from {csv_path}")
    return cases


def merge_result(case: dict, result: dict) -> dict:
    """Merge classification result into the case row."""
    case["Your_Risk_Category"] = result.get("risk_category", "")
    case["Your_Nodule_Type"] = result.get("nodule_type", "")
    case["Your_Nodule_Size"] = result.get("nodule_size", "")
    case["Your_Nodule_Count"] = result.get("nodule_count", "")
    case["Your_Category_or_Guideline"] = result.get("category_guideline", "")
    case["Your_Management_Recommendation"] = result.get("management_recommendation", "")
    case["Your_Reasoning"] = result.get("reasoning", "")
    return case


def write_output(cases: list[dict], output_path: str) -> None:
    """Write completed assignment CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow(case)
    logger.info(f"Wrote {len(cases)} cases to {output_path}")
