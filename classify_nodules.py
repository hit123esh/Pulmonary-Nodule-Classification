#!/usr/bin/env python3
"""
Pulmonary Nodule Classification System — Main Orchestrator.

Processes all radiology cases from assignment_cases.csv, applies the correct
clinical guideline engine (Fleischner 2017 or Lung-RADS v2022), and outputs
completed_assignment.csv with classifications and reasoning.

Usage:
    python classify_nodules.py
"""

import logging
import sys
from pathlib import Path

import nlp_extractor
import fleischner_engine
import lungrads_engine
import output_generator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("classification.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    base_dir = Path(__file__).parent
    input_csv = base_dir / "assignment_cases.csv"
    output_csv = base_dir / "completed_assignment.csv"

    logger.info("=" * 60)
    logger.info("Pulmonary Nodule Classification System")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load cases
    # ------------------------------------------------------------------
    cases = output_generator.read_cases(str(input_csv))
    logger.info(f"Processing {len(cases)} cases...")

    # ------------------------------------------------------------------
    # Process each case
    # ------------------------------------------------------------------
    for case in cases:
        case_id = case.get("Case_ID", "UNKNOWN")
        case_type = case.get("Type", "")
        report = case.get("Report", "")

        logger.info("-" * 40)
        logger.info(f"Processing {case_id} (Type: {case_type})")

        # Step 1: NLP extraction
        features = nlp_extractor.extract_features(report)

        # Log any warnings
        for w in features.get("warnings", []):
            logger.warning(f"  [{case_id}] {w}")

        # Log extracted features
        logger.info(f"  Age: {features.get('age')}")
        logger.info(f"  Smoking: {features.get('smoking')}")
        logger.info(f"  Primary size: {features.get('primary_size')} mm")
        logger.info(f"  Primary type: {features.get('primary_type')}")
        logger.info(f"  Count: {features.get('count_label')}")
        logger.info(f"  Status: {features.get('status')}")
        logger.info(f"  Upper lobe: {features.get('upper_lobe')}")
        logger.info(f"  Spiculated: {features.get('spiculated')}")
        logger.info(f"  Emphysema: {features.get('emphysema')}")
        logger.info(f"  Perifissural: {features.get('perifissural')}")

        # Step 2: Route to correct engine
        if case_type == "Fleischner":
            result = fleischner_engine.classify(features)
        elif case_type == "Lung-RADS":
            result = lungrads_engine.classify(features)
        else:
            logger.error(f"  [{case_id}] Unknown case type: {case_type}")
            result = {
                "risk_category": "",
                "nodule_type": "",
                "nodule_size": "",
                "nodule_count": "",
                "category_guideline": f"ERROR: Unknown type '{case_type}'",
                "management_recommendation": "Manual review required.",
                "reasoning": f"Case type '{case_type}' is not recognized.",
            }

        # Step 3: Merge result
        output_generator.merge_result(case, result)

        logger.info(f"  → Guideline: {result.get('category_guideline')}")
        logger.info(f"  → Management: {result.get('management_recommendation')}")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_generator.write_output(cases, str(output_csv))

    logger.info("=" * 60)
    logger.info(f"COMPLETE — Output written to: {output_csv}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    for case in cases:
        cid = case.get("Case_ID", "")
        cat = case.get("Your_Category_or_Guideline", "")
        mgmt = case.get("Your_Management_Recommendation", "")
        print(f"  {cid:6s} | {cat:55s} | {mgmt[:60]}")
    print("=" * 60)
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
