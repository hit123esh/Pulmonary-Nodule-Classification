# Automated Pulmonary Nodule Classification System

A **deterministic, rule-based clinical decision engine** that processes free-text radiology reports and classifies pulmonary nodules using:

- **Fleischner Society 2017** (Incidental Nodules)
- **Lung-RADS v2022** (Lung Cancer Screening)

This project uses **regex-based NLP extraction + guideline-driven logic**.  
No machine learning — purely structured clinical reasoning.

---

## Overview

The system:

1. Parses free-text radiology reports  
2. Extracts structured clinical features (age, smoking, size, type, morphology, growth, etc.)  
3. Applies the correct guideline based on case type  
4. Generates classification, management recommendation, and reasoning  
5. Outputs results to a structured CSV file  

It processes **20 radiology cases**:
- F-1 to F-10 → Fleischner 2017  
- L-1 to L-10 → Lung-RADS v2022  

---

## Architecture

The project follows a modular, 3-layer structure:

- **NLP Layer** → `nlp_extractor.py`  
  Extracts features using regex with negation handling  

- **Decision Engine Layer**
  - `fleischner_engine.py`
  - `lungrads_engine.py`
  Implements deterministic guideline tables  

- **Output Layer** → `output_generator.py`  
  Merges results and exports `completed_assignment.csv`

Main entry point:

```bash
python3 classify_nodules.py
