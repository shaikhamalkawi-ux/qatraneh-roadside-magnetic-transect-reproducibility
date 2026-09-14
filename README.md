# Qatraneh Roadside Magnetic Transect — Reproducibility Package

This repository accompanies the manuscript **“Direction Retention and Spatial-Model Stability in Roadside Magnetic Transects.”**

It preserves the machine-readable Qatraneh observation table, derived result tables, and deterministic Python code used to reproduce the Qatraneh fixed-table analyses and the reported model-selection robustness checks.

## Scope

Included:
- 24-row rounded Qatraneh observation table;
- cutpoint and leave-one-distance results;
- exhaustive sparse-station sign-retention results;
- locked and separately recomputed Qatraneh nested G/P/E model decisions;
- model-margin, lambda-grid, outer-tail, observed-maximum, and single-station jackknife outputs;
- endpoint-anchored station-layout sensitivity and symmetric-plateau Qatraneh sensitivity outputs;
- deterministic figure/data builder and separate G/P/E and symmetric-plateau recomputation scripts.

Not redistributed:
- the official Daejeon publisher supplementary XLSX. The repository includes only the documented reconciliation and derived summary records already present in the manuscript package. No executable workbook-to-result Daejeon analysis is included. Obtaining the publisher supplement alone does not supply an end-to-end replay; both extraction and fitting code would also be needed.

No unavailable Qatraneh measurements, 11 m/13 m values, unrounded replicates, chemistry, mineralogy, or background measurements are synthesized.

## Reproduction

A Python environment with NumPy, pandas, SciPy, and Matplotlib is required.

From the repository root:

```bash
python reproducibility/recompute_qatraneh_gep.py
python reproducibility/lambda_grid_sensitivity.py
python reproducibility/recompute_qatraneh_gep_symmetricP.py
python reproducibility/build_data_and_figures.py
```

The separate model-recomputation script is the authoritative path for the Qatraneh nested G/P/E model-selection results. See `README_REPRODUCIBILITY_SCOPE.md` for the exact reproducibility boundary.

## Authors

1. Mahdi Salem Q. Lataifeh — Department of Physics, Yarmouk University, Irbid, Jordan
2. Ghassan Malkawi — Computer Information Science (CIS), Faculty of Computer Information Science, Higher Colleges of Technology, Al Ain Campus, Al Ain P.O. Box 17155, United Arab Emirates
3. Ahmed Abdelaziz Elsayed — Department of Computer Engineering and Computational Sciences, School of Engineering, Applied Sciences and Technology, Canadian University Dubai, Dubai P.O. Box 117781, United Arab Emirates
4. Haroun Albarghouthy — Higher Colleges of Technology, Al Ain Campus, Al Ain P.O. Box 17155, United Arab Emirates

Correspondence: **Haroun Albarghouthy** — `halbarghouthy@hct.ac.ae`

## Citation and DOI

`CITATION.cff` and `.zenodo.json` are included for GitHub/Zenodo archiving. A DOI is intentionally **not** written here until it is reserved or minted by Zenodo.

## Release note

This manuscript-associated package accompanies the manuscript prepared for journal submission and preserves the reported Qatraneh numerical results, supporting sensitivities, and the explicitly labeled figure-derived Di Martino et al. (2026) descriptive consistency check.

## Terminology note
Some historical machine-readable CSV files retain the column name `Route`. In the manuscript, this same side-by-measurement analytical unit is consistently termed a **profile**. The column name is preserved to avoid altering archived result schemas.

### Figure-derived Di Martino consistency check

`data/DiMartino2026_Figure5_DigitizedApprox.csv` contains approximate magnetic-susceptibility values read from published Figure 5 of Di Martino et al. (2026, DOI 10.3390/atmos17010114). `data/DiMartino2026_Figure5_DescriptiveCheck.csv` records the arithmetic fractions derived from the rounded figure readings (displayed in the manuscript as approximately 0.96 for railway and 0.83 for street) of the observed 3-45 m decline already realized by 15 m. These files are descriptive only, are not original raw measurements, and are not used to compute k* or D*. The plotted values were read from the published axis scale and rounded to the nearest 0.1 × 10^-6 m^3 kg^-1.

## Current reproducibility conventions

`data/Qatraneh_R15R4_SparseSignRetention.csv` uses the field name `nonpositive` for `total - positive`, because that count can include exactly zero Theil--Sen slopes. The numerical values, $R_k$ values, and $k^*$ values are unchanged. The figure builder verifies that the field-profile subsets used in the negative-layout panel contain no zero slopes.

Figure 2 is generated with the retrospective observed-maximum wording used in the manuscript. The complete-record denominator is unavailable prospectively; the figure is not a validated field-stopping rule.

The public repository preserves executable Qatraneh calculations and the declared non-executable Daejeon boundary. The immutable Git commit cited in the manuscript Data Availability Statement identifies the submission-associated source state.

## Licensing
Software/code: MIT License. Author-owned data, documentation and figures: CC BY 4.0. Third-party material retains its original terms. See `LICENSES.md`.
