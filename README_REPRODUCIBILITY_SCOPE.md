# Reproducibility scope

## Qatraneh model-selection recomputation
`reproducibility/recompute_qatraneh_gep.py` reconstructs the manuscript-stated G/P/E fitting and LOOCV logic directly from the 24-row Qatraneh observation table.

The implementation uses 501 logarithmically spaced E-model lambda candidates over the stated geometry-derived bounds, enforces A >= 0 for E and beta >= 0 for P, reselects nonlinear parameters inside every held-out fold, and applies the G -> P -> E complexity tie rule. It reproduces all 70 reported Qatraneh nested profile-window decisions and the quoted close model margins.

## Fully classified outer-tail record
The primary outer-tail table is restricted to 6-12 m windows where all three models are LOOCV-admissible in every fold. Among 27 disagreement windows, the complete-record class is tail-best in 24, the inner winner in 1, and the third model in 2.

The 5-m-inclusive 35-window reconstruction is retained as a secondary diagnostic. It yields 30 complete-class tail-best, 2 inner-winner tail-best, and 3 third-model tail-best cases. The three third-model cases are North Field at 5, 6, and 7 m, with G tail-best in all three.

## Retrospective observed-maximum comparison
Five Qatraneh profiles have C10 = 1.00: the full observed maximum had already appeared by 10 m. Three of those five nevertheless have D* = 12 m. This fixed-record comparison does not depend on the illustrative 90% threshold. Its denominator requires the complete record. It is not a prospective stopping criterion and does not show that a field observer would have judged the transect complete.

## Complete-record single-station model jackknife
The complete-record model-selection procedure was rerun after deleting each of the 12 Qatraneh stations in turn for every profile. The complete-record class is retained in 118/120 profile-deletion analyses; 8/10 profiles are invariant under all 12 deletions. The only changes are the two North Coarse profiles when 8 m is deleted (G -> P). Every deletion of 12 m or 14 m preserves the complete-record winner.

## Lambda-grid resolution sensitivity
The E-model search was repeated with 51, 101, 251, 501, and 1001 logarithmically spaced lambda candidates over the stated bounds. Every resolution reproduces the same 70/70 reported nested model decisions.

## Directional-symmetry sensitivity on Qatraneh
A sensitivity implementation allows the plateau-model beta coefficient to take either sign. On Qatraneh, this symmetric-plateau sensitivity reproduces all 70 reported nested decisions and the primary outer-tail and single-station-jackknife conclusions. The manuscript retains the one-sided plateau as its declared primary candidate because the complete Qatraneh profile directions are positive.

## External-data reproducibility boundary
The Daejeon branch is NOT an executable raw-data replay in this package. The official Lee et al. (2020) publisher supplementary XLSX is not redistributed, and the package does not include workbook-to-result extraction/fitting code. `build_data_and_figures.py` writes fixed Daejeon reconciliation and summary records; it does not read the workbook, recover profile runs, fit Daejeon models, or recompute their truncation results. Obtaining the workbook alone is therefore not a complete replay path. A future numerical replay would require both that source and a documented extraction/fitting implementation. The Daejeon results remain illustrative and are not external validation of Qatraneh.

## Figure-derived descriptive check
The Di Martino et al. (2026) values supplied here are approximate readings from published Figure 5, not raw measurements. They are rounded to the nearest 0.1 x 10^-6 m^3 kg^-1, support only a descriptive consistency check, and are not used to estimate k* or D*.

## Sign-count schema
`data/Qatraneh_R15R4_SparseSignRetention.csv` now names `total - positive` as `nonpositive`. It includes both negative and exactly zero Theil-Sen slopes. Only the column heading changes: all numeric values, R_k values, and k* values are preserved. The builder asserts that field-profile subsets have no zero slopes before using nonpositive counts in the negative-field-layout figure. The separate endpoint-anchored CSV already distinguishes negative and zero counts correctly and is unchanged.
