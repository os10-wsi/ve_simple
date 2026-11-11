# Findings

Proteins: 204; unit rows: 1839; domains (default mask): 296.

Domain groups by organism:

|   taxid |   MEMBRANE_DOMAIN |   MIXED |   SOLUBLE_DOMAIN |   SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN |
|--------:|------------------:|--------:|-----------------:|-------------------------------------:|
|    9606 |                51 |       0 |               71 |                                   12 |
|  559292 |                44 |       2 |               94 |                                   22 |

## 1. Size invariance (Spearman |rho| vs N within SOLUBLE_DOMAIN; |rho| > 0.3 flagged)

Most size-invariant metrics:

| metric                  |   n |   spearman_rho_vs_N |     nu |
|:------------------------|----:|--------------------:|-------:|
| frac_strand             | 165 |              -0.001 | -0.283 |
| frac_helix              | 165 |              -0.004 |  0.006 |
| belt_width_scan_A_kd    |  86 |              -0.005 |  0.008 |
| hmoment_angle_kd        | 165 |               0.018 |  0.038 |
| belt_width_A_ww         | 165 |              -0.019 | -0.037 |
| solidity_ses            | 165 |               0.019 |  0.026 |
| ca_acylindricity        | 165 |              -0.02  | -0.028 |
| inversion_index_kd      | 165 |               0.022 |        |
| inversion_index_ww      | 165 |              -0.023 |        |
| acylindricity           | 165 |               0.029 |  0.072 |
| belt_center_scan_A_kd   |  86 |              -0.03  | -0.397 |
| b_over_c                | 165 |              -0.031 | -0.032 |
| belt_contrast_scan_atom |  86 |              -0.036 |  0.044 |
| belt_width_scan_A_atom  |  86 |               0.036 | -0.189 |
| rg_over_rmin_sphere     | 165 |              -0.04  | -0.031 |
| belt_contrast_scan_ww   |  86 |              -0.041 |  0.21  |
| h_mean_kd               | 165 |               0.044 | -0.175 |
| n_helices_gt30          | 165 |               0.045 |  0.814 |
| belt_contrast_scan_kd   |  86 |              -0.057 |  0.196 |
| mean_helix_len          | 165 |               0.06  |  0.032 |

Most size-dependent metrics:

| metric            |   n |   spearman_rho_vs_N |    nu |
|:------------------|----:|--------------------:|------:|
| rmin_empirical    | 165 |               1     | 0.38  |
| v_vdw             | 165 |               0.997 | 0.994 |
| rmin_sphere       | 165 |               0.997 | 0.331 |
| v_ses             | 165 |               0.997 | 1.034 |
| void_interstitial | 165 |               0.992 | 1.113 |
| a_ses             | 165 |               0.96  | 0.759 |
| sasa              | 165 |               0.947 | 0.707 |
| v_hull            | 165 |               0.941 | 1.008 |
| a_hull            | 165 |               0.906 | 0.66  |
| ca_lambda3        | 165 |               0.823 | 0.847 |

## 2-3. Membrane vs soluble domains after the CATH class 1 control

Size- and organism-stratified AUC (P(membrane > soluble)); `survives` = |AUC-0.5| >= 0.15 overall and within CATH class 1, same direction.

| metric                          |   auc_all |   auc_cath1 |   n_a_cath1 |   n_b_cath1 | survives_cath1_control   |
|:--------------------------------|----------:|------------:|------------:|------------:|:-------------------------|
| h_mean_ww                       |     0.981 |       0.997 |          62 |          42 | True                     |
| h_mean_kd                       |     0.957 |       0.992 |          62 |          42 | True                     |
| h_depth_slope_ww                |     0.022 |       0.011 |          62 |          42 | True                     |
| h_depth_slope_kd                |     0.017 |       0.017 |          62 |          42 | True                     |
| belt_contrast_scan_kd           |     0.967 |       0.983 |          62 |          42 | True                     |
| h_mean_atom                     |     0.954 |       0.977 |          62 |          42 | True                     |
| inversion_index_kd              |     0.965 |       0.975 |          62 |          42 | True                     |
| belt_contrast_scan_atom         |     0.948 |       0.972 |          62 |          42 | True                     |
| h_depth_slope_atom              |     0.017 |       0.045 |          62 |          42 | True                     |
| inversion_index_ww              |     0.944 |       0.949 |          62 |          42 | True                     |
| belt_contrast_scan_ww           |     0.92  |       0.907 |          62 |          42 | True                     |
| belt_profile_contrast_scan_ww   |     0.829 |       0.872 |          62 |          42 | True                     |
| belt_width_A_kd                 |     0.204 |       0.14  |          62 |          42 | True                     |
| inversion_index_atom            |     0.821 |       0.856 |          62 |          42 | True                     |
| belt_profile_contrast_scan_kd   |     0.859 |       0.847 |          62 |          42 | True                     |
| belt_normal_angle_pa1_atom      |     0.272 |       0.179 |          62 |          42 | True                     |
| radial_slope_kd                 |     0.845 |       0.813 |          62 |          42 | True                     |
| rmin_empirical                  |     0.675 |       0.807 |          62 |          42 | True                     |
| belt_normal_angle_pa1_kd        |     0.28  |       0.207 |          62 |          42 | True                     |
| v_vdw                           |     0.68  |       0.788 |          62 |          42 | True                     |
| rmin_sphere                     |     0.68  |       0.788 |          62 |          42 | True                     |
| belt_width_scan_A_atom          |     0.188 |       0.214 |          62 |          42 | True                     |
| belt_profile_contrast_scan_atom |     0.188 |       0.214 |          62 |          42 | True                     |
| belt_normal_angle_pa1_ww        |     0.317 |       0.217 |          62 |          42 | True                     |
| mean_helix_len                  |     0.923 |       0.782 |          62 |          42 | True                     |

## 5. Prolate vs oblate

| group                              |   n |   median_S |    q25 |   q75 |   frac_oblate | oblate_reference_fits   |
|:-----------------------------------|----:|-----------:|-------:|------:|--------------:|:------------------------|
| MEMBRANE_DOMAIN                    |  95 |      0.149 |  0.063 | 0.305 |         0.042 | False                   |
| SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN |  34 |      0.076 |  0.025 | 0.201 |         0.118 | False                   |
| SOLUBLE_DOMAIN                     | 165 |      0.034 |  0.007 | 0.098 |         0.182 | False                   |
| TM_CORE (MULTIPASS)                | 100 |      0.026 | -0.001 | 0.179 |         0.28  | False                   |

## 4. Where does elongation live?

| set                                |   n |   median_n_res |   median_a_over_c |   median_kappa2 |   median_prolateness |
|:-----------------------------------|----:|---------------:|------------------:|----------------:|---------------------:|
| MULTIPASS chain (trimmed)          |  89 |          269   |             2.15  |           0.193 |                0.147 |
| MULTIPASS TM_CORE                  | 100 |          188   |             1.78  |           0.098 |                0.026 |
| MEMBRANE_DOMAIN                    |  95 |          208   |             2.144 |           0.198 |                0.149 |
| SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN |  34 |          101.5 |             1.85  |           0.141 |                0.076 |
| SOLUBLE chain (trimmed)            |  89 |          173   |             2.116 |           0.2   |                0.131 |
| SOLUBLE_DOMAIN                     | 165 |          114   |             1.741 |           0.108 |                0.034 |

## 6. Recovering the UniProt membrane label from structure alone

| unit          | model                            |   auc |   auc_oriented |   n_pos |   n_neg |
|:--------------|:---------------------------------|------:|---------------:|--------:|--------:|
| DOMAIN        | logistic: all metrics            | 0.998 |          0.998 |      95 |     165 |
| DOMAIN        | logistic: hydrophobicity metrics | 0.989 |          0.989 |      95 |     165 |
| DOMAIN        | h_mean_ww                        | 0.987 |          0.987 |      95 |     165 |
| DOMAIN        | h_depth_slope_kd                 | 0.029 |          0.971 |      95 |     165 |
| DOMAIN        | h_mean_kd                        | 0.966 |          0.966 |      95 |     165 |
| DOMAIN        | h_depth_slope_ww                 | 0.034 |          0.966 |      95 |     165 |
| DOMAIN        | h_mean_atom                      | 0.96  |          0.96  |      95 |     165 |
| DOMAIN        | inversion_index_kd               | 0.96  |          0.96  |      95 |     165 |
| CHAIN_TRIMMED | h_mean_ww                        | 0.958 |          0.958 |      89 |      89 |
| DOMAIN        | belt_contrast_scan_kd            | 0.955 |          0.955 |      95 |     165 |
| DOMAIN        | h_depth_slope_atom               | 0.046 |          0.954 |      95 |     165 |
| CHAIN_TRIMMED | logistic: hydrophobicity metrics | 0.946 |          0.946 |      89 |      89 |
| DOMAIN        | inversion_index_ww               | 0.942 |          0.942 |      95 |     165 |
| DOMAIN        | belt_contrast_scan_atom          | 0.937 |          0.937 |      95 |     165 |
| CHAIN_TRIMMED | h_mean_atom                      | 0.931 |          0.931 |      89 |      89 |

## 7. Sensitivity (metrics whose membrane/soluble call changes)

| metric                    |   auc_default |   auc_plddt50 |   auc_plddt90 |   auc_class_final |   auc_chain_default |   auc_chain_no_ted | conclusion_moves_with                |
|:--------------------------|--------------:|--------------:|--------------:|------------------:|--------------------:|-------------------:|:-------------------------------------|
| rg_mass                   |         0.807 |         0.806 |         0.613 |             0.804 |               0.447 |              0.438 | plddt90                              |
| rg_ca                     |         0.798 |         0.8   |         0.605 |             0.793 |               0.442 |              0.434 | plddt90                              |
| rmin_empirical            |         0.675 |         0.651 |         0.497 |             0.675 |               0.595 |              0.535 | plddt90                              |
| rg_ca_over_rmin_empirical |         0.751 |         0.767 |         0.632 |             0.744 |               0.402 |              0.406 | plddt90                              |
| rh_proxy                  |         0.807 |         0.806 |         0.613 |             0.804 |               0.447 |              0.438 | plddt90                              |
| a_over_c                  |         0.786 |         0.792 |         0.648 |             0.803 |               0.462 |              0.466 | plddt90                              |
| a_over_b                  |         0.771 |         0.792 |         0.612 |             0.794 |               0.442 |              0.517 | plddt90                              |
| asphericity               |         0.795 |         0.806 |         0.643 |             0.816 |               0.457 |              0.511 | plddt90                              |
| prolateness               |         0.785 |         0.802 |         0.609 |             0.805 |               0.443 |              0.515 | plddt90                              |
| ca_lambda1                |         0.836 |         0.85  |         0.646 |             0.844 |               0.445 |              0.453 | plddt90                              |
| ca_ax_a                   |         0.836 |         0.85  |         0.646 |             0.844 |               0.445 |              0.453 | plddt90                              |
| ca_a_over_c               |         0.782 |         0.788 |         0.649 |             0.795 |               0.461 |              0.474 | plddt90                              |
| ca_a_over_b               |         0.763 |         0.786 |         0.589 |             0.784 |               0.442 |              0.522 | plddt90                              |
| ca_kappa2                 |         0.791 |         0.806 |         0.639 |             0.81  |               0.467 |              0.515 | plddt90                              |
| ca_asphericity            |         0.79  |         0.805 |         0.631 |             0.808 |               0.461 |              0.517 | plddt90                              |
| ca_prolateness            |         0.781 |         0.801 |         0.591 |             0.798 |               0.442 |              0.52  | plddt90                              |
| v_hull                    |         0.72  |         0.705 |         0.547 |             0.693 |               0.45  |              0.454 | plddt90                              |
| a_hull                    |         0.752 |         0.728 |         0.583 |             0.731 |               0.46  |              0.453 | plddt90                              |
| dmax_over_2rg             |         0.66  |         0.679 |         0.648 |             0.677 |               0.591 |              0.579 | plddt90                              |
| contact_density_ca        |         0.569 |         0.562 |         0.545 |             0.629 |               0.656 |              0.64  | no_ted (chain)                       |
| sasa                      |         0.716 |         0.7   |         0.544 |             0.69  |               0.515 |              0.481 | plddt90                              |
| sasa_scaled               |         0.688 |         0.669 |         0.625 |             0.649 |               0.459 |              0.45  | plddt90, class_final                 |
| n_helices                 |         0.648 |         0.638 |         0.678 |             0.67  |               0.62  |              0.658 | plddt90, class_final, no_ted (chain) |
| n_helices_gt30            |         0.755 |         0.78  |         0.559 |             0.755 |               0.839 |              0.815 | plddt90                              |
| hmoment_kd                |         0.61  |         0.584 |         0.594 |             0.645 |               0.648 |              0.705 | no_ted (chain)                       |

## 8. What TED's globularity terms capture (top |Spearman|)

| ted_term    | metric                    |   n |   spearman |
|:------------|:--------------------------|----:|-----------:|
| ted_norm_rg | rg_over_rmin_sphere       | 296 |      0.811 |
| ted_norm_rg | sphericity_hull           | 296 |     -0.767 |
| ted_norm_rg | a_over_c                  | 296 |      0.736 |
| ted_norm_rg | rg_ca_over_rmin_empirical | 296 |      0.735 |
| ted_norm_rg | kappa2                    | 296 |      0.733 |
| ted_norm_rg | lambda1                   | 296 |      0.726 |
| ted_norm_rg | ax_a                      | 296 |      0.726 |
| ted_norm_rg | asphericity               | 296 |      0.723 |
| ted_norm_rg | ca_lambda1                | 296 |      0.722 |
| ted_norm_rg | ca_ax_a                   | 296 |      0.722 |
| ted_norm_rg | ca_kappa2                 | 296 |      0.704 |
| ted_norm_rg | ca_a_over_c               | 296 |      0.702 |

PCA explained variance (first 5 PCs): 0.27, 0.20, 0.10, 0.05, 0.03. Figures: analysis/embeddings.png, analysis/distributions.png.

## Caveats

- Classes and domain labels come from UniProt annotation (`class_annotation`); `class_final` (after the structural belt cross-check) is used only as a sensitivity analysis, to avoid circularity with the structure-only ROC.
- The 0.15 AUC-effect threshold is an exploratory choice. The belt-detector threshold (slab contrast >= 2.0 Kyte-Doolittle units) was chosen by inspecting the Stage 0 distributions, so the belt confusion matrix on Stage 0 is optimistic; re-check it on the full run.
- TED assigns no domain to long coiled coils and other non-globular structure, so the default exclusion removes them entirely from CHAIN_TRIMMED; the no_ted variant keeps them.
- The surface-based belt detector misses beta barrels: the water-filled pore is solvent-exposed surface and dilutes the hydrophobic band (VDAC1 contrast 0.96).
- TED domains are from AFDB v4 models; domains whose sequence (md5) does not match the cached model within +-30 residues are dropped.
- Small groups (e.g. beta barrels, CATH class 1 soluble domains in a given size bin) give unstable AUCs; check n columns.
