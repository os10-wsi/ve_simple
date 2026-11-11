# Stage 0 report (development sandbox, 200 proteins + sanity set)

Run on 200 proteins (50 MULTIPASS + 50 SOLUBLE per organism, random, seed 0) drawn from a random
pool of reviewed human and yeast proteins downloaded into a ve_simple-layout cache, plus the 7
sanity proteins: 204 proteins in total, AlphaFold DB v6 models. Everything below comes from the
pipeline commands (`scripts/stage0.sh` runs the same steps on the cluster cache).

## Run

| quantity | value |
|---|---|
| proteins / unit rows | 204 / 1,839 |
| protein errors / unit errors | 0 / 0 |
| units skipped for voxel box > 256 A (non-voxel metrics still computed) | 9 (all CHAIN_FULL of large disordered proteins) |
| time per protein, one core (median / mean / max) | 4.9 / 5.7 / 25 s |
| projected full run (~27,000 proteins) | ~45 core-hours |
| TED domains verified against the cached v6 sequence | 355 of 356 (3 by shifting, 1 dropped) |
| proteins with no TED domain | 15 of 200 |

Median fraction of residues excluded by the default mask (pLDDT < 70, outside TED, segments < 10):
human SOLUBLE 0.60, human MULTIPASS 0.30, yeast SOLUBLE 0.40, yeast MULTIPASS 0.33.
Disorder inflates Rg/Rmin: CHAIN_FULL / CHAIN_TRIMMED median 1.08, 90th percentile 1.54.

## Sanity set (stage 4)

See `stage0/sanity.tsv`. 6 of 7 checks pass.

| protein | unit | result |
|---|---|---|
| human lysozyme P61626 | DOMAIN | **FAIL**: a/c = 1.59 against the spec's 1.2-1.5; Rg/Rmin = 0.99 and no belt as expected |
| spectrin repeat, SPTAN1 Q13813 (CATH 1.20.58.60) | DOMAIN | PASS: a/c 6.5, kappa2 0.82 |
| GPCR ADRB2 P07550 | CHAIN_TRIMMED | PASS: belt detected (slab contrast 2.8) |
| transporter GLUT1 P11166 | CHAIN_TRIMMED | PASS: belt detected |
| VDAC1 P21796 | CHAIN_TRIMMED | PASS as beta-barrel class; belt **not** detected (contrast 0.96) |
| tropomyosin TPM1 P09493 | CHAIN_TRIMMED, no_ted | PASS: a/c 27, 100% helix. TED has no domain, so the default CHAIN_TRIMMED is empty |
| alpha-synuclein P37840 | CHAIN_FULL | PASS: Rg/Rmin 3.0; no residue survives trimming |

## Belt detector

The spec's belt (widest run of 4 A slabs above the median, along principal axis 1) does not separate
MULTIPASS from SOLUBLE chains on Stage 0 (contrast AUC 0.61; width AUC 0.29). It was kept as columns
(`belt_*_A_kd` etc.) and supplemented by a direction scan: the most hydrophobic 30 A slab of the
surface shell over 200 directions, flanked by surface on both sides (`belt_contrast_scan_*`, AUC 0.89).
The detector used for the cross-check is `belt_contrast_scan_kd >= 2.0`, chosen from these Stage 0
distributions (class medians 1.1 soluble, 2.5 multipass), so the matrix below is optimistic:

| class_annotation | belt not detected | belt detected |
|---|---|---|
| MULTIPASS | 24 | 65 |
| SOLUBLE | 80 | 7 |

class_final therefore moves 24 MULTIPASS and 7 SOLUBLE proteins to AMBIGUOUS; MULTIPASS proteins
without a trimmed unit (no TED domain) keep their class.

## Deviations from the spec and decisions for review

1. **Cavities.** The spec's `interior & ~atom_mask & ~solvent` is every gap between atoms inside the
   molecular surface (~30% of its volume). It is reported as `void_interstitial`; `cavity_*` use enclosed
   probe-accessible space (probe-centre space not connected to the outside, grown by 1.4 A).
2. **Belt.** As above: literal version kept, scan version used for detection.
3. **Molecular surface calibration.** Interior = EDT distance to solvent probe centres > 1.4 + 0.2 A; the
   0.2 A offset was calibrated on spheres (volume and area within 2% for radii 8-16 A).
4. **TED globularity.** TED has no "Qscore"; `norm_rg` and `packing_density` are stored. `norm_rg`
   correlates with Rg/Rmin_sphere at rho 0.81 and with a/c at 0.74: it is mostly elongation/compactness.
5. **Beta barrels** are missed by the surface belt detector because the pore lining is surface. With
   the rule you chose, class_final moves barrels without a belt to AMBIGUOUS; excluding barrels from the
   cross-check is a one-line change.
6. **TED exclusion removes non-globular structure**: long coiled coils and other regions without a TED
   domain disappear from CHAIN_TRIMMED. The `no_ted` variant keeps them.
7. **Hydrophobicity scales**: Wimley-White octanol values are negated so higher = more hydrophobic, like
   Kyte-Doolittle. On the atomic scale carbonyl/carboxyl/amide/guanidino carbons score 0.
8. **DSSP** is mdtraj's implementation (simplified H/E/C). An SSE is a helix of >= 4 or a strand of >= 2 residues.

## First observations (Stage 0 only; small groups, not conclusions)

- Hydrophobicity distribution separates MEMBRANE from SOLUBLE domains almost perfectly and survives the
  CATH class 1 control (h_mean, inversion index, H(depth) slope, slab contrast: stratified AUC 0.95-0.99).
- Membrane domains are more elongated/prolate than soluble domains (a/c AUC 0.79, prolateness 0.79),
  but this weakens at pLDDT >= 90 (AUC ~0.6), and it does not hold for whole trimmed chains
  (MULTIPASS vs SOLUBLE a/c AUC 0.46).
- No group is oblate: median prolateness is positive everywhere; TM_CORE is the most spherical (28% oblate).

Full tables: `stage0/findings.md`; figures `stage0/distributions.png`, `stage0/embeddings.png`.
