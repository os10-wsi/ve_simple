"""Generate notebooks/membrane_vs_soluble.ipynb (narrative) and notebooks/explore.ipynb (browser).

Notebooks are generated from this script so their code can be reviewed in git; outputs are cleared.
"""
from pathlib import Path

import nbformat as nbf

SETUP = r'''import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Markdown, display
sys.path.insert(0, str(Path.cwd().parent))          # membrane_analysis/ (when not pip-installed)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from mvsa import analyses as A, plots as P, report as R, stats as S
from mvsa.data import load, available_predictors
from mvsa.groups import CLASS_ORDER, CLASS_LABEL, CLASS_COLOR, MEMBRANE, MAIN_PREDICTORS, PREDICTOR_GROUP, DIRECTION

# ---------------------------------------------------------------- parameters
DATA_DIR = Path(os.environ.get("MVSA_DATA", "../data"))     # written by `python -m mvsa.prepare`
FIG_DIR = Path(os.environ.get("MVSA_FIGURES", "../figures"))
INCLUDE_LIKELY = True       # count likely pathogenic / likely benign as labels
MIN_STARS = 0               # ClinVar review filter: 0 all, 1 drop 'no assertion criteria', 2 multiple submitters+
N_BOOT = int(os.environ.get("MVSA_NBOOT", 300))   # protein bootstrap resamples for CIs
MIN_PER_CLASS = 5           # per-protein AUCs need this many pathogenic AND benign variants
HUMAN = 9606

P.setup(FIG_DIR)
D = load(DATA_DIR)
prot = D["proteins"]
lab_all = D["labelled_variants"]
lab = S.labels(lab_all, include_likely=INCLUDE_LIKELY, min_stars=MIN_STARS)     # pathogenic (y=1) / benign (y=0)
lab = lab[lab.protein_class.notna()]
sample = D["variant_sample"]
PRED = [p for p in MAIN_PREDICTORS if p in available_predictors(lab)]
VEP = [p for p in ("alphamissense", "esm1v", "saprot") if p in PRED]
print("data:", D["meta"].get("created", "?"), "| predictors with data:", ", ".join(PRED))
missing = [p for p in MAIN_PREDICTORS if p not in PRED]
if missing:
    print("no data yet (skipped):", ", ".join(missing))'''

NARRATIVE = [
    ("md", "# Variant-effect prediction and structure: multipass membrane proteins vs all other proteins\n\n"
           "This notebook asks whether multipass membrane proteins are a blind spot for current variant-effect prediction, "
           "which would motivate a systematic mutagenesis study of membrane proteins. It is organised around three questions:\n\n"
           "1. **Clinical performance.** Do predictors separate ClinVar pathogenic from benign variants less well in membrane proteins?\n"
           "2. **Dynamic range and error.** Are scores compressed, mis-calibrated, or more contradictory in membrane proteins, "
           "and where do variants of uncertain significance accumulate?\n"
           "3. **Structure.** How do membrane proteins differ in shape, packing and the distribution of hydrophobicity, in ways that "
           "predictors trained mostly on soluble proteins would not capture?\n\n"
           "**Groups.** Every protein has one class from UniProt annotation (`class_annotation` from the shape pipeline). "
           "**Multipass membrane** (orange) is the group of interest; single-pass, beta-barrel, soluble and ambiguous "
           "(peripheral, lipid-anchored or conflicting annotation) proteins are each shown, plus all non-multipass proteins pooled.\n\n"
           "**Conventions.** Every predictor is oriented so that a higher score means more likely pathogenic, so an AUC above 0.5 "
           "is always better than chance. Confidence intervals resample whole proteins, because variants of one protein are not "
           "independent. Multiple comparisons are BH-adjusted across predictors. Clinical analyses use human proteins only (yeast "
           "has no ClinVar); structural analyses use both organisms."),
    ("code", SETUP),
    ("md", "## 0. Data overview"),
    ("code", "ov = A.overview(prot[prot.taxid == HUMAN], lab_all)\nov"),
    ("code", "cov = pd.DataFrame({c: [lab.loc[lab.protein_class == c, p].notna().mean() for p in PRED] for c in CLASS_ORDER}, index=PRED)\n"
             "print('Fraction of labelled variants with a score, by class:')\ncov.round(2)"),

    # -------------------------------------------------------------------------------- (i)
    ("md", "## 1. Clinical performance: pathogenic vs benign\n\n"
           "### 1.1 Pooled across proteins\nAUC over every labelled variant of each class, with protein-bootstrap 95% CIs. "
           "The right panel shows multipass minus each comparator; values below zero mean predictors do worse on multipass proteins."),
    ("code", "pooled = A.pooled_performance(lab, PRED, n_boot=N_BOOT)\n"
             "fig, ax = plt.subplots(1, 2, figsize=(12, 0.45 * len(PRED) + 1.6), gridspec_kw={'width_ratios': [1.2, 1]})\n"
             "P.forest(pooled[pooled['class'].isin(CLASS_ORDER)], 'predictor', 'class', ax=ax[0], groups=CLASS_ORDER,\n"
             "         title='AUC by protein class')\n"
             "d = pooled[pooled['class'].isin(['SOLUBLE', 'NON_MULTIPASS'])].rename(columns={'mp_minus_delta': 'dd', 'mp_minus_delta_low': 'dl', 'mp_minus_delta_high': 'dh'})\n"
             "P.forest(d, 'predictor', 'class', value='dd', lo='dl', hi='dh', ref=0, ax=ax[1], xlabel='AUC difference (multipass - comparator)',\n"
             "         title='Multipass minus comparator')\n"
             "ax[1].set_yticklabels([])\nfig.tight_layout(); P.save(fig, '1_1_pooled_auc')"),
    ("code", "cols = ['predictor', 'class', 'n_proteins', 'n_pathogenic', 'n_benign', 'auc', 'ci_low', 'ci_high', 'mp_minus_delta', 'mp_minus_delta_low', 'mp_minus_delta_high', 'mp_minus_q']\n"
             "pooled[cols].round(3)"),
    ("md", "### 1.2 ROC curves"),
    ("code", "show = [p for p in ('alphamissense', 'esm1v', 'thermompnn', 'blosum62') if p in PRED]\n"
             "fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.4), squeeze=False)\n"
             "for ax, p in zip(axes[0], show):\n"
             "    for c, (fpr, tpr, a) in A.roc_curves(lab, p, ['MULTIPASS', 'SOLUBLE', 'SINGLE_PASS', 'AMBIGUOUS']).items():\n"
             "        ax.plot(fpr, tpr, color=CLASS_COLOR[c], lw=2 if c == MEMBRANE else 1.3, label=f'{CLASS_LABEL[c]} {a:.2f}')\n"
             "    ax.plot([0, 1], [0, 1], color=P.AXIS, lw=0.8); ax.set_title(P.label(p), loc='left')\n"
             "    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(fontsize=7, loc='lower right')\n"
             "fig.tight_layout(); P.save(fig, '1_2_roc')"),
    ("md", "### 1.3 Per protein\nAUC inside each protein (from the ve_simple tables), for proteins with at least `MIN_PER_CLASS` pathogenic "
           "and benign variants. This removes any between-gene component but keeps fewer proteins."),
    ("code", "pp, pp_comp = A.per_protein_performance(D['protein_auc'], PRED, MIN_PER_CLASS)\n"
             "show = [p for p in PRED if p in set(pp.predictor)][:6]\n"
             "fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.4), squeeze=False)\n"
             "for ax, p in zip(axes[0], show):\n"
             "    P.class_violin(pp[pp.predictor == p], 'auc', [c for c in CLASS_ORDER if c != 'BETA_BARREL_MEMBRANE'], ax=ax, ylabel='per-protein AUC', title=P.label(p), ref=0.5)\n"
             "fig.tight_layout(); P.save(fig, '1_3_per_protein_auc')\n"
             "pp_comp.round(3)"),
    ("md", "### 1.4 Where in the protein\nInside multipass proteins, variants are split by location: transmembrane segment facing the lipid "
           "(exposed in the AlphaFold model) or buried, juxtamembrane (within 5 residues of a TM segment), and extramembrane buried or exposed. "
           "Soluble proteins are split into buried and exposed residues for reference (relative SASA threshold 0.2)."),
    ("code", "loc = A.location_performance(lab, PRED, n_boot=max(100, N_BOOT // 2))\n"
             "loc['group'] = loc['class'].map({'MULTIPASS': 'Multipass', 'SOLUBLE': 'Soluble'}) + ': ' + loc['location']\n"
             "show = [p for p in ('alphamissense', 'esm1v', 'saprot', 'thermompnn', 'blosum62') if p in PRED]\n"
             "fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.6), sharey=True, squeeze=False)\n"
             "for ax, p in zip(axes[0], show):\n"
             "    d = loc[loc.predictor == p].reset_index(drop=True)\n"
             "    y = np.arange(len(d))\n"
             "    for c in d['class'].unique():\n"
             "        m = (d['class'] == c).to_numpy()\n"
             "        ax.errorbar(d.auc[m], y[m], xerr=[(d.auc - d.ci_low)[m], (d.ci_high - d.auc)[m]], fmt='o', ms=6, elinewidth=1.4,\n"
             "                    color=CLASS_COLOR[c], ecolor=CLASS_COLOR[c], mec='white')\n"
             "    ax.axvline(0.5, color=P.AXIS, lw=1); ax.set_title(P.label(p), loc='left'); ax.set_xlabel('AUC')\n"
             "    ax.set_yticks(y, [f\"{g} (n={a}/{b})\" for g, a, b in zip(d.group, d.n_pathogenic, d.n_benign)], fontsize=7)\n"
             "axes[0][0].invert_yaxis(); fig.tight_layout(); P.save(fig, '1_4_auc_by_location')\n"
             "loc[['predictor', 'class', 'location', 'n_proteins', 'n_pathogenic', 'n_benign', 'auc', 'ci_low', 'ci_high']].round(3)"),
    ("md", "### 1.5 Is the gap explained by easier label mixes or structural context?\n"
           "*Matched* recomputes the AUC within strata of residue pLDDT, burial and the protein's fraction of pathogenic variants, "
           "weighting both classes by the multipass strata. *Within-protein* averages AUCs computed inside each protein. "
           "If the multipass deficit persists after both, it is not an artefact of which genes or residues happen to be labelled."),
    ("code", "matched = A.matched_performance(lab, [p for p in PRED if p in ('alphamissense', 'esm1v', 'saprot', 'thermompnn', 'foldx', 'blosum62')], n_boot=max(50, N_BOOT // 3))\n"
             "d = matched.copy(); d['group'] = d.estimate\n"
             "fig, ax = plt.subplots(figsize=(6.5, 0.5 * d.predictor.nunique() + 1.5))\n"
             "for j, (k, col) in enumerate(zip(['raw', 'matched', 'within-protein'], ['#8f8d85', '#eb6834', '#6250d6'])):\n"
             "    s = d[d.estimate == k].reset_index(drop=True); y = np.arange(len(s)) + (j - 1) * 0.22\n"
             "    ax.errorbar(s.delta, y, xerr=[s.delta - s.delta_low, s.delta_high - s.delta], fmt='o', color=col, label=k, ms=5)\n"
             "ax.axvline(0, color=P.AXIS); ax.set_yticks(range(s.shape[0]), [P.label(p) for p in s.predictor]); ax.invert_yaxis()\n"
             "ax.set_xlabel('AUC difference, multipass - soluble'); ax.legend(fontsize=8); fig.tight_layout(); P.save(fig, '1_5_matched')\n"
             "display(Markdown(R.matched_takeaway(matched)))"),
    ("md", "### Takeaway (i)"),
    ("code", "display(Markdown(R.performance_takeaway(pooled, 'SOLUBLE')))\ndisplay(Markdown(R.performance_takeaway(pooled, 'NON_MULTIPASS')))"),

    # -------------------------------------------------------------------------------- (ii)
    ("md", "## 2. Dynamic range and error\n\n### 2.1 Score distributions over all possible substitutions\n"
           "Every possible missense change (a random sample of the proteome-wide saturation tables), oriented so higher = more pathogenic."),
    ("code", "show = [p for p in PRED if p in sample][:8]\n"
             "fig, axes = plt.subplots(2, (len(show) + 1) // 2, figsize=(3.5 * ((len(show) + 1) // 2), 6.4), squeeze=False)\n"
             "for ax, p in zip(axes.ravel(), show):\n"
             "    P.ecdf(sample.assign(v=sample[p] * DIRECTION[p]), 'v', ['MULTIPASS', 'SINGLE_PASS', 'SOLUBLE', 'AMBIGUOUS'], ax=ax, xlabel=P.label(p) + ' (oriented)')\n"
             "for ax in axes.ravel()[len(show):]: ax.axis('off')\n"
             "fig.tight_layout(); P.save(fig, '2_1_score_ecdf')"),
    ("md", "Inside multipass proteins, scores of substitutions in TM segments vs elsewhere:"),
    ("code", "mp = sample[sample.protein_class == MEMBRANE]\n"
             "fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 3), squeeze=False)\n"
             "for ax, p in zip(axes[0], show):\n"
             "    for region, col in (('TM', '#eb6834'), ('Juxtamembrane', '#eda100'), ('Non-TM', '#2a78d6')):\n"
             "        v = np.sort((mp.loc[mp.region == region, p] * DIRECTION[p]).dropna())\n"
             "        if len(v): ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2, label=region)\n"
             "    ax.set_title(P.label(p), loc='left', fontsize=9); ax.legend(fontsize=7)\n"
             "fig.tight_layout(); P.save(fig, '2_1_tm_vs_rest_ecdf')\n"
             "A.tm_vs_rest_scores(D['protein_scores'], PRED).round(3)"),
    ("md", "### 2.2 Dynamic range per protein\nThe spread of each protein's scores (5th to 95th percentile over all substitutions), split into "
           "between-position spread (how much positions differ) and within-position spread (how much the substitutions at one position differ). "
           "A compressed range means the predictor cannot rank variants within that protein."),
    ("code", "dyn = A.dynamic_range(D['protein_scores'], PRED)\n"
             "ps = D['protein_scores']; ps = ps[ps.region == 'all']\n"
             "show = [p for p in ('alphamissense', 'esm1v', 'saprot', 'thermompnn', 'blosum62') if p in PRED]\n"
             "fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.3), squeeze=False)\n"
             "for ax, p in zip(axes[0], show):\n"
             "    P.class_violin(ps[ps.predictor == p], 'dynamic_range', [c for c in CLASS_ORDER if c != 'BETA_BARREL_MEMBRANE'], ax=ax, ylabel='5-95% range', title=P.label(p))\n"
             "fig.tight_layout(); P.save(fig, '2_2_dynamic_range')\n"
             "dyn.round(3)"),
    ("md", "### 2.3 Thresholds and error rates\nA threshold tuned on soluble proteins (Youden's J), or AlphaMissense's published cut-off, "
           "applied unchanged to every class. A higher false-negative rate in multipass proteins means pathogenic variants are missed there."),
    ("code", "thr = A.threshold_transfer(lab, PRED)\n"
             "d = thr[thr.threshold == 'Youden on soluble']\n"
             "fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * d.predictor.nunique() + 1.6))\n"
             "for ax, m, t in zip(axes, ['fnr', 'fpr'], ['False-negative rate', 'False-positive rate']):\n"
             "    w = d.pivot_table(index='predictor', columns='class', values=m).reindex(columns=[c for c in CLASS_ORDER if c in set(d['class'])])\n"
             "    h = 0.8 / len(w.columns)\n"
             "    for j, c in enumerate(w.columns):\n"
             "        ax.barh(np.arange(len(w)) + j * h, w[c], height=h, color=CLASS_COLOR[c], label=CLASS_LABEL[c])\n"
             "    ax.set_yticks(np.arange(len(w)) + 0.4 - h / 2, [P.label(p) for p in w.index]); ax.invert_yaxis(); ax.set_xlabel(t)\n"
             "axes[0].legend(fontsize=7); fig.tight_layout(); P.save(fig, '2_3_error_rates')\n"
             "display(Markdown(R.threshold_takeaway(thr)))\nam = A.alphamissense_zones(lab_all[lab_all.protein_class.notna()])\nam.round(3)"),
    ("md", "### 2.4 Calibration\nEach predictor is mapped to a probability of pathogenicity with a logistic fit on soluble variants. "
           "ClinVar's pathogenic fraction differs between classes for reasons of ascertainment, so probabilities are shifted to each "
           "class's own pathogenic fraction before checking them. If the map then transfers, multipass variants lie on the diagonal; "
           "remaining deviations mean the same score carries different information in membrane proteins."),
    ("code", "calib, curves = A.calibration(lab, PRED)\n"
             "show = [p for p in ('alphamissense', 'esm1v', 'saprot', 'thermompnn') if p in set(calib.predictor)]\n"
             "fig, axes = plt.subplots(1, max(1, len(show)), figsize=(3.4 * max(1, len(show)), 3.4), squeeze=False)\n"
             "for ax, p in zip(axes[0], show):\n"
             "    for c in ('SOLUBLE', 'MULTIPASS', 'SINGLE_PASS'):\n"
             "        r = curves[(curves.predictor == p) & (curves['class'] == c)]\n"
             "        if len(r): ax.plot(r.pred, r.obs, 'o-', color=CLASS_COLOR[c], ms=4, lw=1.6, label=CLASS_LABEL[c])\n"
             "    ax.plot([0, 1], [0, 1], color=P.AXIS, lw=0.8); ax.set_title(P.label(p), loc='left')\n"
             "    ax.set_xlabel('Predicted P(pathogenic), prevalence-adjusted'); ax.set_ylabel('Observed fraction pathogenic'); ax.legend(fontsize=7)\n"
             "fig.tight_layout(); P.save(fig, '2_4_calibration')\n"
             "display(Markdown(R.calibration_takeaway(calib)))\ncalib.round(3)"),
    ("md", "### 2.5 Do predictors agree?\nRank correlation between predictors over all substitutions, per class, and the per-variant spread of "
           "the predictors' proteome percentiles (higher = more disagreement)."),
    ("code", "corr = A.correlation_by_class(sample, PRED)\n"
             "if MEMBRANE in corr and 'SOLUBLE' in corr:\n"
             "    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))\n"
             "    P.heatmap(corr[MEMBRANE], ax=axes[0], title='Multipass: Spearman')\n"
             "    P.heatmap(corr['SOLUBLE'], ax=axes[1], title='Soluble: Spearman')\n"
             "    P.heatmap(corr[MEMBRANE] - corr['SOLUBLE'], ax=axes[2], title='Difference (multipass - soluble)', vmin=-0.5, vmax=0.5)\n"
             "    fig.tight_layout(); P.save(fig, '2_5_predictor_correlation')"),
    ("code", "dis_lab = A.disagreement(lab, sample, [p for p in PRED if p != 'solvent_exposure'])\n"
             "dis_all = A.disagreement(sample, sample, [p for p in PRED if p != 'solvent_exposure'])\n"
             "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))\n"
             "P.class_violin(dis_all, 'disagreement_sd', [c for c in CLASS_ORDER if c != 'BETA_BARREL_MEMBRANE'], ax=axes[0], ylabel='SD of predictor percentiles', title='All substitutions')\n"
             "d = dis_all[dis_all.protein_class == MEMBRANE]\n"
             "P.class_violin(d.assign(g=d.region), 'disagreement_sd', ['TM', 'Juxtamembrane', 'Non-TM'], group='g', ax=axes[1], ylabel='SD of predictor percentiles', title='Multipass, by region')\n"
             "fig.tight_layout(); P.save(fig, '2_5_disagreement')\n"
             "dis_all.groupby('protein_class').disagreement_sd.describe().round(2)"),
    ("md", "### 2.6 Variants of uncertain significance\nWhere VUS accumulate, and how predictors would call them using thresholds tuned "
           "on soluble proteins. A large VUS burden with split predictor calls marks where experimental maps would add most."),
    ("code", "vb = A.vus_burden(prot[prot.taxid == HUMAN])\ndisplay(vb.round(3))\n"
             "per_vus, agree = A.vus_reclassification(lab_all[lab_all.protein_class.notna()], PRED, VEP)\n"
             "if len(agree):\n"
             "    a = agree.set_index('class').reindex([c for c in CLASS_ORDER + ['NON_MULTIPASS'] if c in set(agree['class'])])\n"
             "    fig, ax = plt.subplots(figsize=(7, 2.8))\n"
             "    left = np.zeros(len(a))\n"
             "    for k, col in (('all pathogenic', '#e34948'), ('split', '#eda100'), ('all benign', '#2a78d6')):\n"
             "        ax.barh(range(len(a)), a[k], left=left, color=col, label=k); left += a[k].fillna(0).to_numpy()\n"
             "    ax.set_yticks(range(len(a)), [CLASS_LABEL[c] for c in a.index]); ax.invert_yaxis(); ax.set_xlabel('Fraction of VUS')\n"
             "    ax.set_title('Agreement of ' + ', '.join(P.label(v) for v in VEP) + ' on VUS', loc='left'); ax.legend(fontsize=7, ncol=3, loc='lower right')\n"
             "    fig.tight_layout(); P.save(fig, '2_6_vus_agreement')\n"
             "per_vus.pivot_table(index='predictor', columns='class', values='called_pathogenic').round(3)"),

    # -------------------------------------------------------------------------------- (iii)
    ("md", "## 3. Structural differences\n\n### 3.1 Model confidence and disorder\nAll proteins (human and yeast). "
           "`frac_excluded` is the fraction of residues removed as low-confidence or outside any TED domain."),
    ("code", "pr = prot.rename(columns={'class_annotation': 'protein_class'})\n"
             "fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))\n"
             "P.class_violin(pr, 'plddt_mean', CLASS_ORDER, ax=axes[0], ylabel='mean pLDDT')\n"
             "P.class_violin(pr, 'frac_excluded', CLASS_ORDER, ax=axes[1], ylabel='fraction excluded')\n"
             "fig.tight_layout(); P.save(fig, '3_1_confidence')"),
    ("md", "### 3.2 Shape, compactness and hydrophobicity: size-matched effects\nCliff's delta of multipass vs each class within size bins "
           "(so larger membrane proteins are only compared with proteins of similar size); +1 means always higher in multipass."),
    ("code", "sh = D['shape_units']\n"
             "con_t = A.structural_contrasts(sh, 'CHAIN_TRIMMED')\ncon_d = A.structural_contrasts(sh, 'DOMAIN')\n"
             "fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * con_t.metric.nunique() + 1.6), sharey=True)\n"
             "for ax, con, t in zip(axes, [con_t, con_d], ['Trimmed chains', 'TED domains']):\n"
             "    d = con[con['class'].isin(['SOLUBLE', 'SINGLE_PASS', 'NON_MULTIPASS'])].assign(lo=lambda x: x.cliffs_delta_size_matched, hi=lambda x: x.cliffs_delta_size_matched)\n"
             "    P.forest(d, 'metric', 'class', value='cliffs_delta_size_matched', lo='lo', hi='hi', ref=0, ax=ax, xlabel=\"Cliff's delta (multipass vs class)\", title=t)\n"
             "fig.tight_layout(); P.save(fig, '3_2_structure_effects')\n"
             "display(Markdown(R.structure_takeaway(con_t)))"),
    ("md", "### 3.3 Burial and environment of residues by location"),
    ("code", "res = D['residues']\nbl = A.burial_by_location(res)\n"
             "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))\n"
             "order = [('MULTIPASS', l) for l in ['TM lipid-facing', 'TM buried', 'Juxtamembrane', 'Extramembrane exposed', 'Extramembrane buried']] + [('SOLUBLE', 'Exposed'), ('SOLUBLE', 'Buried')]\n"
             "for ax, m, t in zip(axes, ['median_depth', 'frac_helix'], ['Median burial depth (A)', 'Fraction helix']):\n"
             "    v = [bl[(bl.protein_class == c) & (bl.location == l)][m].squeeze() if len(bl[(bl.protein_class == c) & (bl.location == l)]) else np.nan for c, l in order]\n"
             "    ax.barh(range(len(order)), v, color=[CLASS_COLOR[c] for c, _ in order]); ax.set_yticks(range(len(order)), [f'{c.title()}: {l}' for c, l in order], fontsize=8)\n"
             "    ax.invert_yaxis(); ax.set_xlabel(t)\n"
             "fig.tight_layout(); P.save(fig, '3_3_burial_by_location')\nbl.round(3)"),
    ("md", "### 3.4 Structure and prediction quality\nDoes predictive performance depend on structural context differently in membrane proteins? "
           "Pooled AUC within bins of residue burial depth and model confidence, and per-protein AUC vs protein-level structure."),
    ("code", "p0 = VEP[0] if VEP else PRED[0]\n"
             "ctx_depth = A.auc_by_context(lab, p0, 'depth_mean', [-0.1, 1, 3, 6, 100], ['surface (<1 A)', '1-3 A', '3-6 A', 'core (>6 A)'], n_boot=max(50, N_BOOT // 3))\n"
             "ctx_plddt = A.auc_by_context(lab, p0, 'plddt_model', [0, 70, 90, 101], ['<70', '70-90', '>=90'], n_boot=max(50, N_BOOT // 3))\n"
             "fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))\n"
             "for ax, d, t in zip(axes, [ctx_depth, ctx_plddt], ['by burial depth', 'by pLDDT']):\n"
             "    for j, c in enumerate([MEMBRANE, 'SOLUBLE']):\n"
             "        s = d[d['class'] == c].reset_index(drop=True); x = np.arange(len(s)) + (j - 0.5) * 0.25\n"
             "        ax.errorbar(x, s.auc, yerr=[s.auc - s.ci_low, s.ci_high - s.auc], fmt='o', color=CLASS_COLOR[c], label=CLASS_LABEL[c])\n"
             "    ax.set_xticks(range(len(s)), s['bin']); ax.axhline(0.5, color=P.AXIS); ax.set_ylabel(P.label(p0) + ' AUC'); ax.set_title(t, loc='left'); ax.legend(fontsize=7)\n"
             "fig.tight_layout(); P.save(fig, '3_4_auc_by_context')\n"
             "A.auc_vs_structure(D['protein_auc'], sh, p0, A.STRUCT_METRICS, MIN_PER_CLASS).round(3)"),

    # -------------------------------------------------------------------------------- summary
    ("md", "## 4. Summary: multipass vs soluble across all analyses"),
    ("code", "summary = R.summary_table(pooled, matched, thr, calib, dyn, con_t)\n"
             "summary.to_csv(FIG_DIR / 'summary_multipass_vs_soluble.tsv', sep='\\t', index=False)\n"
             "print('Figures and summary table written to', FIG_DIR.resolve())\nsummary.round(3)"),
]

EXPLORE = [
    ("md", "# Explorer: any predictor or structural metric, multipass vs every other class\n\n"
           "Set the parameters in the next cell and run the notebook (or use the dropdowns at the end if `ipywidgets` is installed)."),
    ("code", SETUP + "\n\n# ---------------------------------------------------------------- explorer choices\n"
             "PREDICTOR = 'alphamissense'          # any of: " + "'alphamissense', 'esm1v', 'saprot', 'thermompnn', 'foldx', 'blosum62', "
             "'abs_delta_hydrophobicity', 'abs_delta_volume', 'abs_delta_charge', 'delta_hydrophobicity', 'delta_volume', 'delta_charge', 'solvent_exposure'\n"
             "METRIC = 'a_over_c'                  # any column of shape_units (see SHAPE_METRICS below)\n"
             "UNIT = 'CHAIN_TRIMMED'               # CHAIN_FULL, CHAIN_TRIMMED, DOMAIN, TM_CORE\n"
             "CLASSES = CLASS_ORDER\n"
             "GENE = 'CFTR'                        # protein lookup\n"
             "SHAPE_METRICS = [c for c in D['shape_units'].columns if pd.api.types.is_numeric_dtype(D['shape_units'][c])]"),
    ("md", "## Predictor view"),
    ("code", "def predictor_view(p):\n"
             "    if p not in lab or lab[p].notna().sum() == 0:\n"
             "        print(p, 'has no data'); return\n"
             "    pooled = A.pooled_performance(lab, [p], n_boot=min(N_BOOT, 200))\n"
             "    loc = A.location_performance(lab, [p], n_boot=100)\n"
             "    fig, axes = plt.subplots(2, 3, figsize=(16, 8))\n"
             "    P.forest(pooled[pooled['class'].isin(CLASSES)], 'class', 'class', ax=axes[0, 0], groups=[c for c in CLASSES], title='Pooled AUC')\n"
             "    for c, (fpr, tpr, a) in A.roc_curves(lab, p, CLASSES).items():\n"
             "        axes[0, 1].plot(fpr, tpr, color=CLASS_COLOR[c], label=f'{CLASS_LABEL[c]} {a:.2f}')\n"
             "    axes[0, 1].plot([0, 1], [0, 1], color=P.AXIS); axes[0, 1].legend(fontsize=7); axes[0, 1].set_title('ROC', loc='left')\n"
             "    pp, _ = A.per_protein_performance(D['protein_auc'], [p], MIN_PER_CLASS)\n"
             "    P.class_violin(pp, 'auc', CLASSES, ax=axes[0, 2], ylabel='per-protein AUC', ref=0.5, title='Per protein')\n"
             "    P.ecdf(sample.assign(v=sample[p] * DIRECTION[p]), 'v', CLASSES, ax=axes[1, 0], xlabel='oriented score (all substitutions)')\n"
             "    l = loc.reset_index(drop=True)\n"
             "    axes[1, 1].errorbar(l.auc, range(len(l)), xerr=[l.auc - l.ci_low, l.ci_high - l.auc], fmt='o', color=P.INK2)\n"
             "    axes[1, 1].set_yticks(range(len(l)), l['class'].str.title() + ': ' + l.location, fontsize=7); axes[1, 1].axvline(0.5, color=P.AXIS)\n"
             "    axes[1, 1].invert_yaxis(); axes[1, 1].set_title('AUC by location', loc='left')\n"
             "    calib, curves = A.calibration(lab, [p])\n"
             "    for c in CLASSES:\n"
             "        r = curves[curves['class'] == c] if len(curves) else curves\n"
             "        if len(r): axes[1, 2].plot(r.pred, r.obs, 'o-', color=CLASS_COLOR[c], ms=3, label=CLASS_LABEL[c])\n"
             "    axes[1, 2].plot([0, 1], [0, 1], color=P.AXIS); axes[1, 2].set_title('Calibration (fitted on soluble)', loc='left'); axes[1, 2].legend(fontsize=7)\n"
             "    fig.suptitle(P.label(p), x=0.01, ha='left'); fig.tight_layout(); plt.show()\n"
             "    display(pooled[['class', 'n_proteins', 'n_pathogenic', 'n_benign', 'auc', 'ci_low', 'ci_high', 'mp_minus_delta', 'mp_minus_q']].round(3))\n\n"
             "predictor_view(PREDICTOR)"),
    ("md", "## Structural metric view"),
    ("code", "def metric_view(metric, unit):\n"
             "    sh = D['shape_units']; d = sh[sh.unit == unit]\n"
             "    if metric not in d: print(metric, 'not available'); return\n"
             "    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))\n"
             "    P.class_violin(d, metric, CLASSES, ax=axes[0], title=f'{metric} ({unit})')\n"
             "    for c in CLASSES:\n"
             "        s = d[d.protein_class == c]\n"
             "        axes[1].scatter(s.n_res, s[metric], s=4, alpha=0.4, color=CLASS_COLOR[c], label=CLASS_LABEL[c])\n"
             "    axes[1].set_xscale('log'); axes[1].set_xlabel('residues'); axes[1].set_ylabel(metric); axes[1].legend(fontsize=7, markerscale=3)\n"
             "    fig.tight_layout(); plt.show()\n"
             "    display(A.structural_contrasts(sh, unit, [metric]).round(3))\n\n"
             "metric_view(METRIC, UNIT)"),
    ("md", "## Protein lookup"),
    ("code", "def protein_view(gene):\n"
             "    hit = prot[prot.gene.str.upper() == gene.upper()]\n"
             "    if hit.empty: print('no protein with gene', gene); return\n"
             "    acc = hit.accession.iat[0]\n"
             "    display(hit.T)\n"
             "    a = D['protein_auc']; display(a[a.accession == acc][['predictor', 'n_pathogenic', 'n_benign', 'auc', 'ci_low', 'ci_high']].round(3))\n"
             "    v = lab_all[lab_all.accession == acc]\n"
             "    if len(v):\n"
             "        p = PREDICTOR if PREDICTOR in v else PRED[0]\n"
             "        fig, ax = plt.subplots(figsize=(10, 3))\n"
             "        for cl, col in (('pathogenic', '#e34948'), ('likely_pathogenic', '#e87ba4'), ('vus', '#8f8d85'), ('likely_benign', '#1baf7a'), ('benign', '#2a78d6')):\n"
             "            s = v[v.clinvar_class == cl]; ax.scatter(s.position, s[p] * DIRECTION[p], s=8, color=col, label=cl)\n"
             "        for r in D['residues'][(D['residues'].accession == acc) & D['residues'].in_tm].itertuples():\n"
             "            ax.axvspan(r.position - 0.5, r.position + 0.5, color='#f3e6dc', lw=0, zorder=0)\n"
             "        ax.set_xlabel('residue (shaded: TM segments)'); ax.set_ylabel(P.label(p) + ' (oriented)'); ax.legend(fontsize=7, ncol=5)\n"
             "        ax.set_title(f'{gene} ({acc})', loc='left'); plt.show()\n\n"
             "protein_view(GENE)"),
    ("md", "## Interactive (optional)\nWith `ipywidgets` installed (`pip install ipywidgets`), dropdowns re-draw the views."),
    ("code", "try:\n"
             "    import ipywidgets as W\n"
             "    W.interact(predictor_view, p=W.Dropdown(options=PRED, value=PREDICTOR))\n"
             "    W.interact(metric_view, metric=W.Dropdown(options=SHAPE_METRICS, value=METRIC), unit=W.Dropdown(options=['CHAIN_FULL', 'CHAIN_TRIMMED', 'DOMAIN', 'TM_CORE'], value=UNIT))\n"
             "    W.interact(protein_view, gene=W.Text(value=GENE))\n"
             "except ImportError:\n"
             "    print('ipywidgets not installed: edit the parameters above and re-run instead')"),
]


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, path)
    print("wrote", path)


root = Path(__file__).resolve().parents[1] / "notebooks"
build(NARRATIVE, root / "membrane_vs_soluble.ipynb")
build(EXPLORE, root / "explore.ipynb")
