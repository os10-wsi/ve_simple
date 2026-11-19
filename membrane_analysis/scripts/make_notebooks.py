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
from mvsa.data import load, available_predictors, dataset_label
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
TAXID = D["meta"].get("taxid") or HUMAN
ORGANISM = {9606: "human", 559292: "yeast"}.get(TAXID, str(TAXID))
HAS_LABELS = len(lab) >= 20 and lab.y.nunique() == 2      # yeast has no ClinVar: clinical sections are skipped
SKIP = "No ClinVar pathogenic/benign labels in this dataset (" + ORGANISM + "): section skipped."
PRED = [p for p in MAIN_PREDICTORS if p in available_predictors(lab if HAS_LABELS else sample)]
pooled = matched = thr = calib = pd.DataFrame()           # filled by the clinical sections when labels exist
VEP = [p for p in ("alphamissense", "esm1v", "saprot") if p in PRED]
print("data:", D["meta"].get("created", "?"), "| organism:", ORGANISM, "| predictors with data:", ", ".join(PRED))
if not HAS_LABELS:
    display(Markdown(f"**{SKIP.replace('section skipped', 'clinical-performance, threshold, calibration and VUS sections are skipped')}** "
                     "Score distributions, dynamic range, predictor agreement and structure are shown."))
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
    ("code", "ov = A.overview(prot, lab_all)\nov"),
    ("code", "src, what = (lab, 'labelled variants') if HAS_LABELS else (sample, 'sampled substitutions')\n"
             "cov = pd.DataFrame({c: [src.loc[src.protein_class == c, p].notna().mean() for p in PRED] for c in CLASS_ORDER}, index=PRED)\n"
             "print(f'Fraction of {what} with a score, by class:')\ncov.round(2)"),

    # -------------------------------------------------------------------------------- (i)
    ("md", "## 1. Clinical performance: pathogenic vs benign\n\n"
           "### 1.1 Pooled across proteins\nAUC over every labelled variant of each class, with protein-bootstrap 95% CIs. "
           "The right panel shows multipass minus each comparator; values below zero mean predictors do worse on multipass proteins."),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    pooled = A.pooled_performance(lab, PRED, n_boot=N_BOOT)\n    fig, ax = plt.subplots(1, 2, figsize=(12, 0.45 * len(PRED) + 1.6), gridspec_kw={'width_ratios': [1.2, 1]})\n    P.forest(pooled[pooled['class'].isin(CLASS_ORDER)], 'predictor', 'class', ax=ax[0], groups=CLASS_ORDER,\n             title='AUC by protein class')\n    d = pooled[pooled['class'].isin(['SOLUBLE', 'NON_MULTIPASS'])].rename(columns={'mp_minus_delta': 'dd', 'mp_minus_delta_low': 'dl', 'mp_minus_delta_high': 'dh'})\n    P.forest(d, 'predictor', 'class', value='dd', lo='dl', hi='dh', ref=0, ax=ax[1], xlabel='AUC difference (multipass - comparator)',\n             title='Multipass minus comparator')\n    ax[1].set_yticklabels([])\n    fig.tight_layout(); P.save(fig, '1_1_pooled_auc')"),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    cols = ['predictor', 'class', 'n_proteins', 'n_pathogenic', 'n_benign', 'auc', 'ci_low', 'ci_high', 'mp_minus_delta', 'mp_minus_delta_low', 'mp_minus_delta_high', 'mp_minus_q']\n    display(pooled[cols].round(3))"),
    ("md", "### 1.2 ROC curves"),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    show = [p for p in ('alphamissense', 'esm1v', 'thermompnn', 'blosum62') if p in PRED]\n    fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.4), squeeze=False)\n    for ax, p in zip(axes[0], show):\n        for c, (fpr, tpr, a) in A.roc_curves(lab, p, ['MULTIPASS', 'SOLUBLE', 'SINGLE_PASS', 'AMBIGUOUS']).items():\n            ax.plot(fpr, tpr, color=CLASS_COLOR[c], lw=2 if c == MEMBRANE else 1.3, label=f'{CLASS_LABEL[c]} {a:.2f}')\n        ax.plot([0, 1], [0, 1], color=P.AXIS, lw=0.8); ax.set_title(P.label(p), loc='left')\n        ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(fontsize=7, loc='lower right')\n    fig.tight_layout(); P.save(fig, '1_2_roc')"),
    ("md", "### 1.3 Per protein\nAUC inside each protein (from the ve_simple tables), for proteins with at least `MIN_PER_CLASS` pathogenic "
           "and benign variants. This removes any between-gene component but keeps fewer proteins."),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    pp, pp_comp = A.per_protein_performance(D['protein_auc'], PRED, MIN_PER_CLASS)\n    show = [p for p in PRED if p in set(pp.predictor)][:6]\n    fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.4), squeeze=False)\n    for ax, p in zip(axes[0], show):\n        P.class_violin(pp[pp.predictor == p], 'auc', [c for c in CLASS_ORDER if c != 'BETA_BARREL_MEMBRANE'], ax=ax, ylabel='per-protein AUC', title=P.label(p), ref=0.5)\n    fig.tight_layout(); P.save(fig, '1_3_per_protein_auc')\n    display(pp_comp.round(3))"),
    ("md", "### 1.4 Where in the protein\nInside multipass proteins, variants are split by location: transmembrane segment facing the lipid "
           "(exposed in the AlphaFold model) or buried, juxtamembrane (within 5 residues of a TM segment), and extramembrane buried or exposed. "
           "Soluble proteins are split into buried and exposed residues for reference (relative SASA threshold 0.2)."),
    ("code", 'if not HAS_LABELS:\n    print(SKIP)\nelse:\n    loc = A.location_performance(lab, PRED, n_boot=max(100, N_BOOT // 2))\n    loc[\'group\'] = loc[\'class\'].map({\'MULTIPASS\': \'Multipass\', \'SOLUBLE\': \'Soluble\'}) + \': \' + loc[\'location\']\n    show = [p for p in (\'alphamissense\', \'esm1v\', \'saprot\', \'thermompnn\', \'blosum62\') if p in PRED]\n    fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.6), sharey=True, squeeze=False)\n    for ax, p in zip(axes[0], show):\n        d = loc[loc.predictor == p].reset_index(drop=True)\n        y = np.arange(len(d))\n        for c in d[\'class\'].unique():\n            m = (d[\'class\'] == c).to_numpy()\n            ax.errorbar(d.auc[m], y[m], xerr=[(d.auc - d.ci_low)[m], (d.ci_high - d.auc)[m]], fmt=\'o\', ms=6, elinewidth=1.4,\n                        color=CLASS_COLOR[c], ecolor=CLASS_COLOR[c], mec=\'white\')\n        ax.axvline(0.5, color=P.AXIS, lw=1); ax.set_title(P.label(p), loc=\'left\'); ax.set_xlabel(\'AUC\')\n        ax.set_yticks(y, [f"{g} (n={a}/{b})" for g, a, b in zip(d.group, d.n_pathogenic, d.n_benign)], fontsize=7)\n    axes[0][0].invert_yaxis(); fig.tight_layout(); P.save(fig, \'1_4_auc_by_location\')\n    display(loc[[\'predictor\', \'class\', \'location\', \'n_proteins\', \'n_pathogenic\', \'n_benign\', \'auc\', \'ci_low\', \'ci_high\']].round(3))'),
    ("md", "### 1.5 Is the gap explained by easier label mixes or structural context?\n"
           "*Matched* recomputes the AUC within strata of residue pLDDT, burial and the protein's fraction of pathogenic variants, "
           "weighting both classes by the multipass strata. *Within-protein* averages AUCs computed inside each protein. "
           "If the multipass deficit persists after both, it is not an artefact of which genes or residues happen to be labelled."),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    matched = A.matched_performance(lab, [p for p in PRED if p in ('alphamissense', 'esm1v', 'saprot', 'thermompnn', 'foldx', 'blosum62')], n_boot=max(50, N_BOOT // 3))\n    d = matched.copy(); d['group'] = d.estimate\n    fig, ax = plt.subplots(figsize=(6.5, 0.5 * d.predictor.nunique() + 1.5))\n    for j, (k, col) in enumerate(zip(['raw', 'matched', 'within-protein'], ['#8f8d85', '#eb6834', '#6250d6'])):\n        s = d[d.estimate == k].reset_index(drop=True); y = np.arange(len(s)) + (j - 1) * 0.22\n        ax.errorbar(s.delta, y, xerr=[s.delta - s.delta_low, s.delta_high - s.delta], fmt='o', color=col, label=k, ms=5)\n    ax.axvline(0, color=P.AXIS); ax.set_yticks(range(s.shape[0]), [P.label(p) for p in s.predictor]); ax.invert_yaxis()\n    ax.set_xlabel('AUC difference, multipass - soluble'); ax.legend(fontsize=8); fig.tight_layout(); P.save(fig, '1_5_matched')\n    display(Markdown(R.matched_takeaway(matched)))"),
    ("md", "### Takeaway (i)"),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    display(Markdown(R.performance_takeaway(pooled, 'SOLUBLE')))\n    display(Markdown(R.performance_takeaway(pooled, 'NON_MULTIPASS')))"),

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
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    thr = A.threshold_transfer(lab, PRED)\n    d = thr[thr.threshold == 'Youden on soluble']\n    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * d.predictor.nunique() + 1.6))\n    for ax, m, t in zip(axes, ['fnr', 'fpr'], ['False-negative rate', 'False-positive rate']):\n        w = d.pivot_table(index='predictor', columns='class', values=m).reindex(columns=[c for c in CLASS_ORDER if c in set(d['class'])])\n        h = 0.8 / len(w.columns)\n        for j, c in enumerate(w.columns):\n            ax.barh(np.arange(len(w)) + j * h, w[c], height=h, color=CLASS_COLOR[c], label=CLASS_LABEL[c])\n        ax.set_yticks(np.arange(len(w)) + 0.4 - h / 2, [P.label(p) for p in w.index]); ax.invert_yaxis(); ax.set_xlabel(t)\n    axes[0].legend(fontsize=7); fig.tight_layout(); P.save(fig, '2_3_error_rates')\n    display(Markdown(R.threshold_takeaway(thr)))\n    am = A.alphamissense_zones(lab_all[lab_all.protein_class.notna()])\n    display(am.round(3))"),
    ("md", "### 2.4 Calibration\nEach predictor is mapped to a probability of pathogenicity with a logistic fit on soluble variants. "
           "ClinVar's pathogenic fraction differs between classes for reasons of ascertainment, so probabilities are shifted to each "
           "class's own pathogenic fraction before checking them. If the map then transfers, multipass variants lie on the diagonal; "
           "remaining deviations mean the same score carries different information in membrane proteins."),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    calib, curves = A.calibration(lab, PRED)\n    show = [p for p in ('alphamissense', 'esm1v', 'saprot', 'thermompnn') if p in set(calib.predictor)]\n    fig, axes = plt.subplots(1, max(1, len(show)), figsize=(3.4 * max(1, len(show)), 3.4), squeeze=False)\n    for ax, p in zip(axes[0], show):\n        for c in ('SOLUBLE', 'MULTIPASS', 'SINGLE_PASS'):\n            r = curves[(curves.predictor == p) & (curves['class'] == c)]\n            if len(r): ax.plot(r.pred, r.obs, 'o-', color=CLASS_COLOR[c], ms=4, lw=1.6, label=CLASS_LABEL[c])\n        ax.plot([0, 1], [0, 1], color=P.AXIS, lw=0.8); ax.set_title(P.label(p), loc='left')\n        ax.set_xlabel('Predicted P(pathogenic), prevalence-adjusted'); ax.set_ylabel('Observed fraction pathogenic'); ax.legend(fontsize=7)\n    fig.tight_layout(); P.save(fig, '2_4_calibration')\n    display(Markdown(R.calibration_takeaway(calib)))\n    display(calib.round(3))"),
    ("md", "### 2.5 Do predictors agree?\nRank correlation between predictors over all substitutions, per class, and the per-variant spread of "
           "the predictors' proteome percentiles (higher = more disagreement)."),
    ("code", "corr = A.correlation_by_class(sample, PRED)\n"
             "if MEMBRANE in corr and 'SOLUBLE' in corr:\n"
             "    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))\n"
             "    P.heatmap(corr[MEMBRANE], ax=axes[0], title='Multipass: Spearman')\n"
             "    P.heatmap(corr['SOLUBLE'], ax=axes[1], title='Soluble: Spearman')\n"
             "    P.heatmap(corr[MEMBRANE] - corr['SOLUBLE'], ax=axes[2], title='Difference (multipass - soluble)', vmin=-0.5, vmax=0.5)\n"
             "    fig.tight_layout(); P.save(fig, '2_5_predictor_correlation')"),
    ("code", "dis_all = A.disagreement(sample, sample, [p for p in PRED if p != 'solvent_exposure'])\n"
             "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))\n"
             "P.class_violin(dis_all, 'disagreement_sd', [c for c in CLASS_ORDER if c != 'BETA_BARREL_MEMBRANE'], ax=axes[0], ylabel='SD of predictor percentiles', title='All substitutions')\n"
             "d = dis_all[dis_all.protein_class == MEMBRANE]\n"
             "P.class_violin(d.assign(g=d.region), 'disagreement_sd', ['TM', 'Juxtamembrane', 'Non-TM'], group='g', ax=axes[1], ylabel='SD of predictor percentiles', title='Multipass, by region')\n"
             "fig.tight_layout(); P.save(fig, '2_5_disagreement')\n"
             "dis_all.groupby('protein_class').disagreement_sd.describe().round(2)"),
    ("md", "### 2.6 Variants of uncertain significance\nWhere VUS accumulate, and how predictors would call them using thresholds tuned "
           "on soluble proteins. A large VUS burden with split predictor calls marks where experimental maps would add most."),
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    vb = A.vus_burden(prot)\n    display(vb.round(3))\n    per_vus, agree = A.vus_reclassification(lab_all[lab_all.protein_class.notna()], PRED, VEP)\n    if len(agree):\n        a = agree.set_index('class').reindex([c for c in CLASS_ORDER + ['NON_MULTIPASS'] if c in set(agree['class'])])\n        fig, ax = plt.subplots(figsize=(7, 2.8))\n        left = np.zeros(len(a))\n        for k, col in (('all pathogenic', '#e34948'), ('split', '#eda100'), ('all benign', '#2a78d6')):\n            ax.barh(range(len(a)), a[k], left=left, color=col, label=k); left += a[k].fillna(0).to_numpy()\n        ax.set_yticks(range(len(a)), [CLASS_LABEL[c] for c in a.index]); ax.invert_yaxis(); ax.set_xlabel('Fraction of VUS')\n        ax.set_title('Agreement of ' + ', '.join(P.label(v) for v in VEP) + ' on VUS', loc='left'); ax.legend(fontsize=7, ncol=3, loc='lower right')\n        fig.tight_layout(); P.save(fig, '2_6_vus_agreement')\n    display(per_vus.pivot_table(index='predictor', columns='class', values='called_pathogenic').round(3))"),

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
    ("code", "if not HAS_LABELS:\n    print(SKIP)\nelse:\n    p0 = VEP[0] if VEP else PRED[0]\n    ctx_depth = A.auc_by_context(lab, p0, 'depth_mean', [-0.1, 1, 3, 6, 100], ['surface (<1 A)', '1-3 A', '3-6 A', 'core (>6 A)'], n_boot=max(50, N_BOOT // 3))\n    ctx_plddt = A.auc_by_context(lab, p0, 'plddt_model', [0, 70, 90, 101], ['<70', '70-90', '>=90'], n_boot=max(50, N_BOOT // 3))\n    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))\n    for ax, d, t in zip(axes, [ctx_depth, ctx_plddt], ['by burial depth', 'by pLDDT']):\n        for j, c in enumerate([MEMBRANE, 'SOLUBLE']):\n            s = d[d['class'] == c].reset_index(drop=True); x = np.arange(len(s)) + (j - 0.5) * 0.25\n            ax.errorbar(x, s.auc, yerr=[s.auc - s.ci_low, s.ci_high - s.auc], fmt='o', color=CLASS_COLOR[c], label=CLASS_LABEL[c])\n        ax.set_xticks(range(len(s)), s['bin']); ax.axhline(0.5, color=P.AXIS); ax.set_ylabel(P.label(p0) + ' AUC'); ax.set_title(t, loc='left'); ax.legend(fontsize=7)\n    fig.tight_layout(); P.save(fig, '3_4_auc_by_context')\n    display(A.auc_vs_structure(D['protein_auc'], sh, p0, A.STRUCT_METRICS, MIN_PER_CLASS).round(3))"),

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
             "    if p not in sample or sample[p].notna().sum() == 0:\n"
             "        print(p, 'has no data'); return\n"
             "    if not HAS_LABELS:\n"
             "        print(SKIP.replace('section skipped', 'showing score distributions only'))\n"
             "        fig, ax = plt.subplots(figsize=(6, 4))\n"
             "        P.ecdf(sample.assign(v=sample[p] * DIRECTION[p]), 'v', CLASSES, ax=ax, xlabel=P.label(p) + ' (oriented, all substitutions)')\n"
             "        plt.show(); return\n"
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
             "predictor_view(PREDICTOR if PREDICTOR in PRED else PRED[0])"),
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
             "    W.interact(predictor_view, p=W.Dropdown(options=PRED, value=PREDICTOR if PREDICTOR in PRED else PRED[0]))\n"
             "    W.interact(metric_view, metric=W.Dropdown(options=SHAPE_METRICS, value=METRIC), unit=W.Dropdown(options=['CHAIN_FULL', 'CHAIN_TRIMMED', 'DOMAIN', 'TM_CORE'], value=UNIT))\n"
             "    W.interact(protein_view, gene=W.Text(value=GENE))\n"
             "except ImportError:\n"
             "    print('ipywidgets not installed: edit the parameters above and re-run instead')"),
]


TALK_SETUP = r"""import os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Markdown, display
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)
from mvsa import analyses as A, stats as S, talk as T, thermo as Th
from mvsa.data import load, available_predictors, dataset_label
from mvsa.groups import MEMBRANE, MAIN_PREDICTORS, DIRECTION

DATA_DIR = Path(os.environ.get("MVSA_DATA", "../data_human"))
FIG_DIR = Path(os.environ.get("MVSA_FIGURES", "../figures_talk"))
N_BOOT = int(os.environ.get("MVSA_NBOOT", 300))
N_BOTTOM = int(os.environ.get("MVSA_NBOTTOM", 100))   # size of the 'worst proteins' lists
MIN_PER_CLASS = 5                                      # per-protein AUC needs this many P and B variants
GOOD_AUC = 0.85                                        # 'ranked well' threshold used in the ThermoMPNN headline

T.style(FIG_DIR)
D = load(DATA_DIR)
TAXID = D["meta"].get("taxid") or 9606
ORGANISM = dataset_label(D["meta"])
STRUCT_SETS = [x for x in os.environ.get("MVSA_STRUCT_SETS", "human,yeast").split(",") if x]   # datasets on the structure figure
lab_all = D["labelled_variants"]
lab = S.labels(lab_all); lab = lab[lab.protein_class.notna()]
sample = D["variant_sample"]
HAS_LABELS = len(lab) >= 20 and lab.y.nunique() == 2
PRED = [p for p in MAIN_PREDICTORS if p in available_predictors(lab if HAS_LABELS else sample)]
VEP = [p for p in ("alphamissense", "esm1v", "saprot") if p in PRED]
PCORR = pd.read_parquet(DATA_DIR / "protein_correlations.parquet") if (DATA_DIR / "protein_correlations.parquet").exists() else None
TABLE_DIR = FIG_DIR / "tables"; TABLE_DIR.mkdir(parents=True, exist_ok=True)

def note(*lines):
    display(Markdown("**Slide notes**\n\n" + "\n".join(f"- {l}" for l in lines)))

def fmt_q(q):
    return "q < 0.001" if q < 0.001 else f"q = {q:.2g}"

print(f"{ORGANISM}: {len(PRED)} predictors | labels: {HAS_LABELS} | per-protein correlations: {PCORR is not None}")
if PCORR is None:
    print("protein_correlations.parquet missing: run lsf/protcorr.sh first; agreement analyses are skipped")"""

TALK = [
    ("md", "# Membrane proteins and variant-effect predictors: talk figures\n\n"
           "One message per figure, written for slides. Every headline is computed from the numbers below it, so it "
           "always says what this dataset shows. PNG and PDF files go to `figures_talk_<name>/`, tables to "
           "`figures_talk_<name>/tables/`. Under each figure, *slide notes* give the numbers to quote.\n\n"
           "Sections: 1 structure · 2 clinical performance · 3 where predictors fail · 4 per-protein failures · "
           "5 how predictors relate · 6 calibration · 7 VUS · 8 ThermoMPNN's worst proteins (clinical) · "
           "9 ThermoMPNN's least-agreeing proteins · 10 overlap of the two lists."),
    ("code", TALK_SETUP),

    ("md", "## 1. Membrane proteins are built differently"),
    ("code", "def shape_units_for(d):\n"
             "    meta = json.loads((d / 'meta.json').read_text())\n"
             "    pr = pd.read_parquet(d / 'proteins.parquet', columns=['accession', 'taxid', 'class_annotation', 'in_ve_results'])\n"
             "    pr = pr[pr.in_ve_results.astype(bool)] if pr.in_ve_results.any() else pr[pr.taxid == meta['taxid']]\n"
             "    su = pd.read_parquet(d / 'shape_units.parquet')\n"
             "    su = su[su.accession.isin(pr.accession)].copy()\n"
             "    su['protein_class'] = su.accession.map(pr.set_index('accession').class_annotation)\n"
             "    return su\n\n"
             "orgs = {o: DATA_DIR.parent / f'data_{o}' for o in STRUCT_SETS}\n"
             "orgs = {o: d for o, d in orgs.items() if (d / 'meta.json').exists()} or {ORGANISM: DATA_DIR}\n"
             "contr = {o: A.structural_contrasts(shape_units_for(d)) for o, d in orgs.items()}\n"
             "fig = T.structure_dots(contr)\n"
             "c0 = next(iter(contr.values())); c0 = c0[c0['class'] == 'SOLUBLE'].set_index('metric')\n"
             "big = c0.reindex(list(T.STRUCT_SHOW)).cliffs_delta_size_matched.abs().dropna()\n"
             "n_big = int((big >= 0.5).sum())\n"
             "T.headline(fig, 'Membrane proteins are built differently' + (' — in ' + ', '.join(contr) if len(contr) > 1 else ''),\n"
             "           f'{n_big} of {len(big)} structural features differ strongly (|δ| ≥ 0.5) from size-matched soluble proteins')\n"
             "T.save(fig, '01_structure'); plt.show()\n"
             "lines = []\n"
             "for m in big.sort_values(ascending=False).index[:5]:\n"
             "    vals = ', '.join(f\"{o} {c[(c['class']=='SOLUBLE') & (c.metric==m)].cliffs_delta_size_matched.iat[0]:+.2f}\" for o, c in contr.items())\n"
             "    lines.append(f'{T.STRUCT_SHOW[m]}: δ {vals}')\n"
             "note(*lines, 'δ = Cliff\\'s delta within size bins; +1 means every multipass protein is higher.')"),

    ("md", "## 2. Clinical performance: no overall deficit"),
    ("code", "if HAS_LABELS:\n"
             "    pooled = A.pooled_performance(lab, PRED, n_boot=N_BOOT)\n"
             "    so = pooled[pooled['class'] == 'SOLUBLE']\n"
             "    worse = so[(so.mp_minus_delta_high < 0) & (so.mp_minus_q < 0.05)]\n"
             "    better = so[(so.mp_minus_delta_low > 0) & (so.mp_minus_q < 0.05)]\n"
             "    fig = T.auc_dumbbell(pooled, PRED)\n"
             "    title = ('Clinical AUC is no lower in membrane proteins' if not len(worse)\n"
             "             else f'{len(worse)} of {len(PRED)} predictors are worse on membrane proteins')\n"
             "    mpl_ = lab[lab.protein_class == MEMBRANE]\n"
             "    T.headline(fig, title, f'{int(mpl_.y.sum()):,} pathogenic and {int((mpl_.y == 0).sum()):,} benign ClinVar variants in '\n"
             "               f'{mpl_.accession.nunique():,} multipass proteins, against soluble proteins')\n"
             "    T.save(fig, '02_clinical_auc'); plt.show()\n"
             "    best = pooled[pooled['class'] == MEMBRANE].sort_values('auc', ascending=False).iloc[0]\n"
             "    note(f'Significantly worse on multipass: {len(worse)}; significantly better: {len(better)} (BH q < 0.05).',\n"
             "         f'Best on multipass: {T.name(best.predictor)} AUC {best.auc:.3f} ({best.ci_low:.3f}–{best.ci_high:.3f}).',\n"
             "         'Physicochemical features (BLOSUM62, |Δ| volume/charge/hydrophobicity) sit at 0.54–0.65 in every class: the gap is between predictor types, not protein classes.')\n"
             "else:\n"
             "    print('No ClinVar labels for', ORGANISM)"),

    ("md", "## 3. Where predictors fail: inside vs outside the membrane"),
    ("code", "if HAS_LABELS:\n"
             "    LOCP = [p for p in VEP + ['thermompnn'] if p in PRED]\n"
             "    loc = A.location_performance(lab, LOCP, n_boot=min(N_BOOT, 200))\n"
             "    fig = T.location_dots(loc, LOCP)\n"
             "    g = lambda p, c, l: loc[(loc.predictor == p) & (loc['class'] == c) & (loc.location == l)].auc.mean()\n"
             "    gap_ext = np.nanmean([g(p, MEMBRANE, 'Extramembrane exposed') - g(p, 'SOLUBLE', 'Exposed') for p in VEP] +\n"
             "                         [g(p, MEMBRANE, 'Extramembrane buried') - g(p, 'SOLUBLE', 'Buried') for p in VEP])\n"
             "    gap_tm = np.nanmean([g(p, MEMBRANE, 'TM lipid-facing') - g(p, 'SOLUBLE', 'Exposed') for p in VEP])\n"
             "    if gap_tm > 0.01 and gap_ext < -0.01:\n"
             "        title = 'Predictors excel in TM helices but lose ground outside the membrane'\n"
             "    elif gap_ext < -0.01:\n"
             "        title = 'Membrane proteins are predicted less well outside the membrane'\n"
             "    else:\n"
             "        title = 'Prediction quality by location inside membrane proteins'\n"
             "    T.headline(fig, title, f'Sequence predictors: TM lipid-facing {gap_tm:+.3f}, extramembrane {gap_ext:+.3f} AUC vs soluble residues of the same exposure')\n"
             "    T.save(fig, '03_location'); plt.show()\n"
             "    loc.to_csv(TABLE_DIR / 'auc_by_location.tsv', sep='\\t', index=False)\n"
             "    note(*[f\"{T.name(p)}: TM lipid-facing {g(p, MEMBRANE, 'TM lipid-facing'):.3f} · extramembrane exposed {g(p, MEMBRANE, 'Extramembrane exposed'):.3f} \"\n"
             "           f\"vs soluble exposed {g(p, 'SOLUBLE', 'Exposed'):.3f} · extramembrane buried {g(p, MEMBRANE, 'Extramembrane buried'):.3f} vs soluble buried {g(p, 'SOLUBLE', 'Buried'):.3f}\" for p in LOCP],\n"
             "         'Check the CIs in the figure before quoting a gap: differences of 0.02-0.04 need overlapping intervals checked.')"),

    ("md", "## 4. Per-protein failures: how often does a predictor get a protein wrong?"),
    ("code", "if HAS_LABELS:\n"
             "    pp, ppc = A.per_protein_performance(D['protein_auc'], PRED, MIN_PER_CLASS)\n"
             "    fig, fp = T.frac_poor(pp, PRED, 0.7)\n"
             "    m = fp[(fp['class'] == MEMBRANE) & fp.predictor.isin(VEP + ['thermompnn'])].sort_values('frac', ascending=False).iloc[0]\n"
             "    so_ = fp[(fp['class'] == 'SOLUBLE') & (fp.predictor == m.predictor)].frac.iat[0]\n"
             "    T.headline(fig, f'{T.name(m.predictor)} ranks {m.frac:.0%} of membrane proteins poorly',\n"
             "               f'Per-protein AUC below 0.7 ({so_:.0%} of soluble proteins); proteins with ≥ {MIN_PER_CLASS} pathogenic and benign variants')\n"
             "    T.save(fig, '04_per_protein_failures'); plt.show()\n"
             "    note(*[f\"{T.name(r.predictor)}: {r.frac:.0%} of {r.n} multipass vs {fp[(fp['class']=='SOLUBLE') & (fp.predictor==r.predictor)].frac.iat[0]:.0%} soluble\"\n"
             "           for r in fp[fp['class'] == MEMBRANE].sort_values('frac', ascending=False).itertuples()])"),

    ("md", "## 5. How predictors relate to each other"),
    ("code", "corr = A.correlation_by_class(sample, PRED)\n"
             "if MEMBRANE in corr and 'SOLUBLE' in corr:\n"
             "    fig, diff = T.corr_panels(corr, PRED)\n"
             "    (a_, b_), d_ = diff.index[0], diff.iloc[0]\n"
             "    tm = diff[[('thermompnn' in k) for k in diff.index]]\n"
             "    T.headline(fig, f\"{T.name(b_)} and {T.name(a_)} agree {'more' if d_ > 0 else 'less'} in membrane proteins (Δρ = {d_:+.2f})\",\n"
             "               'Spearman ρ between predictors over a 1% sample of all possible substitutions; scores oriented so higher = more damaging')\n"
             "    T.save(fig, '05_predictor_correlations'); plt.show()\n"
             "    pd.concat({c: m for c, m in corr.items() if c in (MEMBRANE, 'SOLUBLE')}).to_csv(TABLE_DIR / 'predictor_correlations.tsv', sep='\\t')\n"
             "    note(*[f'{T.name(b)} – {T.name(a)}: Δρ {v:+.2f}' for (a, b), v in diff.head(5).items()],\n"
             "         *[f'ThermoMPNN – {T.name(b if a == \"thermompnn\" else a)}: ρ {corr[MEMBRANE].loc[a, b]:.2f} multipass vs {corr[\"SOLUBLE\"].loc[a, b]:.2f} soluble' for (a, b), v in tm.head(4).items()])"),

    ("md", "## 6. Scores do not transfer: calibration"),
    ("code", "if HAS_LABELS:\n"
             "    calib, _ = A.calibration(lab, PRED)\n"
             "    fig, w = T.calibration_bars(calib, PRED)\n"
             "    k = int((w.ratio > 1).sum())\n"
             "    T.headline(fig, f'A score means something different on membrane proteins',\n"
             "               f'Calibration error is higher on multipass proteins for {k} of {len(w)} predictors, after adjusting for ClinVar prevalence')\n"
             "    T.save(fig, '06_calibration'); plt.show()\n"
             "    note(*[f'{T.name(p)}: {r[MEMBRANE]:.3f} vs {r[\"SOLUBLE\"]:.3f} ({r.ratio:.1f}×)' for p, r in w.sort_values('ratio', ascending=False).iterrows()],\n"
             "         'No CIs on these yet: quote the consistent direction, not the size.')"),

    ("md", "## 7. Variants of uncertain significance"),
    ("code", "if HAS_LABELS and len(VEP) >= 2:\n"
             "    per_vus, agree = A.vus_reclassification(lab_all[lab_all.protein_class.notna()], PRED, VEP)\n"
             "    fig = T.vus_bars(agree, ', '.join(T.name(v) for v in VEP))\n"
             "    a = agree.set_index('class')\n"
             "    n_vus = int((lab_all[lab_all.protein_class == MEMBRANE].clinvar_class == 'vus').sum())\n"
             "    T.headline(fig, f\"{a.loc[MEMBRANE, 'all pathogenic']:.0%} of membrane-protein VUS look pathogenic to every predictor\",\n"
             "               f\"vs {a.loc['SOLUBLE', 'all pathogenic']:.0%} for soluble proteins; {n_vus:,} VUS in multipass proteins\")\n"
             "    T.save(fig, '07_vus'); plt.show()\n"
             "    pv = per_vus.pivot_table(index='predictor', columns='class', values='called_pathogenic')\n"
             "    note(*[f'{T.name(p)} calls {r[MEMBRANE]:.0%} of multipass VUS pathogenic vs {r[\"SOLUBLE\"]:.0%} soluble' for p, r in pv.iterrows() if p in VEP + ['thermompnn']])"),

    ("md", "## 8. ThermoMPNN's worst membrane proteins (clinical AUC)\n\n"
           "Multipass proteins ranked by ThermoMPNN's own ClinVar AUC; the bottom `N_BOTTOM` are compared with the other "
           "multipass proteins that have enough labels. Is the protein hard for every predictor, or only for ThermoMPNN? "
           "What does ThermoMPNN track in these proteins, and where are their pathogenic variants?"),
    ("code", "if HAS_LABELS:\n"
             "    tt = Th.protein_table(D, lab, PCORR, MIN_PER_CLASS)\n"
             "    scored = tt.dropna(subset=['auc_thermompnn'])\n"
             "    n_b = min(N_BOTTOM, len(scored) // 3)   # keep at least twice as many proteins to compare against\n"
             "    bot = Th.bottom(scored, 'auc_thermompnn', n_b)\n"
             "    flag = scored.accession.isin(bot.accession)\n"
             "    print(f'{len(scored)} multipass proteins have a ThermoMPNN AUC; bottom {len(bot)}: AUC {bot.auc_thermompnn.min():.2f}–{bot.auc_thermompnn.max():.2f}')\n"
             "    cands = [f'auc_{p}' for p in ('esm1v', 'alphamissense', 'saprot') if f'auc_{p}' in scored]\n"
             "    other = max(cands, key=lambda c: scored[c].notna().sum())   # the sequence predictor with most per-protein AUCs\n"
             "    fig = T.thermo_scatter(scored, flag, other)\n"
             "    good = (bot[other] >= GOOD_AUC).mean()\n"
             "    T.headline(fig, f'{good:.0%} of ThermoMPNN\\'s worst membrane proteins are ranked well by {T.name(other[4:])}',\n"
             "               f'Bottom {len(bot)} of {len(scored)} multipass proteins by ThermoMPNN AUC; well = {T.name(other[4:])} AUC ≥ {GOOD_AUC}')\n"
             "    T.save(fig, '08a_thermo_vs_vep'); plt.show()\n"
             "    auc_cols = [f'auc_{p}' for p in PRED if f'auc_{p}' in scored and p != 'thermompnn']\n"
             "    fig = T.profile(scored, flag, auc_cols, 'Per-protein AUC (median and IQR)', ref=0.5)\n"
             "    T.headline(fig, 'Is the protein hard, or only hard for ThermoMPNN?', 'Other predictors on the same proteins')\n"
             "    T.save(fig, '08b_other_predictors'); plt.show()"),
    ("code", "if HAS_LABELS and PCORR is not None:\n"
             "    rho_cols = [c for c in scored if c.startswith('rho_')]\n"
             "    fig = T.profile(scored, flag, rho_cols, 'Spearman ρ with ThermoMPNN over all substitutions of each protein (median and IQR)')\n"
             "    dm = {c: scored.loc[flag, c].median() - scored.loc[~flag, c].median() for c in rho_cols}\n"
             "    c_ = max(dm, key=lambda c: abs(dm[c]))\n"
             "    T.headline(fig, f\"In its worst proteins ThermoMPNN correlates {'more' if dm[c_] > 0 else 'less'} with {T.name(c_[4:])}\",\n"
             "               f'Median ρ {scored.loc[flag, c_].median():.2f} in the bottom {len(bot)} vs {scored.loc[~flag, c_].median():.2f} in other multipass proteins')\n"
             "    T.save(fig, '08c_thermo_correlations'); plt.show()"),
    ("code", "if HAS_LABELS:\n"
             "    feats = list(Th.FEATURE_LABEL) + [c for c in scored if c.startswith('rho_')]\n"
             "    feats = [f for f in feats if f not in ('n_P', 'n_B')]\n"
             "    cmp_clin = Th.contrast(scored, flag, feats)\n"
             "    assert len(cmp_clin), 'too few proteins to compare'\n"
             "    fig = T.effects(cmp_clin, Th.FEATURE_LABEL, top=10)\n"
             "    top = cmp_clin.iloc[0]\n"
             "    T.headline(fig, f\"Worst proteins differ most in {T.flabel(top.feature, Th.FEATURE_LABEL)}\",\n"
             "               'Features of the bottom proteins vs other multipass proteins (Cliff\\'s δ; BH-adjusted q)')\n"
             "    T.save(fig, '08d_what_differs'); plt.show()\n"
             "    fc = Th.family_counts(scored, flag)\n"
             "    fig = T.family_bars(fc)\n"
             "    f0 = fc.iloc[0]\n"
             "    T.headline(fig, f'{fc.index[0]}s make up {f0.frac_bottom:.0%} of the worst {len(bot)}',\n"
             "               f'vs {f0.frac_all:.0%} of all multipass proteins with a ThermoMPNN AUC (family from UniProt)')\n"
             "    T.save(fig, '08e_families'); plt.show()\n"
             "    cmp_clin.to_csv(TABLE_DIR / 'thermo_bottom_clinical_features.tsv', sep='\\t', index=False)\n"
             "    note(*[f\"{T.flabel(r.feature, Th.FEATURE_LABEL)}: {r.median_bottom:.3g} vs {r.median_rest:.3g} (δ {r.cliffs_delta:+.2f}, {fmt_q(r.q)})\" for r in cmp_clin.head(6).itertuples()])"),
    ("code", "SHOW = ['rank', 'gene', 'protein_name', 'family', 'length', 'n_tm_helical', 'n_P', 'n_B', 'auc_thermompnn', 'auc_esm1v',\n"
             "        'auc_saprot', 'auc_alphamissense', 'rho_esm1v', 'rho_alphamissense', 'rho_rsa', 'P_in_TM', 'P_exposed', 'thermo_P_minus_B']\n"
             "if HAS_LABELS:\n"
             "    out = bot[[c for c in SHOW if c in bot] + [c for c in bot if c not in SHOW]]\n"
             "    out.to_csv(TABLE_DIR / f'thermo_bottom{len(bot)}_clinical.tsv', sep='\\t', index=False)\n"
             "    print(f'full table: {TABLE_DIR}/thermo_bottom{len(bot)}_clinical.tsv')\n"
             "    display(out[[c for c in SHOW if c in out]].head(40).round(3).style.hide(axis='index'))"),

    ("md", "## 9. Where ThermoMPNN disagrees with the sequence predictors\n\n"
           "Every protein (no ClinVar needed): ThermoMPNN's *agreement* is the median Spearman ρ between its scores and "
           "those of ESM-1v, SaProt and (human) AlphaMissense over all substitutions. The bottom `N_BOTTOM` multipass "
           "proteins by agreement are compared with the rest."),
    ("code", "if PCORR is not None:\n"
             "    ta = Th.protein_table(D, lab if HAS_LABELS else None, PCORR, MIN_PER_CLASS, classes=(MEMBRANE, 'SOLUBLE'))\n"
             "    fig = T.agreement_hist(ta)\n"
             "    mm, ms_ = (ta[ta.class_annotation == c].agreement.median() for c in (MEMBRANE, 'SOLUBLE'))\n"
             "    T.headline(fig, f\"ThermoMPNN agrees {'less' if mm < ms_ else 'more'} with sequence predictors on membrane proteins\",\n"
             "               f'Median per-protein ρ {mm:.2f} multipass vs {ms_:.2f} soluble')\n"
             "    T.save(fig, '09a_agreement'); plt.show()\n"
             "    tmp = ta[ta.class_annotation == MEMBRANE].dropna(subset=['agreement'])\n"
             "    bota = Th.bottom(tmp, 'agreement', min(N_BOTTOM, len(tmp) // 3))\n"
             "    flag_a = tmp.accession.isin(bota.accession)\n"
             "    part = {f'rho_{p}' for p in Th.PARTNERS}\n"
             "    feats_a = [f for f in list(Th.FEATURE_LABEL) + [c for c in tmp if c.startswith('rho_') or c.startswith('auc_')]\n"
             "               if f not in part and f not in ('agreement', 'n_P', 'n_B')]\n"
             "    cmp_a = Th.contrast(tmp, flag_a, feats_a)\n"
             "    assert len(cmp_a), 'too few proteins to compare'\n"
             "    fig = T.effects(cmp_a, Th.FEATURE_LABEL, top=10)\n"
             "    top = cmp_a.iloc[0]\n"
             "    T.headline(fig, f\"Low-agreement proteins differ most in {T.flabel(top.feature, Th.FEATURE_LABEL)}\",\n"
             "               f'Bottom {len(bota)} of {len(tmp)} multipass proteins by agreement vs the rest (Cliff\\'s δ)')\n"
             "    T.save(fig, '09b_agreement_what_differs'); plt.show()\n"
             "    fca = Th.family_counts(tmp, flag_a)\n"
             "    fig = T.family_bars(fca)\n"
             "    T.headline(fig, f'{fca.index[0]}s make up {fca.iloc[0].frac_bottom:.0%} of the {len(bota)} least-agreeing proteins',\n"
             "               f'vs {fca.iloc[0].frac_all:.0%} of all multipass proteins')\n"
             "    T.save(fig, '09c_agreement_families'); plt.show()\n"
             "    cols = [c for c in ['rank', 'gene', 'protein_name', 'family', 'length', 'n_tm_helical', 'agreement', 'rho_esm1v', 'rho_saprot',\n"
             "                        'rho_alphamissense', 'rho_rsa', 'rho_depth_mean', 'rho_plddt_model', 'rho_in_tm', 'plddt_mean', 'frac_excluded', 'auc_thermompnn'] if c in bota]\n"
             "    bota[cols + [c for c in bota if c not in cols]].to_csv(TABLE_DIR / f'thermo_bottom{len(bota)}_agreement.tsv', sep='\\t', index=False)\n"
             "    cmp_a.to_csv(TABLE_DIR / 'thermo_bottom_agreement_features.tsv', sep='\\t', index=False)\n"
             "    note(*[f\"{T.flabel(r.feature, Th.FEATURE_LABEL)}: {r.median_bottom:.3g} vs {r.median_rest:.3g} (δ {r.cliffs_delta:+.2f}, {fmt_q(r.q)})\" for r in cmp_a.head(6).itertuples()])\n"
             "    display(bota[cols].head(40).round(3).style.hide(axis='index'))\n"
             "else:\n"
             "    print('run lsf/protcorr.sh first')"),

    ("md", "## 10. Overlap of the two lists"),
    ("code", "if HAS_LABELS and PCORR is not None:\n"
             "    a_set, b_set = set(bot.accession), set(bota.accession)\n"
             "    both = a_set & b_set\n"
             "    # expected overlap if unrelated: only proteins eligible for both rankings count\n"
             "    elig = set(scored.accession) & set(tmp.accession)\n"
             "    exp = len(a_set & elig) * len(b_set & elig) / max(len(elig), 1)\n"
             "    from scipy.stats import hypergeom\n"
             "    p = hypergeom.sf(len(both) - 1, len(elig), len(b_set & elig), len(a_set & elig)) if both else 1.0\n"
             "    display(Markdown(f'**{len(both)} proteins are in both lists** (≈{exp:.0f} expected by chance among the {len(elig)} proteins '\n"
             "                     f'eligible for both; hypergeometric p = {p:.2g}).'))\n"
             "    ov = tt[tt.accession.isin(both)].merge(ta[['accession', 'agreement']].rename(columns={'agreement': 'agree'}), on='accession', how='left')\n"
             "    ov = ov.sort_values('auc_thermompnn')\n"
             "    ov.to_csv(TABLE_DIR / 'thermo_bottom_both_lists.tsv', sep='\\t', index=False)\n"
             "    display(ov[[c for c in ['gene', 'protein_name', 'family', 'auc_thermompnn', 'auc_esm1v', 'agreement', 'rho_rsa', 'P_in_TM'] if c in ov]].round(3).style.hide(axis='index'))"),
]


FAM_SETUP = r"""import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Markdown, display
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)
from mvsa import talk as T, families as F
from mvsa.data import load, available_predictors
from mvsa.groups import MAIN_PREDICTORS

DATA_DIR = Path(os.environ.get("MVSA_DATA", "../data_families"))
FIG_DIR = Path(os.environ.get("MVSA_FIGURES", "../figures_families"))
MEMBERSHIP = Path(os.environ.get("MVSA_MEMBERSHIP", "../../sets/families/family_membership.tsv"))
MIN_PER_FAMILY = 20          # families with fewer scored proteins are left out of the comparisons

T.style(FIG_DIR)
TABLE_DIR = FIG_DIR / "tables"; TABLE_DIR.mkdir(parents=True, exist_ok=True)
D = load(DATA_DIR)
mem = F.membership(MEMBERSHIP)
PCORR = pd.read_parquet(DATA_DIR / "protein_correlations.parquet") if (DATA_DIR / "protein_correlations.parquet").exists() else None
PRED = [p for p in MAIN_PREDICTORS if p in available_predictors(D["variant_sample"])]
t = F.protein_table(D, PCORR, mem, PRED)
keep = t.family.value_counts()
t = t[t.family.isin(keep[keep >= MIN_PER_FAMILY].index)]
LAB = {"length": "length", "n_tm_helical": "TM helices", "plddt_mean": "pLDDT (whole chain)", "frac_excluded": "fraction disordered/excluded",
       "plddt_trimmed": "pLDDT (structured part)", "frac_helix": "helix fraction", "inversion_index_kd": "hydrophobic surface (inversion)",
       "h_depth_slope_kd": "hydrophobicity rises to core", "belt_contrast_scan_kd": "hydrophobic belt", "contact_density_ca": "packing",
       "cavity_frac": "cavities", "a_over_c": "elongation", "sphericity_hull": "sphericity"}

def pair_label(c):
    a, b = c[4:].split("__")
    return f"{T.name(a)} – {T.name(b)}"

def note(*lines):
    display(Markdown("**Slide notes**\n\n" + "\n".join(f"- {l}" for l in lines)))

t = t.drop(columns=[c for c in t if (c.startswith('rho_') or c.startswith('dr_') or c.startswith('tm_')) and t[c].isna().all()])
KEY = F.main_pair(t)          # ESM-1v - ThermoMPNN agreement, or the next sequence-vs-stability pair with data
print(f"{t.accession.nunique()} proteins in {t.family.nunique()} families | predictors: {', '.join(PRED)} | "
      f"per-protein correlations: {PCORR is not None}")
if PCORR is None:
    print("protein_correlations.parquet missing: run lsf/protcorr.sh families first; agreement sections are skipped")"""

FAMILIES = [
    ("md", "# Variant-effect predictors across membrane protein families\n\n"
           "Representatives of the largest integral-membrane Pfam families (MMseqs2 cluster representatives, see "
           "`scripts/family_sets.py`) scored with every predictor and compared family by family. There are no clinical labels, "
           "so the questions are about the predictors themselves: how much they agree, how widely they spread their scores, "
           "how strongly they penalise transmembrane positions, and which protein properties go with disagreement. "
           "Figures go to `figures_families/`, tables to `figures_families/tables/`."),
    ("code", FAM_SETUP),

    ("md", "## 1. The families"),
    ("code", "summ = F.family_summary(t, [f'dr_{p}' for p in PRED] + [c for c in t if c.startswith('rho_') or c.startswith('tm_')] + F.SHAPE + F.PROTEIN + ['plddt_trimmed'])\n"
             "dom = pd.crosstab(t.family, t.domain.fillna('unknown'), normalize='index')\n"
             "order = summ.sort_values('n_proteins').index.tolist()\n"
             "fig, ax = plt.subplots(figsize=(11, 0.38 * len(order) + 1.8))\n"
             "left = np.zeros(len(order))\n"
             "for k, col in (('Bacteria', T.SOL), ('Archaea', '#8a9a5b'), ('Eukaryota', T.MP), ('unknown', T.LIGHT)):\n"
             "    if k in dom:\n"
             "        v = dom.reindex(order)[k].fillna(0).to_numpy(); ax.barh(range(len(order)), v, left=left, color=col, label=k); left += v\n"
             "ax.set_yticks(range(len(order)), [f\"{f}  ({int(summ.loc[f, 'n_proteins'])}, {summ.loc[f, 'n_tm_helical']:.0f} TM)\" for f in order], fontsize=11)\n"
             "ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); ax.set_xlabel('Share of representatives')\n"
             "ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)\n"
             "T.headline(fig, f'{len(order)} membrane superfamilies, {t.accession.nunique():,} representative proteins',\n"
             "           'Domain of life of the representatives; brackets: proteins scored, median TM helices')\n"
             "T.save(fig, 'F1_families'); plt.show()\n"
             "summ.round(3).to_csv(TABLE_DIR / 'family_summary.tsv', sep='\\t')\n"
             "t.to_csv(TABLE_DIR / 'family_proteins.tsv', sep='\\t', index=False)\n"
             "note(f\"Multipass by UniProt annotation: {(t.class_annotation == 'MULTIPASS').mean():.0%} of representatives (the rest are mostly unannotated TrEMBL topology)\",\n"
             "     *[f'{f}: {int(r.n_proteins)} proteins, {r.frac_eukaryota:.0%} eukaryotic, median length {r.length:.0f}' for f, r in summ.sort_values('n_proteins', ascending=False).iterrows()])"),

    ("md", "## 2. How much do the predictors agree, family by family?\n\n"
           "Median per-protein Spearman ρ between two predictors over all substitutions. Rows are families, sorted by ESM-1v – ThermoMPNN "
           "agreement; the colour shows how each family compares with the others for the same pair (z-score within the column)."),
    ("code", "rho_cols = [c for c in summ if c.startswith('rho_')]\n"
             "if rho_cols and KEY:\n"
             "    key = KEY\n"
             "    m = summ[rho_cols].sort_values(key)\n"
             "    m.columns = [pair_label(c) for c in m.columns]\n"
             "    fig = T.matrix(m, cbar='z-score within pair')\n"
             "    lo_, hi_ = m.index[0], m.index[-1]\n"
             "    col = pair_label(key)\n"
             "    T.headline(fig, f'Sequence and stability predictors agree least in {lo_} and most in {hi_}',\n"
             "               f'{col}: median ρ {m.loc[lo_, col]:.2f} vs {m.loc[hi_, col]:.2f}; numbers are medians over each family\\'s proteins')\n"
             "    T.save(fig, 'F2_agreement_matrix'); plt.show()\n"
             "    fig = T.family_boxes(t, key, f'Spearman ρ, {pair_label(key)}, per protein')\n"
             "    spread = t.groupby('family')[key].median()\n"
             "    T.headline(fig, f'{pair_label(key)} agreement ranges from {spread.min():.2f} to {spread.max():.2f} across families',\n"
             "               'Each point is one representative protein; boxes show the interquartile range')\n"
             "    T.save(fig, 'F3_thermo_esm_by_family'); plt.show()\n"
             "    note(*[f'{pair_label(c)}: family medians {summ[c].min():.2f}–{summ[c].max():.2f} (lowest {summ[c].idxmin()}, highest {summ[c].idxmax()})' for c in rho_cols if summ[c].notna().any()])"),

    ("md", "## 3. Score spread and how strongly TM positions are penalised"),
    ("code", "dr = summ[[f'dr_{p}' for p in PRED if f'dr_{p}' in summ]].sort_values(f'dr_esm1v' if 'dr_esm1v' in summ else f'dr_{PRED[0]}')\n"
             "dr.columns = [T.name(c[3:]) for c in dr.columns]\n"
             "fig = T.matrix(dr, fmt='{:.2g}', cbar='z-score within predictor')\n"
             "T.headline(fig, 'How widely each predictor spreads its scores, by family',\n"
             "           'Median per-protein 5–95% range of oriented scores; a narrow range means the predictor barely separates substitutions')\n"
             "T.save(fig, 'F4_dynamic_range'); plt.show()\n"
             "tm = summ[[f'tm_{p}' for p in PRED if f'tm_{p}' in summ]]\n"
             "if len(tm.columns):\n"
             "    tm = tm.sort_values(tm.columns[0]); tm.columns = [T.name(c[3:]) for c in tm.columns]\n"
             "    fig = T.matrix(tm, fmt='{:+.2g}', cbar='z-score within predictor')\n"
             "    T.headline(fig, 'How much more damaging each predictor finds TM substitutions',\n"
             "               'Median per protein of (median score in TM segments − median score outside); positive = TM penalised')\n"
             "    T.save(fig, 'F5_tm_penalty'); plt.show()"),

    ("md", "## 4. Structure of each family"),
    ("code", "sc = [c for c in F.SHAPE if c in summ and c != 'plddt_mean'] + (['plddt_trimmed'] if 'plddt_trimmed' in summ else [])\n"
             "sm = summ[sc].sort_values('inversion_index_kd' if 'inversion_index_kd' in sc else sc[0])\n"
             "sm.columns = [LAB.get(c, c) for c in sm.columns]\n"
             "fig = T.matrix(sm, fmt='{:.2g}', cbar='z-score within metric')\n"
             "T.headline(fig, 'Structural fingerprint of each family', 'Median over representatives (trimmed chains)')\n"
             "T.save(fig, 'F6_structure'); plt.show()"),

    ("md", "## 5. What goes with disagreement?\n\n"
           "Correlation between ESM-1v – ThermoMPNN agreement and protein properties: pooled over all proteins (open), and the median of "
           "the correlations computed inside each family (filled). Only the within-family value is free of differences between families."),
    ("code", "key = KEY\n"
             "if key:\n"
             "    feats = F.PROTEIN + [c for c in F.SHAPE if c != 'plddt_mean'] + ['plddt_trimmed', 'cluster_size'] + [f'dr_{p}' for p in PRED]\n"
             "    drv = F.drivers(t, key, feats)\n"
             "    assert len(drv), 'too few proteins per family to compute within-family correlations'\n"
             "    fig = T.drivers_dots(drv, {**LAB, 'cluster_size': 'cluster size', **{f'dr_{p}': f'{T.name(p)} range' for p in PRED}})\n"
             "    top = drv.iloc[0]\n"
             "    T.headline(fig, f\"Within families, agreement goes with {LAB.get(top.feature, top.feature)} (ρ = {top.rho_within_median:+.2f})\",\n"
             "               f'Spearman ρ between {pair_label(key)} agreement and each protein property')\n"
             "    T.save(fig, 'F7_agreement_drivers'); plt.show()\n"
             "    drv.round(3).to_csv(TABLE_DIR / 'agreement_drivers.tsv', sep='\\t', index=False)\n"
             "    de = F.domain_effect(t, key)\n"
             "    if len(de):\n"
             "        de['diff'] = de.eukaryota - de.bacteria\n"
             "        display(de.sort_values('diff').round(3).style.hide(axis='index'))\n"
             "        note(f\"Eukaryotic minus bacterial agreement: median {de['diff'].median():+.2f} across {len(de)} families with both\")"),

    ("md", "## 6. The least-agreeing proteins in each family"),
    ("code", "if KEY:\n"
             "    key = KEY\n"
             "    worst = t.sort_values(key).groupby('family').head(5)\n"
             "    cols = [c for c in ['family', 'accession', 'gene', 'protein_name', 'domain', 'kingdom', 'length', 'n_tm_helical', key,\n"
             "                        F.pair_col('esm1v', 'saprot'), F.pair_col('thermompnn', 'solvent_exposure'), 'plddt_mean', 'frac_excluded'] if c in worst]\n"
             "    worst[cols].to_csv(TABLE_DIR / 'least_agreeing_per_family.tsv', sep='\\t', index=False)\n"
             "    display(worst[cols].round(3).style.hide(axis='index'))"),
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
build(TALK, root / "talk_figures.ipynb")
build(FAMILIES, root / "families.ipynb")
