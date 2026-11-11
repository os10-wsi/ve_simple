"""Generate notebooks/analysis.ipynb (outputs cleared). Run it with:
    jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output analysis.executed.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = [
    ("md", "# Proteome-wide protein shape analysis\n\nReads `results/metrics.parquet` (one row per protein x unit x "
           "mask variant x domain) and runs the Stage 5 analyses from `protshape.analysis`. "
           "Classes and domain labels are UniProt-annotation based (`class_annotation`); `class_final` is a sensitivity analysis."),
    ("code", "import pandas as pd, numpy as np\nfrom pathlib import Path\nfrom IPython.display import Image, display\n"
             "from protshape import analysis as A\npd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)\n"
             "import os\nWORK = Path(os.environ.get('PROTSHAPE_WORK', '../results' if Path('../results/metrics.parquet').exists() else 'results'))\n"
             "m = pd.read_parquet(WORK / 'metrics.parquet')\n"
             "metrics = A.metric_columns(m[(m.unit == 'DOMAIN') & (m.mask_variant == 'default')])\n"
             "print(m.accession.nunique(), 'proteins;', len(metrics), 'metrics')\n"
             "A.domains(m).groupby(['taxid', 'group']).size().unstack(fill_value=0)"),
    ("md", "## 1. Size dependence (within SOLUBLE_DOMAIN / SOLUBLE proteins)"),
    ("code", "sd = A.size_dependence(m, metrics)\nsd[sd.unit == 'DOMAIN'].assign(a=lambda x: x.spearman_rho_vs_N.abs())"
             ".sort_values('a').drop(columns='a').round(3)"),
    ("md", "## 2. Membrane vs soluble domains, size-matched (Cliff's delta and AUC per size bin)"),
    ("code", "mc = A.main_comparison(m, metrics)\nkey = A.SHAPE + A.HYDRO\n"
             "mc[mc.metric.isin(key) & (mc.a == 'MEMBRANE_DOMAIN') & (mc.b == 'SOLUBLE_DOMAIN')]"
             ".pivot_table(index='metric', columns='size_bin', values='auc').round(2)"),
    ("md", "Per organism (stratified AUC), then pooled with organism as a stratum:"),
    ("code", "per = []\nfor org, t in ((9606, 'human'), (559292, 'yeast')):\n"
             "    r = A.main_comparison(m, metrics, organism=org)\n"
             "    per.append(r[(r.size_bin == 'stratified') & (r.a == 'MEMBRANE_DOMAIN') & (r.b == 'SOLUBLE_DOMAIN')]"
             "[['metric', 'auc']].rename(columns={'auc': t}))\n"
             "pooled = mc[(mc.size_bin == 'stratified') & (mc.a == 'MEMBRANE_DOMAIN') & (mc.b == 'SOLUBLE_DOMAIN')][['metric', 'auc']]"
             ".rename(columns={'auc': 'pooled'})\n"
             "per[0].merge(per[1], on='metric').merge(pooled, on='metric').set_index('metric').loc[lambda x: x.index.isin(key)].round(2)"),
    ("md", "## 3. Secondary-structure control (CATH class 1) and beta barrels"),
    ("code", "ctrl = A.cath1_control(m, metrics)\nA.surviving_metrics(mc, ctrl).round(3).head(25)"),
    ("code", "A.barrel_vs_cath2(m, metrics).query(\"size_bin == 'stratified'\").sort_values('auc').round(3).head(10)"),
    ("md", "## 4. Whole-protein view and where elongation lives"),
    ("code", "A.elongation_location(m).round(3)"),
    ("code", "wp = A.whole_protein(m, metrics)\nwp[(wp.size_bin == 'stratified') & wp.metric.isin(A.SHAPE)]"
             ".pivot_table(index='metric', columns='comparison', values='auc').round(2)"),
    ("md", "## 5. Prolate vs oblate"),
    ("code", "A.prolateness_by_group(m).round(3)"),
    ("md", "## 6. Hydrophobicity vs depth, and recovering the membrane label from structure alone"),
    ("code", "A.hydrophobicity_by_group(m).round(2)"),
    ("code", "A.structure_only_roc(m, metrics).sort_values('auc_oriented', ascending=False).round(3).head(20)"),
    ("md", "## 7. Sensitivity: pLDDT 50 / 90, no TED exclusion, class_final"),
    ("code", "sens = A.sensitivity(m, metrics)\nsens[sens.conclusion_moves_with != ''].round(3)"),
    ("md", "## 8. What TED's globularity terms capture"),
    ("code", "A.ted_globularity_correlation(m, metrics).assign(a=lambda x: x.spearman.abs())"
             ".sort_values('a', ascending=False).drop(columns='a').round(3).head(20)"),
    ("md", "## 9. PCA and UMAP of DOMAIN units"),
    ("code", "emb = A.embeddings(m, metrics)\nprint('PCA explained variance:', np.round(emb.attrs['explained'], 3))\n"
             "A.plot_embeddings(emb, WORK / 'analysis' / 'embeddings.png')\ndisplay(Image(WORK / 'analysis' / 'embeddings.png'))"),
    ("code", "A.plot_distributions(m, WORK / 'analysis' / 'distributions.png')\ndisplay(Image(WORK / 'analysis' / 'distributions.png'))"),
    ("md", "## 10. findings.md\n\n`protshape analyze` writes `results/findings.md` from the same functions."),
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
