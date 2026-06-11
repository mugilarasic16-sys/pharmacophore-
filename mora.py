# =================================================================
# Pharmacophore Mapping Pipeline — Streamlit App
# Author: Mugil | B.Pharm Final Year | Pharmaceutical Chemistry
# =================================================================

import os
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures, Draw
from rdkit.Chem import rdMolDescriptors
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# BACKEND: Pharmacophore Extraction Engine
# ------------------------------------------------------------------

FEATURE_COLORS = {
    "Donor":       "#3B82F6",   # blue
    "Acceptor":    "#EF4444",   # red
    "Hydrophobe":  "#F59E0B",   # amber
    "Aromatic":    "#8B5CF6",   # violet
    "LumpedHydrophobe": "#F97316",  # orange
    "PosIonizable": "#10B981",  # emerald
    "NegIonizable": "#EC4899",  # pink
}

def get_feature_factory():
    fdef_path = os.path.join(Chem.RDConfig.RDDataDir, 'BaseFeatures.fdef')
    return ChemicalFeatures.BuildFeatureFactory(fdef_path)

def run_pharmacophore_pipeline(ligand_data, num_conformers=50):
    """
    For each valid ligand SMILES:
      1. Parse and validate molecule
      2. Generate 3D conformers + MMFF minimization
      3. Extract pharmacophoric features from lowest-energy conformer
    Returns a DataFrame of all features across all ligands.
    """
    processed = []
    skipped = []

    for entry in ligand_data:
        smiles = str(entry.get("SMILES", "")).strip()
        if not smiles:
            continue
        lig_id   = str(entry.get("Ligand ID",   "N/A")).strip()
        lig_name = str(entry.get("Ligand Name", "N/A")).strip()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped.append(f"{lig_id} ({lig_name})")
            continue

        mol = Chem.AddHs(mol)
        mol.SetProp("_Name",      lig_name)
        mol.SetProp("Ligand_ID",  lig_id)
        mol.SetProp("SMILES",     smiles)
        processed.append(mol)

    if not processed:
        return None, [], skipped

    # 3D conformation generation
    errors = []
    valid_mols = []
    for mol in processed:
        result = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, randomSeed=42)
        if not result:
            errors.append(mol.GetProp("_Name"))
            continue
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        valid_mols.append(mol)

    if len(valid_mols) < 2:
        return None, errors + skipped, skipped

    # Feature extraction
    factory = get_feature_factory()
    records = []

    for mol in valid_mols:
        name   = mol.GetProp("_Name")
        lig_id = mol.GetProp("Ligand_ID")
        smiles = mol.GetProp("SMILES")
        feats  = factory.GetFeaturesForMol(mol, confId=0)

        mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
        hbd    = rdMolDescriptors.CalcNumHBD(mol)
        hba    = rdMolDescriptors.CalcNumHBA(mol)

        for f in feats:
            pos = f.GetPos()
            records.append({
                "Ligand ID":      lig_id,
                "Ligand Name":    name,
                "SMILES":         smiles,
                "MW":             round(mol_wt, 2),
                "HBD":            hbd,
                "HBA":            hba,
                "Feature Family": f.GetFamily(),
                "Feature Type":   f.GetType(),
                "X":              round(pos.x, 3),
                "Y":              round(pos.y, 3),
                "Z":              round(pos.z, 3),
            })

    df = pd.DataFrame(records)
    return df, errors, skipped


def find_shared_features(df):
    """Return feature families present in ALL ligands."""
    if df is None or df.empty:
        return []
    ligands = df["Ligand Name"].unique()
    shared = []
    for family in df["Feature Family"].unique():
        sub = df[df["Feature Family"] == family]
        if len(sub["Ligand Name"].unique()) == len(ligands):
            shared.append(family)
    return shared


# ------------------------------------------------------------------
# FRONTEND: Page Layout & Styling
# ------------------------------------------------------------------

st.set_page_config(
    page_title="PharmMap | Pharmacophore Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
  /* Inter font for clean scientific UI */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Top header bar */
  .pharm-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #0F172A 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border-left: 4px solid #3B82F6;
  }
  .pharm-header h1 {
    color: #F8FAFC;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
  }
  .pharm-header p {
    color: #94A3B8;
    font-size: 0.9rem;
    margin: 0;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    min-width: 140px;
    flex: 1;
  }
  .metric-card .label {
    color: #64748B;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
  }
  .metric-card .value {
    color: #F1F5F9;
    font-size: 1.7rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
  }
  .metric-card .sub {
    color: #3B82F6;
    font-size: 0.75rem;
    margin-top: 4px;
  }

  /* Shared feature badge */
  .shared-badge {
    display: inline-block;
    background: #1E3A5F;
    border: 1px solid #3B82F6;
    color: #93C5FD;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 4px 4px 4px 0;
  }
  .shared-none {
    color: #64748B;
    font-size: 0.85rem;
    font-style: italic;
  }

  /* Section headers */
  .section-label {
    color: #3B82F6;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 6px;
  }

  /* Warning / skip boxes */
  .skip-box {
    background: #1C1917;
    border: 1px solid #78350F;
    border-radius: 8px;
    padding: 10px 16px;
    color: #FCD34D;
    font-size: 0.82rem;
    margin-top: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="pharm-header">
  <h1>🧬 PharmMap — Multi-Ligand Pharmacophore Pipeline</h1>
  <p>3D conformer generation · Spatial feature extraction · Cross-ligand pharmacophore analysis</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Parameters")
    num_confs = st.slider(
        "Conformers per ligand",
        min_value=10, max_value=150, value=50, step=10,
        help="More conformers → better geometry sampling, but slower. 50 is a good default."
    )
    st.markdown("---")
    st.markdown("### 🔎 Feature Filter")
    feature_filter = st.multiselect(
        "Show only selected feature families",
        options=["Donor", "Acceptor", "Hydrophobe", "LumpedHydrophobe",
                 "Aromatic", "PosIonizable", "NegIonizable"],
        default=[],
        help="Leave empty to show all feature families."
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>Built with RDKit · Streamlit<br>"
        "Pharmacophore features from BaseFeatures.fdef</small>",
        unsafe_allow_html=True
    )


# ── Input Tabs ───────────────────────────────────────────────────
st.markdown('<div class="section-label">Input — Ligand Dataset</div>', unsafe_allow_html=True)
tabs = st.tabs(["⌨️ Manual Entry", "📂 Upload CSV"])

final_input_data = None

with tabs[0]:
    st.caption("Edit cells directly. Add rows with the '+' button. Requires at least 2 valid SMILES.")
    default_data = pd.DataFrame([
        {"Ligand ID": "LIG-001", "Ligand Name": "Captopril",   "SMILES": "CC(CS)C(=O)N1CCCC1C(=O)O"},
        {"Ligand ID": "LIG-002", "Ligand Name": "Enalaprilat", "SMILES": "CC(NC(C)C(=O)O)C(=O)N1CCCC1C(=O)O"},
        {"Ligand ID": "LIG-003", "Ligand Name": "",             "SMILES": ""},
    ])
    input_df = st.data_editor(
        default_data, num_rows="dynamic",
        use_container_width=True, key="manual_grid"
    )
    final_input_data = input_df.to_dict(orient="records")

with tabs[1]:
    st.caption("CSV must have columns: `Ligand ID`, `Ligand Name`, `SMILES`")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        file_df = pd.read_csv(uploaded)
        required_cols = {"Ligand ID", "Ligand Name", "SMILES"}
        if not required_cols.issubset(set(file_df.columns)):
            st.error(f"Missing columns. Found: {list(file_df.columns)}. Required: {list(required_cols)}")
        else:
            st.dataframe(file_df.head(10), use_container_width=True)
            final_input_data = file_df.to_dict(orient="records")

# ── Run Button ───────────────────────────────────────────────────
st.markdown("---")
run_btn = st.button("▶  Run Pharmacophore Pipeline", type="primary", use_container_width=True)

if run_btn:
    if not final_input_data:
        st.error("No input data found. Fill the manual grid or upload a CSV.")
        st.stop()

    with st.spinner("Generating 3D conformers and extracting pharmacophoric features…"):
        results_df, errors, skipped = run_pharmacophore_pipeline(
            final_input_data, num_conformers=num_confs
        )

    # Show any skipped/error ligands
    if skipped:
        st.markdown(
            f'<div class="skip-box">⚠️ Skipped (invalid SMILES): {", ".join(skipped)}</div>',
            unsafe_allow_html=True
        )
    if errors:
        st.markdown(
            f'<div class="skip-box">⚠️ 3D embedding failed for: {", ".join(errors)}</div>',
            unsafe_allow_html=True
        )

    if results_df is None or results_df.empty:
        st.error("Pipeline failed. Check your SMILES strings and try again.")
        st.stop()

    # Apply sidebar feature filter
    display_df = results_df.copy()
    if feature_filter:
        display_df = display_df[display_df["Feature Family"].isin(feature_filter)]

    # ── Summary Metrics ─────────────────────────────────────────
    st.success("Pipeline completed successfully.")
    n_ligands  = results_df["Ligand Name"].nunique()
    n_features = len(results_df)
    n_families = results_df["Feature Family"].nunique()
    shared     = find_shared_features(results_df)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="label">Ligands Processed</div>
        <div class="value">{n_ligands}</div>
        <div class="sub">valid molecules</div>
      </div>
      <div class="metric-card">
        <div class="label">Total Features</div>
        <div class="value">{n_features}</div>
        <div class="sub">across all conformers</div>
      </div>
      <div class="metric-card">
        <div class="label">Feature Families</div>
        <div class="value">{n_families}</div>
        <div class="sub">unique types</div>
      </div>
      <div class="metric-card">
        <div class="label">Shared Families</div>
        <div class="value">{len(shared)}</div>
        <div class="sub">present in all ligands</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Shared feature badges
    st.markdown('<div class="section-label">Pharmacophore Features Shared Across All Ligands</div>',
                unsafe_allow_html=True)
    if shared:
        badges = "".join([f'<span class="shared-badge">✓ {f}</span>' for f in shared])
        st.markdown(badges, unsafe_allow_html=True)
    else:
        st.markdown('<span class="shared-none">No single feature family is common to all ligands.</span>',
                    unsafe_allow_html=True)

    # ── Visualizations ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Visualizations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Feature family distribution per ligand
        count_df = (results_df.groupby(["Ligand Name", "Feature Family"])
                    .size().reset_index(name="Count"))
        fig1 = px.bar(
            count_df, x="Feature Family", y="Count",
            color="Ligand Name", barmode="group",
            title="Feature Family Distribution per Ligand",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_dark",
        )
        fig1.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#94A3B8", size=11),
            legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1),
            title_font_color="#F1F5F9",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Overall feature family pie
        pie_df = results_df["Feature Family"].value_counts().reset_index()
        pie_df.columns = ["Feature Family", "Count"]
        fig2 = px.pie(
            pie_df, names="Feature Family", values="Count",
            title="Overall Feature Family Composition",
            color_discrete_sequence=list(FEATURE_COLORS.values()),
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="#0F172A",
            font=dict(color="#94A3B8"),
            title_font_color="#F1F5F9",
        )
        fig2.update_traces(textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    # 3D scatter of feature coordinates
    st.markdown('<div class="section-label">3D Spatial Feature Map</div>', unsafe_allow_html=True)
    plot_df = display_df.copy()
    plot_df["Color"] = plot_df["Feature Family"].map(FEATURE_COLORS).fillna("#94A3B8")

    fig3 = go.Figure()
    for family in plot_df["Feature Family"].unique():
        sub = plot_df[plot_df["Feature Family"] == family]
        fig3.add_trace(go.Scatter3d(
            x=sub["X"], y=sub["Y"], z=sub["Z"],
            mode="markers",
            name=family,
            marker=dict(
                size=6,
                color=FEATURE_COLORS.get(family, "#94A3B8"),
                opacity=0.82,
                line=dict(width=0.5, color="#0F172A"),
            ),
            text=sub["Ligand Name"] + " | " + sub["Feature Type"],
            hovertemplate="<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>",
        ))

    fig3.update_layout(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        scene=dict(
            bgcolor="#0F172A",
            xaxis=dict(backgroundcolor="#0F172A", color="#64748B", gridcolor="#1E293B"),
            yaxis=dict(backgroundcolor="#0F172A", color="#64748B", gridcolor="#1E293B"),
            zaxis=dict(backgroundcolor="#0F172A", color="#64748B", gridcolor="#1E293B"),
        ),
        font=dict(color="#94A3B8"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Data Table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Feature Coordinate Matrix</div>', unsafe_allow_html=True)

    show_cols = ["Ligand ID", "Ligand Name", "Feature Family", "Feature Type",
                 "X", "Y", "Z", "MW", "HBD", "HBA"]
    st.dataframe(
        display_df[show_cols].reset_index(drop=True),
        use_container_width=True, height=350
    )

    # ── Download ─────────────────────────────────────────────────
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Feature Report (.CSV)",
        data=csv_bytes,
        file_name="pharmacophore_features.csv",
        mime="text/csv",
        use_container_width=True,
    )
