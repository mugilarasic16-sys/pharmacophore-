# =================================================================
# 🛠️ SELF-INSTALLATION BOOTSTRAPPER (Runs before anything else)
# =================================================================
import subprocess
import sys

def install_dependencies():
    try:
        # Check if rdkit is already available in the environment
        import rdkit
    except ImportError:
        # If not, force-install it automatically on the cloud container
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit", "pandas", "streamlit"])

# Run the auto-installer
install_dependencies()

# =================================================================
# 🧬 CORE PIPELINE AND USER INTERFACE APPLICATION
# =================================================================
import os
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures

# 🧪 BACKEND: Pharmacophore Extraction Computational Engine
def run_pharmacophore_pipeline(ligand_data, num_conformers=50):
    processed_ligands = []
    feature_summary = []
    
    # Step 1: Input Validation and Structure Initialization
    for entry in ligand_data:
        if not entry.get('SMILES') or str(entry.get('SMILES')).strip() == "":
            continue
            
        lig_id = entry.get('Ligand ID', 'N/A')
        name = entry.get('Ligand Name', 'N/A')
        smiles = str(entry.get('SMILES')).strip()
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error(f"❌ Invalid SMILES string detected for ID {lig_id} ({name}). Skipping.")
            continue
            
        mol = Chem.AddHs(mol)
        mol.SetProp("_Name", str(name))
        mol.SetProp("Ligand_ID", str(lig_id))
        mol.SetProp("SMILES", smiles)
        processed_ligands.append(mol)

    if len(processed_ligands) < 2:
        st.warning("⚠️ You need at least 2 valid ligands to identify a shared pharmacophore layout.")
        return None

    # Step 2: 3D Conformation Generation & Energy Minimization
    for mol in processed_ligands:
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, randomSeed=42)
        AllChem.MMFFOptimizeMoleculeConfs(mol)

    # Step 3: Pharmacophoric Feature Mapping
    fdef_path = os.path.join(Chem.RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    
    for mol in processed_ligands:
        name = mol.GetProp("_Name")
        lig_id = mol.GetProp("Ligand_ID")
        features = feat_factory.GetFeaturesForMol(mol, confId=0)
        
        for f in features:
            pos = f.GetPos()
            feature_summary.append({
                "Ligand ID": lig_id,
                "Ligand Name": name,
                "Feature Family": f.GetFamily(),
                "Feature Type": f.GetType(),
                "Coord X": round(pos.x, 3),
                "Coord Y": round(pos.y, 3),
                "Coord Z": round(pos.z, 3)
            })

    return pd.DataFrame(feature_summary)


# 💻 FRONTEND: Streamlit User Interface Layout
st.set_page_config(page_title="Pharmacophore Mapping Pipeline", layout="wide")

st.title("🧬 Multi-Ligand Pharmacophore Mapping Pipeline")
st.markdown("Enter your ligand metadata and SMILES structures below to generate an automated 3D spatial feature coordinate matrix.")

# Sidebar Configuration Parameters
st.sidebar.header("⚙️ Pipeline Settings")
conf_slider = st.sidebar.slider(
    "Conformations per Ligand", 
    min_value=10, 
    max_value=100, 
    value=50, 
    step=10,
    help="Higher numbers increase geometry accuracy but require more computational time."
)

# Structural Layout: Split options for manual data entry vs file uploading
tabs = st.tabs(["⌨️ Manual Entry Grid", "📂 Bulk File Upload"])

with tabs[0]:
    st.subheader("Interactive Ligand Data Sheet")
    st.caption("Double-click cells below to type or paste your research data directly.")
    
    default_data = pd.DataFrame([
        {"Ligand ID": "LIG-001", "Ligand Name": "Captopril", "SMILES": "CC(CS)C(=O)N1CCCC1C(=O)O"},
        {"Ligand ID": "LIG-002", "Ligand Name": "Enalaprilat", "SMILES": "CC(NC(C)C(=O)O)C(=O)N1CCCC1C(=O)O"},
        {"Ligand ID": "", "Ligand Name": "", "SMILES": ""},
    ])
    
    input_df = st.data_editor(
        default_data, 
        num_rows="dynamic", 
        use_container_width=True,
        key="manual_grid"
    )
    final_input_data = input_df.to_dict(orient="records")

with tabs[1]:
    st.subheader("Bulk Dataset Processing")
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing columns named exactly: 'Ligand ID', 'Ligand Name', and 'SMILES'", 
        type=["csv"]
    )
    
    if uploaded_file is not None:
        file_df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Preview:", file_df.head(5))
        final_input_data = file_df.to_dict(orient="records")

# Execution Action Trigger Button
st.markdown("---")
if st.button("🚀 Run Pharmacophore Pipeline", type="primary", use_container_width=True):
    with st.spinner("Executing 3D conformation modeling and extracting spatial coordinates..."):
        
        results_df = run_pharmacophore_pipeline(final_input_data, num_conformers=conf_slider)
        
        if results_df is not None and not results_df.empty:
            st.success("✨ Shared Pharmacophore Coordinate Matrix Successfully Identified!")
            st.dataframe(results_df, use_container_width=True)
            
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Spatial Features Report (.CSV)",
                data=csv_data,
                file_name="identified_pharmacophore_features.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.error("Pipeline run failed. Please verify that your table rows contain structurally valid SMILES codes.")
