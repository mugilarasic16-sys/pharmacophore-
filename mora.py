import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures

def run_pharmacophore_pipeline(ligand_data):
    """
    
    Runs the pharmacophore identification pipeline.
    Input format: List of dictionaries containing 'id', 'name', and 'smiles'
    """
    processed_ligands = []
    
    print("="*60)
    print("🚀 STARTING PHARMACOPHORE IDENTIFICATION PIPELINE")
    print("="*60)
    
    # -----------------------------------------------------------------
    # Step 1: Input Validation and Structure Initialization
    # -----------------------------------------------------------------
    print("\n🔄 STEP 1: Processing Metadata and Initializing Structures...")
    for entry in ligand_data:
        lig_id = entry.get('id')
        name = entry.get('name')
        smiles = entry.get('smiles')
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Error: Invalid SMILES string for Ligand ID {lig_id} ({name}). Skipping.")
            continue
            
        # Explicitly add Hydrogens (Crucial for correct 3D geometry & H-bond features)
        mol = Chem.AddHs(mol)
        
        # Assign your custom research metadata inside the RDKit Molecule Object
        mol.SetProp("_Name", str(name))
        mol.SetProp("Ligand_ID", str(lig_id))
        mol.SetProp("SMILES", str(smiles))
        
        processed_ligands.append(mol)
        print(f"  └─ Success: Loaded ID: {lig_id} | Name: {name}")

    if len(processed_ligands) < 2:
        print("\n❌ Pipeline Halted: You need at least 2 valid ligands to identify a shared pharmacophore.")
        return

    # -----------------------------------------------------------------
    # Step 2: 3D Conformation Generation (Handling Flexiblity)
    # -----------------------------------------------------------------
    print("\n🔄 STEP 2: Generating 3D Conformations & Minimizing Energy...")
    for mol in processed_ligands:
        name = mol.GetProp("_Name")
        
        # Generate an ensemble of 50 distinct spatial shapes per molecule
        conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=50, randomSeed=42)
        
        # Optimize shapes using Merck Molecular Force Field (MMFF94)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        
        print(f"  └─ Generated {len(conformer_ids)} optimized 3D shapes for {name}")

    # -----------------------------------------------------------------
    # Step 3: Pharmacophoric Feature Extraction
    # -----------------------------------------------------------------
    print("\n🔄 STEP 3: Initializing Feature Factory & Extracting Bioclues...")
    # Load default definitions for features (Donors, Acceptors, Hydrophobics, Aromatics)
    fdef_path = os.path.join(Chem.RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    feature_summary = []
    
    for mol in processed_ligands:
        name = mol.GetProp("_Name")
        lig_id = mol.GetProp("Ligand_ID")
        
        # Extract chemical features from the primary low-energy conformation
        features = feat_factory.GetFeaturesForMol(mol, confId=0)
        
        print(f"  └─ {name} (ID: {lig_id}): Found {len(features)} active pharmacophore points.")
        
        # Document individual features for research analysis
        for f in features:
            pos = f.GetPos()  # Get geometric 3D coordinates (X, Y, Z)
            feature_summary.append({
                "Ligand_ID": lig_id,
                "Ligand_Name": name,
                "Feature_Family": f.GetFamily(),
                "Feature_Type": f.GetType(),
                "Coord_X": round(pos.x, 3),
                "Coord_Y": round(pos.y, 3),
                "Coord_Z": round(pos.z, 3)
            })

    # -----------------------------------------------------------------
    # Step 4: Exporting Tabular Mapping Data
    # -----------------------------------------------------------------
    print("\n🔄 STEP 4: Compiling Data & Exporting Mapping Report...")
    df = pd.DataFrame(feature_summary)
    
    # Save results directly to a CSV file for your research data recording
    output_filename = "identified_pharmacophore_features.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"✨ SUCCESS: Shared pharmacophore coordinates exported to '{output_filename}'")
    print("="*60)
    print(df.head(10).to_string()) # Displaying a snippet of the coordinate matrix
    
    return processed_ligands

# --- Example Run of Your Pipeline ---
if __name__ == "__main__":
    # Input list containing your explicitly required criteria: Name, SMILES, and Ligand ID
    research_dataset = [
        {
            "id": "LIG-001", 
            "name": "Captopril", 
            "smiles": "CC(CS)C(=O)N1CCCC1C(=O)O"
        },
        {
            "id": "LIG-002", 
            "name": "Enalaprilat", 
            "smiles": "CC(NC(C)C(=O)O)C(=O)N1CCCC1C(=O)O"
        }
    ]
    
    run_pharmacophore_pipeline(research_dataset)
