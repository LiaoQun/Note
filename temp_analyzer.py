import pandas as pd
import warnings
from collections import defaultdict

# Suppress pandas warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# The data from the previous analysis
problematic_data = {
    'C#CCCCCCCN': [9, 11, 13, 15, 17, 19, 21],
    'C#Cc1ccc(OC)cc1': [10, 11, 12, 13],
    'C/C=C(/C)CC': [5, 9, 14],
    'C/C=C/N1CCNCC1': [9, 12, 13, 14, 16, 18],
    'C1=COC2=C(C1)CCCC2': [11, 12, 13, 15, 17, 19, 21],
    'C1=Cc2ccccc2C1': [10, 11, 12, 13, 14, 15, 16],
    'C1CC1[C@@H]1CC[C@@H]1C1CC1': [16, 18, 20, 23],
    'C1CCC2(CC1)OO2': [9, 11, 13],
    'C1CCC[C@@H]2C[C@@H]2CC1': [10, 12, 14, 16, 19, 22, 24],
    'C1CN2NN=NC2=N1': [9, 11, 13],
    'C1N=N[C@@H]2O[C@H]12': [7],
    'C=C(C)[C@@H]1CCC=C(C)C1': [10, 12, 16, 18, 20, 21, 24],
    'C=C(OCC)C(N)=O': [7, 9, 11, 14],
    'C=C/C=C/N(CC)CC': [8, 11, 12, 13, 15, 16, 18, 20],
    'C=C1CCC(C)(C)CC1': [9, 11, 13, 15],
    'C=CCC#C[C@@H](O)CCC': [9, 11, 12, 15, 16, 18, 20],
    'C=CCC(C#N)CC=C': [13],
    'C=C[C@H](C)CCC': [6, 8, 13, 15, 17],
    'CC(=O)CC(C)(C)OO': [8, 11],
    'CC(=O)NCCCCCN': [9, 12, 13, 15, 17, 19, 21, 23],
    'CC(=O)NCCCN': [7, 10, 11, 13, 15, 17],
    'CC(=O)O[C@@H]1CCNC1': [9, 13, 15, 17, 18],
    'CC(=O)[C@@H]1O[C@@H]1C': [7, 12],
    'CC(C)(C)OCCNN': [8, 17, 19, 21, 22],
    'CC(C)CO/C=C\C=O': [11, 17, 18],
    'CC(C)OCCC=O': [10, 14, 16, 18],
    'CC(C)OCOCOCO': [12, 16, 18, 20, 22],
    'CC(C)[C@@H]1CN1': [9, 14, 16],
    'CC(O)(O)n1ccnc1': [9, 14, 15],
    'CC/C(C)=C/CC(=O)O': [8, 11, 13, 16, 17],
    'CC/C=N/CCCCC': [8, 11, 13, 14, 16, 18, 20],
    'CC1(C(=O)O)CCCC1': [9, 12],
    'CC1(C)CC[C@@H]2CN21': [15, 17, 20],
    'CC1(C)C[C@@H]1/C=N/O': [14, 17, 18],
    'CC1(NC2CC2)CNC1': [10, 13, 14, 21],
    'CC1=C(C)C(C)=C(C)C1': [12],
    'CC1=CN(C)C(=O)[C@@H]1C': [9, 12, 13, 17],
    'CC1=CO[C@H](C)C1=O': [8, 13],
    'CCC(=O)[C@@H]1CCCO1': [9, 12, 15, 17, 19],
    'CCCC(CCC)CCC': [9, 12, 14, 16],
    'CCCC(CCC)[C@H](C)O': [16, 25, 28],
    'CCCCC/C=C/CCO': [9, 12, 14, 16, 18, 20, 21, 22, 24, 26],
    'CCCN1C(=O)CC[C@@H]1C': [10, 13, 15, 17, 19, 22],
    'CCC[C@H](N)C(=O)O': [7, 10, 12, 15],
    'CCC[C@H](O)CC(=O)CC': [9, 12, 14, 17, 18, 20, 22],
    'CCN(C(C)=O)C(=O)OC': [9, 12, 14, 17],
    'CCO[C@@H]1CC(=O)C1(C)C': [10, 13, 16, 18, 20],
    'CC[C@@H]1CN1C': [6, 9, 12, 14],
    'CC[C@H](C)c1nnc(N)o1': [10, 13, 16, 19],
    'CC[C@H](OC)C(OC)OC': [9, 12, 15, 18],
    'CC[C@H]1CC[C@H]1CC': [8, 11, 14, 16, 21],
    'CC[C@]1(C)CCN1': [10, 12, 15, 17, 19],
    'CCc1ccoc1': [7, 10, 12, 13, 14],
    'CCc1cncc(CC)n1': [10, 13, 15],
    'CCn1ccc(OC)c1C': [10, 13, 15, 16, 17, 20],
    'CCn1cccc(N)c1=O': [15, 16, 17, 18],
    'CN(C)c1cnc[nH]1': [8, 14, 15, 16],
    'CN/N=C(/C)N': [5, 8, 9, 12],
    'CN1C=C[C@H](O)N(C)C1=O': [10, 13, 14, 16, 17],
    'CN1CCCO[C@]1(C)O': [9, 12, 14, 16, 18],
    'CN1CCN(C)C(=O)C1': [9, 12, 14, 16, 19],
    'CN1[C@@H](O)CCC[C@H]1O': [9, 14, 16, 18],
    'CNC(=O)Cn1cccn1': [10, 13, 14, 16, 17, 18],
    'CNCC1CC(O)C1': [8, 11, 12, 14, 17, 18],
    'CNCCCOC(C)C': [8, 11, 12, 14, 16, 18],
    'CNC[C@H](O)CN(C)C': [8, 11, 12, 15, 16, 18],
    'CNc1ccccc1N=O': [10, 13, 15, 16, 17],
    'COC(=O)CCN': [6, 9, 11, 13],
    'COC(=O)[C@]1(C)CCOC1': [10, 13, 16, 18, 20],
    'COC(C#CC=O)OC': [11, 12],
    'COCCN1C=NCCC1': [10, 13, 15, 17, 18, 20, 22],
    'COCOCOCOC': [8, 11, 13],
    'COC[C@H](N)C#N': [6, 9, 12],
    'COc1cc(N)ccc1N': [10, 13, 14, 16, 17, 18],
    'C[C@@H]1CNCC1(C)C': [8, 12, 14, 15],
    'C[C@@H]1CO[C@H](O)[C@H](C)C1': [13, 18, 21],
    'C[C@@H]1C[C@H](O)C[C@@H]1N': [8, 12, 16, 19],
    'C[C@H](C=O)C(=O)N(C)C': [8, 13],
    'C[C@H](N)c1cnc[nH]1': [8, 12, 14, 15, 16],
    'C[C@H](O)CN': [4, 8, 9, 11],
    'Cc1cc1=O': [5, 8],
    'Cc1ccnc2c1C(=O)N2': [14, 15, 16],
    'Cc1n[nH]c(=O)c(N)c1N': [10, 13, 14, 16],
    'Cc1nnc(C)c(C)n1': [9, 12, 15],
    'N#CC1=CCC=C1': [7, 8, 10, 11],
    'NC(=O)c1ccccc1O': [12, 13, 14, 15, 16],
    'NC1=NON/C1=C(/N)N=O': [10, 12],
    'NC[C@H](O)CC1CC1': [8, 10, 13, 14, 16],
    'NNC(=O)C[C@H](N)C(=O)O': [9, 11, 12],
    'Nc1ncnc(N)n1': [8, 10],
    'Nc1nnc2nc[nH]n12': [10, 12, 13],
    'Nn1ccoc1=O': [7, 9, 10],
    'O=C1CCC(=O)O1': [7],
    'O=C=Cc1cccnc1': [9, 10, 11, 12, 13],
    'O=COCCCCO': [7, 8, 10, 12, 14, 16],
    'OCCNC1CCC1': [8, 9, 11, 13, 14, 17],
    'OCCNCCOCCO': [9, 10, 12, 14, 15, 17, 19, 21],
    'O[C@@H]1CN=CNC1': [7, 9, 11, 12, 13],
    'c1cnc2c(c1)CCO2': [10, 11, 12, 13],
}

results = defaultdict(list)

try:
    # Use chunking to read the large CSV file efficiently
    chunk_iter = pd.read_csv(
        'bde_model_methods/rdf_data_190531.csv.gz', 
        compression='gzip', 
        usecols=['molecule', 'bond_index', 'bond_type'],
        chunksize=100000
    )
    
    for chunk in chunk_iter:
        # Filter the chunk to only include molecules we are interested in
        filtered_chunk = chunk[chunk['molecule'].isin(problematic_data.keys())]
        
        for _, row in filtered_chunk.iterrows():
            smiles = row['molecule']
            bond_idx = row['bond_index']
            bond_type = row['bond_type']
            
            # Check if this bond_index was one of the invalid ones for this molecule
            if bond_idx in problematic_data.get(smiles, []):
                # Add to results, ensuring no duplicates
                if bond_type not in results[smiles]:
                    results[smiles].append(bond_type)

except Exception as e:
    print(f"Error reading or processing file: {e}")
    exit()

# Print the final compiled results
if not results:
    print("No matching records found for the invalid bond indices in the provided CSV.")
else:
    print("Found bond types for the invalid bond indices:")
    print("="*40)
    for smiles, bond_types in sorted(results.items()):
        print(f"SMILES: {smiles}")
        print(f"  -> Invalid Bond Types Found: {', '.join(bond_types)}")
        print("-" * 20)
