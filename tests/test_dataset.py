import os
import shutil
import tempfile
import pytest
import torch
from torch_geometric.data import DataLoader
from rdkit import Chem

from src.features.featurizer import Tokenizer
from src.data.dataset import BDEDataset

@pytest.fixture
def mock_tokenizer():
    """Provides a simple tokenizer for testing."""
    # Create a dummy vocabulary file
    vocab_data = {
        "atom_tokenizer": {
            "_data": {
                "unk": 1,
                "('C', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 4, 3)": 2, # CH3 carbon
                "('C', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 4, 2)": 3, # CH2 carbon
                "('O', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 2, 1)": 4, # OH oxygen
            },
            "num_classes": 4
        },
        "bond_tokenizer": {
            "_data": {
                "unk": 1,
                "C-C (rdkit.Chem.rdchem.BondType.SINGLE, False)": 2,
                "C-O (rdkit.Chem.rdchem.BondType.SINGLE, False)": 3,
                "O-C (rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False)": 4, # dummy for flipped O-C
            },
            "num_classes": 4
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(vocab_data, f)
        filepath = f.name
    
    tokenizer = Tokenizer(vocab_filepath=filepath)
    os.remove(filepath) # Clean up dummy file
    return tokenizer

@pytest.fixture
def mock_smiles_data():
    """Provides mock SMILES data with BDE labels."""
    # Ethanol: CCO
    # C0-C1 bond, C1-O2 bond
    # Adding Hs: C0(H3)-C1(H2)-O2(H)
    return [
        ("CCO", { (0, 1): 88.0, (1, 2): 85.0 }), # C-C BDE, C-O BDE
        ("CCC", { (0, 1): 90.0 }), # C-C BDE only for first bond
        ("C", {}) # Single atom, no bonds
    ]

@pytest.fixture
def temp_dataset_dir():
    """Creates a temporary directory for dataset processing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir) # Clean up directory

def test_bde_dataset_init_and_len(mock_smiles_data, mock_tokenizer, temp_dataset_dir):
    """Tests if the dataset initializes and reports correct length."""
    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, tokenizer=mock_tokenizer)
    assert len(dataset) == 2 # "C" molecule should be skipped as it has no bonds

def test_bde_dataset_data_object_structure(mock_smiles_data, mock_tokenizer, temp_dataset_dir):
    """Tests the structure and types of the PyG Data objects."""
    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, tokenizer=mock_tokenizer)
    
    # Test CCO molecule (first in mock_smiles_data)
    data = dataset.get(0) 
    
    # Verify x (atom features)
    # CCO after AddHs has C(idx=0), C(idx=1), O(idx=2) and 6 H atoms. Total 9 atoms.
    # The featurizer.py atom_featurizer considers the original number of heavy atoms.
    # Let's count the number of atoms for CCO after AddHs
    mol_cco = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    num_atoms_cco = mol_cco.GetNumAtoms()
    assert data.x.shape == (num_atoms_cco,)
    assert data.x.dtype == torch.long

    # Verify edge_index (connectivity)
    # CCO has 2 heavy-atom bonds. After adding explicit Hs, it has more.
    # C0-C1, C1-O2, C0-H, C0-H, C0-H, C1-H, C1-H, O2-H
    # All these bonds are doubled (forward/backward)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.dtype == torch.long
    num_edges_cco = mol_cco.GetNumBonds() * 2 # Each bond contributes two directed edges
    assert data.edge_index.shape[1] == num_edges_cco
    
    # Verify edge_attr (bond features)
    assert data.edge_attr.shape == (num_edges_cco,)
    assert data.edge_attr.dtype == torch.long
    assert len(data.edge_attr) == data.edge_index.shape[1]

    # Verify y (BDE labels)
    # For CCO, mock_smiles_data has BDE for (0,1) and (1,2)
    assert data.y.shape == (num_edges_cco,)
    assert data.y.dtype == torch.float
    
    # Verify mask (loss mask)
    assert data.mask.shape == (num_edges_cco,)
    assert data.mask.dtype == torch.bool

    # Check some specific values for CCO (0,1) and (1,2) bonds
    # C0-C1 BDE is 88.0, C1-O2 BDE is 85.0
    # The bonds are processed sequentially by GetBonds()
    # It's hard to assert specific indices without knowing RDKit's internal bond ordering.
    # Instead, check if the values exist in y for masked edges.
    assert (88.0 in data.y[data.mask]) and (85.0 in data.y[data.mask])
    assert torch.sum(data.mask).item() == 4 # Two bonds, each in two directions
    
    # Test CCC molecule (second in mock_smiles_data)
    data_ccc = dataset.get(1)
    mol_ccc = Chem.AddHs(Chem.MolFromSmiles("CCC"))
    num_atoms_ccc = mol_ccc.GetNumAtoms()
    assert data_ccc.x.shape == (num_atoms_ccc,)
    
    num_edges_ccc = mol_ccc.GetNumBonds() * 2
    assert data_ccc.edge_index.shape[1] == num_edges_ccc
    assert data_ccc.edge_attr.shape == (num_edges_ccc,)
    assert data_ccc.y.shape == (num_edges_ccc,)
    assert data_ccc.mask.shape == (num_edges_ccc,)
    
    # CCC has BDE for (0,1) only
    assert (90.0 in data_ccc.y[data_ccc.mask])
    assert torch.sum(data_ccc.mask).item() == 2 # One bond, two directions

def test_bde_dataset_dataloader(mock_smiles_data, mock_tokenizer, temp_dataset_dir):
    """Tests if DataLoader can create batches with correct shapes."""
    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, tokenizer=mock_tokenizer)
    
    # Use a small batch size to ensure multiple graphs are batched
    dataloader = DataLoader(dataset, batch_size=2) 
    batch = next(iter(dataloader))
    
    # Check batch object types and shapes
    assert isinstance(batch, Data)
    
    assert batch.x.dtype == torch.long
    assert batch.edge_index.dtype == torch.long
    assert batch.edge_attr.dtype == torch.long
    assert batch.y.dtype == torch.float
    assert batch.mask.dtype == torch.bool
    assert batch.batch.dtype == torch.long # Batch index for each node
    
    # Check general shapes (exact numbers depend on the batching mechanism and specific molecules)
    assert batch.x.ndim == 1
    assert batch.edge_index.shape[0] == 2
    assert batch.edge_attr.ndim == 1
    assert batch.y.ndim == 1
    assert batch.mask.ndim == 1
    
    # Crucial check from plan: edge_attr length == edge_index columns
    assert len(batch.edge_attr) == batch.edge_index.shape[1]

    # Verify that the total number of masked edges is correct
    assert torch.sum(batch.mask).item() == 4 + 2 # CCO (4 masked) + CCC (2 masked)
    assert 88.0 in batch.y[batch.mask]
    assert 85.0 in batch.y[batch.mask]
    assert 90.0 in batch.y[batch.mask]