import json
import os
import shutil
import tempfile
import pytest
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem

# Import necessary modules from src
from src.config import DataConfig
from src.features import get_featurizer # Import the factory function
from src.features.featurizer import Tokenizer, TokenFeaturizer
from src.features.chemprop_adapter import ChemPropPyGFeaturizer
from src.data.dataset import BDEDataset

@pytest.fixture
def mock_smiles_data():
    """Provides mock SMILES data with BDE labels."""
    # Ethanol: CCO (C0-C1, C1-O2 bonds)
    # Formal bond indices for CCO: 0 (C0-C1), 1 (C1-O2)
    # The (min_idx, max_idx) is based on atom indices
    return [
        ("CCO", { (0, 1): 88.0, (1, 2): 85.0 }), # BDE for C-C (0-1) and C-O (1-2)
        ("CCC", { (0, 1): 90.0 }), # C-C BDE only for first bond (0-1) in propane
        ("C", {}) # Single atom molecule, will become CH4 after AddHs
    ]

@pytest.fixture
def temp_dataset_dir():
    """Creates a temporary directory for dataset processing and cleans up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def data_config_token(mock_smiles_data, temp_dataset_dir):
    """Provides a DataConfig for TokenFeaturizer, including a temporary vocab path."""
    # Create a dummy vocab file for testing TokenFeaturizer
    vocab_filepath = os.path.join(temp_dataset_dir, "vocab.json")
    dummy_tokenizer = Tokenizer()
    dummy_tokenizer.build_from_smiles([smi for smi, _ in mock_smiles_data]) # Use mock_smiles_data as parameter
    dummy_tokenizer.save(vocab_filepath)

    return DataConfig(featurizer_type="TokenFeaturizer", vocab_path=vocab_filepath)

@pytest.fixture
def data_config_chemprop():
    """Provides a DataConfig for ChemPropFeaturizer."""
    return DataConfig(featurizer_type="ChemPropFeaturizer")


@pytest.fixture
def featurizer_token(data_config_token):
    """Provides a configured TokenFeaturizer instance."""
    return get_featurizer(data_config_token)

@pytest.fixture
def featurizer_chemprop(data_config_chemprop):
    """Provides a configured ChemPropPyGFeaturizer instance."""
    return get_featurizer(data_config_chemprop)


@pytest.mark.parametrize("featurizer_name", ["token", "chemprop"])
def test_bde_dataset_init_and_len(featurizer_name, request, mock_smiles_data, temp_dataset_dir):
    """Tests if the dataset initializes and reports correct length for different featurizers."""
    featurizer = request.getfixturevalue(f"featurizer_{featurizer_name}")
    
    # If TokenFeaturizer, ensure vocab is built (prepare_data is called implicitly by get_featurizer if vocab_path exists)
    if isinstance(featurizer, TokenFeaturizer):
        featurizer.prepare_data([smi for smi, _ in mock_smiles_data])
    
    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, featurizer=featurizer)
    assert len(dataset) == 3 # "C" molecule is processed into CH4 and is included


@pytest.mark.parametrize("featurizer_name", ["token", "chemprop"])
def test_bde_dataset_data_object_structure(featurizer_name, request, mock_smiles_data, temp_dataset_dir):
    """Tests the structure and types of the PyG Data objects for different featurizers."""
    featurizer = request.getfixturevalue(f"featurizer_{featurizer_name}")
    
    # If TokenFeaturizer, ensure vocab is built
    if isinstance(featurizer, TokenFeaturizer):
        featurizer.prepare_data([smi for smi, _ in mock_smiles_data])

    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, featurizer=featurizer)
    
    # Test CCO molecule (first in mock_smiles_data)
    data = dataset.get(0) 
    
    mol_cco = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    num_atoms_cco = mol_cco.GetNumAtoms()
    num_bonds_cco_rdkit = mol_cco.GetNumBonds()
    
    # Verify x (atom features)
    assert data.x.shape[0] == num_atoms_cco
    # For discrete features, the shape is just (num_atoms,), but for continuous it's (num_atoms, atom_dim)
    assert data.x.shape[1] == featurizer.atom_dim if not featurizer.is_discrete else True # if is_discrete, shape is (num_atoms)

    assert data.x.dtype == torch.long if featurizer.is_discrete else torch.float

    # Verify edge_index (connectivity)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape[1] == num_bonds_cco_rdkit * 2 # Each bond contributes two directed edges
    
    # Verify edge_attr (bond features)
    assert data.edge_attr.shape[0] == num_bonds_cco_rdkit * 2
    # Same logic for bond attributes: (num_edges,) for discrete, (num_edges, bond_dim) for continuous
    assert data.edge_attr.shape[1] == featurizer.bond_dim if not featurizer.is_discrete else True # if is_discrete, shape is (num_edges)
    assert data.edge_attr.dtype == torch.long if featurizer.is_discrete else torch.float

    # Verify y (BDE labels)
    assert data.y.shape[0] == num_bonds_cco_rdkit * 2
    assert data.y.dtype == torch.float
    
    # Verify mask (loss mask)
    assert data.mask.shape[0] == num_bonds_cco_rdkit * 2
    assert data.mask.dtype == torch.bool

    # Check some specific values for CCO (0,1) and (1,2) bonds
    # C0-C1 BDE is 88.0, C1-O2 BDE is 85.0
    # The number of masked bonds depends on how many bonds have labels
    expected_masked_bonds = 2 # C-C and C-O have labels
    assert torch.sum(data.mask).item() == expected_masked_bonds * 2 # Each labeled bond has 2 directed edges masked

    # Check if the BDE values are present in the masked labels
    masked_y_values = data.y[data.mask].unique().tolist()
    assert 88.0 in masked_y_values
    assert 85.0 in masked_y_values


@pytest.mark.parametrize("featurizer_name", ["token", "chemprop"])
def test_bde_dataset_dataloader(featurizer_name, request, mock_smiles_data, temp_dataset_dir):
    """Tests if DataLoader can create batches with correct shapes for different featurizers."""
    featurizer = request.getfixturevalue(f"featurizer_{featurizer_name}")
    
    # If TokenFeaturizer, ensure vocab is built
    if isinstance(featurizer, TokenFeaturizer):
        featurizer.prepare_data([smi for smi, _ in mock_smiles_data])

    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, featurizer=featurizer)
    
    # Use a small batch size to ensure multiple graphs are batched
    dataloader = DataLoader(dataset, batch_size=2) 
    batch = next(iter(dataloader))
    
    # Check batch object types and shapes
    assert isinstance(batch, Data)
    
    if featurizer.is_discrete:
        assert batch.x.dtype == torch.long
        assert batch.edge_attr.dtype == torch.long
        # For discrete inputs, x and edge_attr are 1D tensors of indices
        assert batch.x.ndim == 1
        assert batch.edge_attr.ndim == 1
    else:
        assert batch.x.dtype == torch.float
        assert batch.edge_attr.dtype == torch.float
        # For continuous inputs, x and edge_attr are 2D tensors of features
        assert batch.x.ndim == 2
        assert batch.edge_attr.ndim == 2
        assert batch.x.shape[1] == featurizer.atom_dim
        assert batch.edge_attr.shape[1] == featurizer.bond_dim

    assert batch.edge_index.dtype == torch.long
    assert batch.y.dtype == torch.float
    assert batch.mask.dtype == torch.bool
    assert batch.batch.dtype == torch.long # Batch index for each node
    
    # Check general shapes
    assert batch.edge_index.shape[0] == 2
    assert batch.y.ndim == 1
    assert batch.mask.ndim == 1
    
    # Crucial check: edge_attr length == edge_index columns
    # batch.edge_attr.shape[0] is the number of edges in the batch
    assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]

    # Verify that the total number of masked edges is correct (CCO + CCC)
    # CCO has 2 labeled bonds -> 4 masked edges (2 directed edges per bond)
    # CCC has 1 labeled bond -> 2 masked edges
    expected_total_masked_edges = 4 + 2 
    assert torch.sum(batch.mask).item() == expected_total_masked_edges
