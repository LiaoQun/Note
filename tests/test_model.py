import pytest
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import shutil
import tempfile
import os
import json

# Import necessary modules from src
from src.config import DataConfig
from src.data.dataset import BDEDataset
from src.features import get_featurizer # Import the factory function
from src.features.featurizer import Tokenizer, TokenFeaturizer
from src.features.chemprop import ChemPropFeaturizer
from src.models.mpnn import BDEInteractionLayer, BDEModel

ATOM_FEATURES = 128 # Matching the original Keras implementation

@pytest.fixture
def mock_smiles_data():
    """Provides sample SMILES data for dataset creation."""
    # Using CCO for molecule and bond indices (0,1) C-C, (1,2) C-O
    return [
        ("CCO", {(0,1): 75.0, (1,2): 80.0}), # C-C and C-O bond BDEs
        ("C=O", {(0,1): 120.0}) # C=O bond BDE
    ]

@pytest.fixture
def temp_dataset_dir():
    """Creates a temporary directory for datasets and cleans up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def data_config_token(temp_dataset_dir):
    """Provides a DataConfig for TokenFeaturizer, including a temporary vocab path."""
    # Create a dummy vocab file for testing TokenFeaturizer
    vocab_filepath = os.path.join(temp_dataset_dir, "vocab.json")
    dummy_tokenizer = Tokenizer()
    dummy_tokenizer.build_from_smiles(["CCO", "C=O"]) # Build a minimal vocab
    dummy_tokenizer.save(vocab_filepath)

    return DataConfig(featurizer_type="TokenFeaturizer", vocab_path=vocab_filepath)

@pytest.fixture
def data_config_chemprop():
    """Provides a DataConfig for ChemPropFeaturizer."""
    return DataConfig(featurizer_type="ChemPropFeaturizer")


@pytest.fixture
def sample_batch_token(mock_smiles_data, data_config_token, temp_dataset_dir):
    """Provides a batch of data using TokenFeaturizer."""
    featurizer = get_featurizer(data_config_token)
    
    # BDEDataset expects vocab to be built if TokenFeaturizer
    featurizer.prepare_data([smi for smi, _ in mock_smiles_data])
    
    dataset = BDEDataset(root=os.path.join(temp_dataset_dir, "token"), smiles_data=mock_smiles_data, featurizer=featurizer)
    dataloader = DataLoader(dataset, batch_size=2)
    return next(iter(dataloader)), featurizer

@pytest.fixture
def sample_batch_chemprop(mock_smiles_data, data_config_chemprop, temp_dataset_dir):
    """Provides a batch of data using ChemPropFeaturizer."""
    featurizer = get_featurizer(data_config_chemprop)
    dataset = BDEDataset(root=os.path.join(temp_dataset_dir, "chemprop"), smiles_data=mock_smiles_data, featurizer=featurizer)
    dataloader = DataLoader(dataset, batch_size=2)
    return next(iter(dataloader)), featurizer

def test_bde_interaction_layer_shape(sample_batch_token):
    """
    Tests the shape consistency of the BDEInteractionLayer's forward pass.
    Uses TokenFeaturizer (discrete) for batch creation.
    """
    batch, featurizer = sample_batch_token
    layer = BDEInteractionLayer(atom_features=ATOM_FEATURES)

    # Initial states for atom and bond (from embeddings)
    initial_atom_state = torch.rand(batch.x.shape[0], ATOM_FEATURES) # Mock embedded atom features
    initial_bond_state = torch.rand(batch.edge_attr.shape[0], ATOM_FEATURES) # Mock embedded bond features
    
    num_atoms = initial_atom_state.shape[0]
    num_edges = initial_bond_state.shape[0]

    # Pass through the layer
    new_atom_state, new_bond_state = layer(initial_atom_state, batch.edge_index, initial_bond_state)

    # Verify that the output shapes are identical to the input shapes
    assert new_atom_state.shape == (num_atoms, ATOM_FEATURES)
    assert new_bond_state.shape == (num_edges, ATOM_FEATURES)

@pytest.mark.parametrize("input_type", ["token", "chemprop"])
def test_bde_model_forward_pass_shape(input_type, request):
    """
    Tests the forward pass of the full BDEModel for both discrete and continuous inputs.
    """
    if input_type == "token":
        batch, featurizer = request.getfixturevalue("sample_batch_token")
    else: # chemprop
        batch, featurizer = request.getfixturevalue("sample_batch_chemprop")
    
    # Instantiate the full model
    model = BDEModel(
        atom_input_dim=featurizer.atom_dim,
        bond_input_dim=featurizer.bond_dim,
        atom_features=ATOM_FEATURES,
        num_messages=3, # Fewer messages for quicker test
        inputs_are_discrete=featurizer.is_discrete
    )

    # Perform a forward pass
    output = model(batch)

    # Verify output shape is a 1D tensor of length num_edges
    assert output.ndim == 1
    assert output.shape[0] == batch.edge_index.shape[1]
    assert output.shape[0] == batch.edge_attr.shape[0]