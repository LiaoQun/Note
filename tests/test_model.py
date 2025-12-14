import pytest
import torch
from torch_geometric.data import DataLoader, Data
import shutil
import tempfile
import os
import json

# Import fixtures from previous test file
from tests.test_dataset import mock_smiles_data, mock_tokenizer, temp_dataset_dir

from src.data.dataset import BDEDataset
from src.models.mpnn import BDEInteractionLayer, BDEModel

ATOM_FEATURES = 128 # Matching the original Keras implementation

@pytest.fixture
def sample_data_batch(mock_smiles_data, mock_tokenizer, temp_dataset_dir):
    """Provides a batch of data from the BDEDataset."""
    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, tokenizer=mock_tokenizer)
    dataloader = DataLoader(dataset, batch_size=2)
    return next(iter(dataloader))

def test_bde_interaction_layer_shape(sample_data_batch):
    """
    Tests the shape consistency of the BDEInteractionLayer's forward pass.
    """
    batch = sample_data_batch
    layer = BDEInteractionLayer(atom_features=ATOM_FEATURES)

    # Mock initial atom and bond states from embeddings
    atom_embedding = torch.nn.Embedding(100, ATOM_FEATURES) # Dummy num_classes
    bond_embedding = torch.nn.Embedding(100, ATOM_FEATURES) # Dummy num_classes

    initial_atom_state = atom_embedding(batch.x)
    initial_bond_state = bond_embedding(batch.edge_attr)
    
    num_atoms = initial_atom_state.shape[0]
    num_edges = initial_bond_state.shape[0]

    # Pass through the layer
    new_atom_state, new_bond_state = layer(initial_atom_state, batch.edge_index, initial_bond_state)

    # Verify that the output shapes are identical to the input shapes
    assert new_atom_state.shape == (num_atoms, ATOM_FEATURES)
    assert new_bond_state.shape == (num_edges, ATOM_FEATURES)

def test_bde_model_forward_pass_shape(sample_data_batch, mock_tokenizer):
    """
    Tests the forward pass of the full BDEModel and verifies the output shape.
    """
    batch = sample_data_batch
    
    # Get vocabulary sizes from the tokenizer
    num_atom_classes = mock_tokenizer.atom_num_classes + 1 # +1 for potential new classes from build
    num_bond_classes = mock_tokenizer.bond_num_classes + 1
    
    # Instantiate the full model
    model = BDEModel(
        num_atom_classes=num_atom_classes,
        num_bond_classes=num_bond_classes,
        atom_features=ATOM_FEATURES,
        num_messages=6 # As per the original implementation file name
    )

    # Perform a forward pass
    output = model(batch)

    # Verify output shape is a 1D tensor of length num_edges
    assert output.ndim == 1
    # Precise validation from the implementation plan
    assert output.shape[0] == batch.edge_index.shape[1]
    assert output.shape[0] == batch.edge_attr.shape[0]
    
def test_bde_model_with_single_data_object(mock_smiles_data, mock_tokenizer, temp_dataset_dir):
    """
    Tests that the BDEModel can process a single Data object in addition to a Batch object.
    """
    dataset = BDEDataset(root=temp_dataset_dir, smiles_data=mock_smiles_data, tokenizer=mock_tokenizer)
    single_data = dataset.get(0) # Get the first data object (CCO)

    # Get vocabulary sizes from the tokenizer
    num_atom_classes = mock_tokenizer.atom_num_classes + 1
    num_bond_classes = mock_tokenizer.bond_num_classes + 1

    model = BDEModel(
        num_atom_classes=num_atom_classes,
        num_bond_classes=num_bond_classes,
        atom_features=ATOM_FEATURES,
        num_messages=2 # Fewer messages for a quicker test
    )

    # Perform a forward pass
    output = model(single_data)

    # Verify output shape
    assert output.ndim == 1
    assert output.shape[0] == single_data.edge_index.shape[1]