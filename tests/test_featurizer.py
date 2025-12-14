import json
import os
import tempfile
import pytest
from rdkit import Chem

# Adjust the path to import from the new 'src' directory
from src.features.featurizer import (
    atom_featurizer,
    bond_featurizer,
    Tokenizer
)

@pytest.fixture
def sample_mol():
    """Provides an ethanol molecule for testing."""
    mol = Chem.MolFromSmiles("CCO")
    return Chem.AddHs(mol)

@pytest.fixture
def sample_vocab_file():
    """Creates a temporary vocabulary JSON file for testing."""
    vocab_data = {
        "atom_tokenizer": {
            "_data": {
                "unk": 1,
                "('C', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 1, 3)": 2
            },
            "num_classes": 2
        },
        "bond_tokenizer": {
            "_data": {
                "unk": 1,
                "C-C (rdkit.Chem.rdchem.BondType.SINGLE, False)": 2
            },
            "num_classes": 2
        }
    }
    # Use a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(vocab_data, f)
        filepath = f.name
    
    yield filepath
    
    # Cleanup the file
    os.remove(filepath)

def test_atom_featurizer(sample_mol):
    """Tests the atom featurizer function."""
    # Test a primary carbon atom (CH3)
    c1 = sample_mol.GetAtomWithIdx(1) # The second carbon atom
    # Expected: ('C', 0 formal charge, 0 chiral tag, not aromatic, not in ring, degree 2, 2 Hs)
    # The get_ring_size in the original code is complex. Let's rely on string representation from the implementation.
    # ('C', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 2, 2)
    
    # We will check the featurizer on the first Carbon atom
    c0 = sample_mol.GetAtomWithIdx(0)
    c0_features = atom_featurizer(c0)
    assert "'C'" in c0_features
    assert "0, 4, 3" in c0_features # Corrected: Degree 4 (1 C + 3 H), 3 Hs

def test_bond_featurizer(sample_mol):
    """Tests the bond featurizer for both normal and flipped directions."""
    # C-C bond
    bond = sample_mol.GetBondWithIdx(0)
    
    # Normal
    feat_normal = bond_featurizer(bond, flipped=False)
    assert feat_normal.startswith("C-C")
    assert "SINGLE, False" in feat_normal

    # Flipped
    feat_flipped = bond_featurizer(bond, flipped=True)
    assert feat_flipped.startswith("C-C") # Still C-C for this bond
    assert "SINGLE, False" in feat_flipped
    
    # C-O bond
    bond_co = sample_mol.GetBondBetweenAtoms(1, 2)
    feat_co_normal = bond_featurizer(bond_co, flipped=False)
    assert feat_co_normal.startswith("C-O")
    
    feat_co_flipped = bond_featurizer(bond_co, flipped=True)
    assert feat_co_flipped.startswith("O-C")
    assert feat_co_normal != feat_co_flipped

def test_tokenizer_init_empty():
    """Tests tokenizer initialization without a vocab file."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize_atom("any_feature") == 1
    assert tokenizer.tokenize_bond("any_feature") == 1
    assert tokenizer.atom_num_classes == 1
    assert tokenizer.bond_num_classes == 1

def test_tokenizer_load_from_json(sample_vocab_file):
    """Tests tokenizer initialization from a predefined vocab file."""
    tokenizer = Tokenizer(vocab_filepath=sample_vocab_file)
    
    # Test known atom
    known_atom_feat = "('C', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 1, 3)"
    assert tokenizer.tokenize_atom(known_atom_feat) == 2
    
    # Test unknown atom
    assert tokenizer.tokenize_atom("unknown_atom_feature") == 1
    
    # Test known bond
    known_bond_feat = "C-C (rdkit.Chem.rdchem.BondType.SINGLE, False)"
    assert tokenizer.tokenize_bond(known_bond_feat) == 2
    
    # Test unknown bond
    assert tokenizer.tokenize_bond("unknown_bond_feature") == 1

def test_tokenizer_build_from_smiles():
    """Tests the dynamic vocabulary building functionality."""
    tokenizer = Tokenizer() # Start with an empty tokenizer
    smiles_list = ["C", "CO"]
    
    tokenizer.build_from_smiles(smiles_list)
    
    # Check if vocab has been populated
    assert tokenizer.atom_num_classes > 1
    assert tokenizer.bond_num_classes > 1
    
    # Check if a known bond is now in the vocab
    # In "CO", there's a C-O bond
    mol = Chem.MolFromSmiles("CO")
    mol = Chem.AddHs(mol)
    bond = mol.GetBondWithIdx(0)
    bond_feat = bond_featurizer(bond, flipped=False)
    
    assert tokenizer.tokenize_bond(bond_feat) > 1 # Should not be 'unk'

def test_tokenizer_oov_handling():
    """Tests that unknown features are mapped to the 'unk' token ID."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize_atom("non_existent_atom_feature_string") == 1
    assert tokenizer.tokenize_bond("non_existent_bond_feature_string") == 1

def test_tokenizer_save_and_load():
    """Tests saving the tokenizer's vocab and reloading it."""
    smiles_list = ["CCO"]
    
    # Create and build a tokenizer
    tokenizer1 = Tokenizer()
    tokenizer1.build_from_smiles(smiles_list)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        save_path = f.name
    
    tokenizer1.save(save_path)
    
    # Create a new tokenizer and load from the saved file
    tokenizer2 = Tokenizer(vocab_filepath=save_path)
    
    # Check if the vocabularies are identical
    assert tokenizer1._atom_vocab == tokenizer2._atom_vocab
    assert tokenizer1._bond_vocab == tokenizer2._bond_vocab
    assert tokenizer1.atom_num_classes == tokenizer2.atom_num_classes
    assert tokenizer1.bond_num_classes == tokenizer2.bond_num_classes
    
    # Cleanup
    os.remove(save_path)
