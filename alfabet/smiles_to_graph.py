import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs

# ==============================================================================
# 檢核點二：建立特徵轉換器 (The Converter)
#
# 任務：撰寫 smiles_to_graph 函數。
# 目標：輸入 SMILES，輸出符合 PyTorch 格式的 Tensor (x, edge_index, edge_attr)。
#       需確保數值邏輯與原作一致。
#
# 這個腳本會實作一個 `smiles_to_graph` 函數，它將取代舊的 nfp/tensorflow
# 預處理流程。
# ==============================================================================

# --- 從 alfabet/preprocessor.py 和 nfp 原始碼中提取的函數 ---

def get_ring_size(obj, max_size=12):
    """
    檢查一個原子或鍵是否在環中，並返回環的大小。
    這是從 nfp 早期版本中找到的輔助函數，因為新版沒有公開。
    """
    if not obj.IsInRing():
        return 0
    else:
        for i in range(max_size, 2, -1): # RDKit's IsInRingSize is slow, so iterate backwards
            if obj.IsInRingSize(i):
                return i
        return 'max' # Should not happen if max_size is large enough

def atom_featurizer(atom):
    """
    從 `alfabet/preprocessor.py` 複製而來。
    返回代表原子類型的字串雜湊。
    """
    return str(
        (
            atom.GetSymbol(),
            atom.GetNumRadicalElectrons(),
            atom.GetFormalCharge(),
            atom.GetChiralTag(),
            atom.GetIsAromatic(),
            get_ring_size(atom, max_size=6),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
    )


def bond_featurizer(bond, flipped=False):
    """
    從 `alfabet/preprocessor.py` 複製而來。
    返回代表鍵類型的字串。
    """
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
        )
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()))
        )

    btype = str((bond.GetBondType(), bond.GetIsConjugated()))
    ring = "R{}".format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ""

    return " ".join([atoms, btype, ring]).strip()


def smiles_to_graph(smiles: str, atom_dict: dict, bond_dict: dict):
    """
    將 SMILES 字串轉換為 PyTorch Geometric 的圖數據。
    這個函數重現了 `nfp.SmilesBondIndexPreprocessor` 的核心邏輯。

    Args:
        smiles (str): 輸入的 SMILES 字串。
        atom_dict (dict): 從 preprocessor.json 載入的原子特徵到整數的映射。
        bond_dict (dict): 從 preprocessor.json 載入的鍵特徵到整數的映射。

    Returns:
        torch_geometric.data.Data: PyG 的圖數據物件，包含 x, edge_index, edge_attr。
    """
    from torch_geometric.data import Data

    mol = MolFromSmiles(smiles)
    mol = AddHs(mol) # 原始 preprocessor 設定 explicit_hs=True

    atom_features = []
    for atom in mol.GetAtoms():
        feature_str = atom_featurizer(atom)
        # 原始碼中，如果特徵不在字典裡，會被映對到 1 (代表 'UNK')
        atom_features.append(atom_dict.get(feature_str, 1))

    # 原始的 nfp 預處理器會為每個化學鍵建立兩個有向邊
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        # 正向邊
        edge_indices.append((start, end))
        feature_str_fwd = bond_featurizer(bond, flipped=False)
        edge_attrs.append(bond_dict.get(feature_str_fwd, 1))

        # 反向邊
        edge_indices.append((end, start))
        feature_str_rev = bond_featurizer(bond, flipped=True)
        edge_attrs.append(bond_dict.get(feature_str_rev, 1))

    x = torch.tensor(atom_features, dtype=torch.long).unsqueeze(1)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == '__main__':
    # 在實際應用中，這些字典應該從 preprocessor.json 檔案載入。
    # 這裡為了方便演示，我們先用空字典，這樣所有特徵都會被映射到 'UNK' (1)。
    # 在下一個檢核點，我們會從檔案載入它們。
    mock_atom_dict = {}
    mock_bond_dict = {}

    test_smiles = 'CCO' # 乙醇
    graph_data = smiles_to_graph(test_smiles, mock_atom_dict, mock_bond_dict)

    print(f"--- 檢核點二：SMILES to Graph 轉換結果 (SMILES: {test_smiles}) ---")
    print(graph_data)
    print("\n說明:")
    print(f"x (節點/原子特徵): shape={graph_data.x.shape}, dtype={graph_data.x.dtype}")
    print("   - 每個原子一個節點，特徵是代表其類型的整數 ID。")
    print(f"edge_index (圖連接性): shape={graph_data.edge_index.shape}, dtype={graph_data.edge_index.dtype}")
    print("   - [2, num_edges] 的矩陣，表示有向邊的起點和終點。")
    print(f"edge_attr (邊/鍵特徵): shape={graph_data.edge_attr.shape}, dtype={graph_data.edge_attr.dtype}")
    print("   - 每個邊一個特徵，是代表其鍵類型的整數 ID。")
