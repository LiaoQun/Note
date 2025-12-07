import torch
import json
import pooch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# ==============================================================================
# 檢核點三：數據管道 (The Pipeline)
#
# 任務：建立 Dataset 與 DataLoader。
# 目標：處理 Graph Batching (將小圖拼成大圖)，確保數據能批量送入 GPU。
#
# 這個腳本會：
# 1. 下載並載入原始的 preprocessor.json 檔案。
# 2. 建立一個 PyTorch Geometric Dataset。
# 3. 使用 DataLoader 將多個圖打包成一個批次 (Batch)。
# ==============================================================================

# --- 從檢核點二複製過來的函數 ---

def get_ring_size(obj, max_size=12):
    if not obj.IsInRing():
        return 0
    else:
        for i in range(max_size, 2, -1):
            if obj.IsInRingSize(i):
                return i
        return 'max'

def atom_featurizer(atom):
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
    mol = MolFromSmiles(smiles)
    mol = AddHs(mol)

    atom_features = []
    for atom in mol.GetAtoms():
        feature_str = atom_featurizer(atom)
        atom_features.append(atom_dict.get(feature_str, 1)) # 1 for 'UNK'

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        edge_indices.append((start, end))
        feature_str_fwd = bond_featurizer(bond, flipped=False)
        edge_attrs.append(bond_dict.get(feature_str_fwd, 1))

        edge_indices.append((end, start))
        feature_str_rev = bond_featurizer(bond, flipped=True)
        edge_attrs.append(bond_dict.get(feature_str_rev, 1))

    x = torch.tensor(atom_features, dtype=torch.long).unsqueeze(1)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# --- 新增的 Dataset 和 DataLoader 實作 ---

class BdeDataset(Dataset):
    """
    一個 PyTorch Geometric Dataset，用於處理 SMILES 字串。
    """
    def __init__(self, smiles_list: list, atom_dict: dict, bond_dict: dict):
        super().__init__()
        self.smiles_list = smiles_list
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        """
        根據索引獲取一個 SMILES，並將其轉換為 PyG 的 Data 物件。
        """
        smiles = self.smiles_list[idx]
        data = smiles_to_graph(smiles, self.atom_dict, self.bond_dict)
        return data


if __name__ == '__main__':
    # 1. 下載並載入 preprocessor.json
    # 這些資訊來自 alfabet/__init__.py 和 alfabet/preprocessor.py
    _model_tag = "v0.1.1"
    _model_files_baseurl = f"https://github.com/pstjohn/alfabet-models/releases/download/{_model_tag}/"
    preprocessor_path = pooch.retrieve(
        _model_files_baseurl + "preprocessor.json",
        known_hash="412d15ca4d0e8b5030e9b497f566566922818ff355b8ee677a91dd23696878ac",
    )

    with open(preprocessor_path, 'r') as f:
        preprocessor_data = json.load(f)

    # 從 JSON 中提取原子和鍵的特徵映射字典
    atom_dict = preprocessor_data['atom_class_map']
    bond_dict = preprocessor_data['bond_class_map']

    print(f"--- 檢核點三：數據管道 (The Pipeline) ---")
    print(f"成功從 {preprocessor_path} 載入特徵字典。")
    print(f"原子特徵數量: {len(atom_dict)}")
    print(f"鍵特徵數量: {len(bond_dict)}\n")

    # 2. 建立 Dataset
    test_smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O'] # 乙醇, 苯, 醋酸
    dataset = BdeDataset(test_smiles_list, atom_dict, bond_dict)

    print(f"已建立 BdeDataset，包含 {len(dataset)} 個分子。")
    print(f"第一個分子 (SMILES: '{test_smiles_list[0]}') 的圖數據:")
    print(f"  {dataset[0]}\n")

    # 3. 建立 DataLoader
    # batch_size=3 會將上面 3 個分子打包成一個批次
    data_loader = DataLoader(dataset, batch_size=3, shuffle=False)

    print("已建立 DataLoader，準備從中提取一個批次...")
    # 從 loader 中取出第一個 (也是唯一一個) 批次
    batch = next(iter(data_loader))

    print("\n--- DataLoader 批次處理結果 ---")
    print("批次物件 (Batch) 內容:")
    print(batch)
    print("\n說明:")
    print(f"x (節點特徵): shape={batch.x.shape}")
    print("   - 所有分子的節點特徵被垂直堆疊在一起。")
    print(f"edge_index (圖連接性): shape={batch.edge_index.shape}")
    print("   - 所有分子的邊索引被合併，節點索引也已自動更新。")
    print(f"edge_attr (邊特徵): shape={batch.edge_attr.shape}")
    print("   - 所有分子的邊特徵被垂直堆疊在一起。")
    print(f"batch (批次索引): shape={batch.batch.shape}")
    print(f"   - 內容: {batch.batch}")
    print("   - 這是一個關鍵張量，長度等於總節點數。它標示了每個節點屬於原始的哪個分子。")
    print("   - 例如，所有值為 0 的節點都來自第一個分子 (CCO)，值為 1 的來自第二個 (c1ccccc1)，以此類推。")
    print("   - 這個張量對於在模型中進行圖層級的操作 (如 global pooling) 至關重要。")

