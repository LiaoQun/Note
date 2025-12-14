# Implementation Plan

本計畫將重構工作拆解為三個階段，每個步驟專注於單一目標，並包含明確的驗收方式。
規則強調：
✓ 模組化、多檔案
✓ 禁止單一巨型檔案
✓ 一步一驗收
✓ 不准自己擴大 scope
✓ **程式碼需包含 Tensor 維度註解**
✓ 什麼時候必須更新文件
✓ 請將重構的檔案更新再獨立的資料夾中，並且全部重寫不要import舊的alfabet或是bde_model_methods檔案

## Phase 1: 數據與特徵工程 (Data & Featurization)

### Step 1: 建立 Tokenizer 與 Featurizer
* **做什麼 (What)**:
    * **`AtomFeaturizer`**: 根據 `alfabet/preprocessor.py`，實作一個函數，對 RDKit Atom 物件生成包含以下特徵的字串：
        * `(atom.GetSymbol(), atom.GetNumRadicalElectrons(), atom.GetFormalCharge(), atom.GetChiralTag(), atom.GetIsAromatic(), get_ring_size(atom, max_size=6), atom.GetDegree(), atom.GetTotalNumHs(includeNeighbors=True))`
    * **`BondFeaturizer`**: 根據 `alfabet/preprocessor.py`，實作一個函數，對 RDKit Bond 物件生成特徵字串。
        * **關鍵點**: 需包含 `flipped: bool` 參數。`flipped=False` 時，字串為 `"{begin_atom}-{end_atom}"`；`flipped=True` 時為 `"{end_atom}-{begin_atom}"`。
        * 字串其餘部分包含 `(bond.GetBondType(), bond.GetIsConjugated())` 以及環資訊。
    * **`Tokenizer`**: 實作一個類別，負責將原子/鍵的特徵字串映射為整數 ID。
        * **初始化**: 預設從 `@etc/preprocessor.json` 載入靜態詞彙表。
        * **動態擴充**: 需設計一個 `build_from_smiles(smiles_list)` 方法，能從新的 SMILES 列表動態生成或擴充詞彙表。
        * **保存**: `Tokenizer` 物件需提供 `save(filepath)` 方法，將更新後的詞彙表寫入指定的 JSON 檔案。
        * **OOV 處理**: 對於未知的特徵，應映射到一個固定的 `unk` token (ID: 1)。
* **怎麼驗 (How to Verify)**:
    * 單元測試：確認 `AtomFeaturizer` 對碳原子能生成正確的特徵字串。
    * 單元測試：確認 `BondFeaturizer` 對同一 C-O 鍵，`flipped` 參數不同時能生成兩個不同的特徵字串。
    * 單元測試：`Tokenizer` 能夠正確載入 JSON、保存 JSON，並將 OOV (Out-of-Vocabulary) 特徵正確地映射到 `unk` ID。

### Step 2: 構建 PyG Dataset
* **做什麼 (What)**: 
    * 繼承 PyG 的 `InMemoryDataset`。
    * 在 `process()` 中遍歷 SMILES 列表，利用 Step 1 的 `Featurizer` 和 `Tokenizer`，將每個分子轉換為一個 PyG `Data` 物件。
    * `Data` 物件需包含以下屬性，並註明形狀與型別：
        * `x`: `torch.LongTensor` of shape `[num_atoms]`. Atom 特徵的整數 ID。
        * `edge_index`: `torch.LongTensor` of shape `[2, num_edges]`. 邊的連接性。
        * `edge_attr`: `torch.LongTensor` of shape `[num_edges]`. Bond 特徵的整數 ID。
        * `y`: `torch.FloatTensor` of shape `[num_edges]`. 每條邊對應的 BDE 標籤。
        * `mask`: `torch.BoolTensor` of shape `[num_edges]`. 對於有 BDE 標籤的邊為 `True`，否則為 `False`。
* **怎麼驗 (How to Verify)**: 
    * 檢查點：使用 `DataLoader` 取出一個 Batch。
    * 驗證：`batch.edge_index` 形狀為 `[2, total_edges_in_batch]`。
    * 驗證：`batch.edge_attr` 形狀為 `[total_edges_in_batch]`，且其長度等於 `batch.edge_index` 的列數。
    * 驗證：`batch.x`, `batch.y`, `batch.mask` 的長度與 batch 中的節點/邊數量匹配，無維度不匹配報錯。

## Phase 2: 模型架構搭建 (Model Architecture)

### Step 3: 實作單層交互層 (BDEInteractionLayer)
* **做什麼 (What)**: 
    * 繼承 `MessagePassing` 類別。
    * 實作原版 `message_block` 邏輯：先執行 Edge Update (MLP)，再執行 Node Update (Message Passing)。
    * 加入 Residual Connections 和 Batch Normalization。
    * **(待釐清)** `message_block` 的具體數學細節與 `forward` 函數的回傳值。
* **怎麼驗 (How to Verify)**: 
    * 假資料測試：輸入隨機生成的 `x` (節點特徵) 和 `edge_attr` (邊特徵)。
    * 驗證：通過一層後，輸出 Tensor 的形狀應與輸入完全一致 (Size check)。

### Step 4: 組裝完整模型 (BDEModel)
* **做什麼 (What)**: 
    * 定義 `Embedding` 層，用於 `x` (Atom ID) 和 `edge_attr` (Bond ID)。
    * **(待釐清)** `BondMean` Embedding 的具體用途與計算邏輯。
    * 堆疊 N 層 `BDEInteractionLayer`。
    * **輸出層**: 實作一個 `Dense` MLP，其輸入為最後一層交互層輸出的**邊特徵 (`edge_attr`)**。
* **怎麼驗 (How to Verify)**: 
    * 前向傳播測試：輸入 Step 2 產生的一個 Batch Data。
    * 驗證：`model(data)` 輸出形狀應為一維向量 `[total_num_edges_in_batch]`。
    * **精確驗證**: `model(data).shape[0] == data.edge_attr.shape[0]` 必須成立。

## Phase 3: 訓練與驗收 (Training & Validation)

### Step 5: 實作 Loss Mask 機制與 Training Loop
* **做什麼 (What)**: 
    * 撰寫 PyTorch 訓練迴圈。
    * **Loss 計算**: Loss 函數應為 MAE。計算時，利用 `data.mask` 過濾出有標籤的預測值與真實值。
        * **實作公式**: `loss = F.l1_loss(predictions[data.mask], data.y[data.mask])`
* **怎麼驗 (How to Verify)**: 
    * Overfit 測試：使用 10 筆假資料進行訓練。
    * 驗證：Loss 在 50-100 個 Epoch 內應大幅下降趨近於 0。

### Step 6: 完整訓練與對齊
* **做什麼 (What)**: 
    * 載入完整的 CSV 訓練數據集。
    * **數據切分**:
        * 在讀取數據後，設定一個固定的 `random_seed`。
        * 將數據集**隨機打亂**並切分為 訓練集 (80%)、驗證集 (10%)、測試集 (10%)。
    * 執行完整訓練流程，並記錄 Validation Set 的 MAE。
* **怎麼驗 (How to Verify)**: 
    * 基準比對：訓練後的 Validation MAE 應接近原論文水準 (約 1-2 kcal/mol)。
    * 確認無邏輯錯誤導致梯度消失或爆炸。
