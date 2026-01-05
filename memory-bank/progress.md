# Implementation Progress

## Phase 1: 數據與特徵工程 (Data & Featurization) - **已完成**

### Step 1: 建立 Tokenizer 與 Featurizer - **已完成**
*   **狀態**: 已完成。
*   **驗證**: `src/features/featurizer.py` 中 `atom_featurizer` 和 `bond_featurizer` 函式已按要求實作，包含 `flipped` 參數。`Tokenizer` 類別已實作載入、動態擴充 (`build_from_smiles`)、保存和 OOV 處理。

### Step 2: 構建 PyG Dataset - **已完成**
*   **狀態**: 已完成。
*   **驗證**: `src/data/dataset.py` 中的 `BDEDataset` 已繼承 `InMemoryDataset`。`process` 方法能將 SMILES 轉換為 PyG `Data` 物件，並包含 `x`, `edge_index`, `edge_attr`, `y`, `mask` 等屬性，其形狀和型別均符合要求。已處理雙向邊的特徵和標籤。

## Phase 2: 模型架構搭建 (Model Architecture) - **已完成**

### Step 3: 實作單層交互層 (BDEInteractionLayer) - **已完成**
*   **狀態**: 已完成。
*   **驗證**: `src/models/mpnn.py` 中的 `BDEInteractionLayer` 已繼承 `MessagePassing`，並實作了 Edge Update 和 Node Update 的邏輯，包含殘差連接和 Batch Normalization。Tensor 維度註解也已添加。

### Step 4: 組裝完整模型 (BDEModel) - **已完成**
*   **狀態**: 已完成。
*   **驗證**: `src/models/mpnn.py` 中的 `BDEModel` 已定義 `Embedding` 層，堆疊了多層 `BDEInteractionLayer`，並實作了基於邊特徵的輸出層，包含了 `BondMean` Embedding 的邏輯。Tensor 維度註解也已添加。

## Phase 3: 訓練與驗收 (Training & Validation) - **已完成**

### Step 5: 實作 Loss Mask 機制與 Training Loop - **已完成**
*   **狀態**: 已完成。
*   **驗證**: `src/training/trainer.py` 中的 `Trainer` 類別已實現 PyTorch 訓練迴圈，並正確使用了 `data.mask` 來計算 MAE loss。包含了驗證迴圈和模型儲存機制。

### Step 6: 完整訓練與對齊 - **已完成**
*   **狀態**: 已完成。
*   **進度**: 數據集切分 (train/val/test 80/10/10 比例，固定隨機種子) 已在 `main.py` 和 `trainer.py` 中實作。訓練流程已集成。Parity Plot 的功能已實現並在 `trainer.py` 的 `evaluate` 方法中整合，將同時顯示訓練集、驗證集和測試集的預測結果與統計資訊。模型的性能基準比對已在其他平台確認，證明與原論文水準一致。

---

# 未來開發階段 (Future Development Phases)

## Phase 1: 基礎建設與立即改善 (Foundational Improvements)
- [x] **程式碼重構與清理**: 移除重複的 `evaluate` 函數。
- [x] **腳本優化**:
    - [x] 優化 `create_training_template.py` 引入命令行參數。
    - [x] 優化 `predict.py` 簡化模型加載邏輯並提供多種輸出格式選項。
- [x] **數據預處理**: 在進行任何特徵提取前，應先將 SMILES 字符串標準化 (Canonicalization)。
- [x] **實驗管理與輸出**:
    - [x] 儲存完整的預測結果（包含原始 SMILES 和碎片資訊）。
    - [x] 儲存運行配置，將當次使用的 `config.json` 複製並保存到該次運行的目錄中。

## Phase 2: 擴展性與穩健性 (Scalability & Robustness)
- [ ] **數據處理與擴充**:
    - [ ] 支援超大型數據集：考慮將 `BDEDataset` 從 `InMemoryDataset` 轉換為標準的 PyG `Dataset`。
    - [ ] 多重數據源輸入：擴展數據加載功能，支持從多個檔案或多種格式（如 CSV, SDF, JSON）讀取數據。
- [ ] **實驗管理與 ML Ops**:
    - [ ] 集成實驗追蹤平台（如 MLflow, TensorBoard, 或 W&B）。
    - [ ] 實施模型版本管理。
- [ ] **測試與品質保證**:
    - [ ] 擴充單元測試覆蓋率。
    - [ ] 建立整合測試。
    - [ ] 導入靜態分析與 Linter。

## Phase 3: 高級功能與性能優化 (Advanced Features & Optimization)
- [ ] **實驗管理與 ML Ops**:
    - [ ] 集成超參數優化 (HPO) 框架（如 Optuna 或 Ray Tune）。
- [ ] **數據處理與擴充**:
    - [ ] 數據增強 (Data Augmentation)：導入如 SMILES 隨機化等策略。
- [ ] **模型與特徵工程**:
    - [ ] 模塊化 Featurizer。
    - [ ] 支持多種 GNN 骨幹。
- [ ] **性能優化**:
    - [ ] 整合大規模計算優化（更高效的批次處理）。
    - [ ] 集成 `torch.compile`。
    - [ ] 實現多 GPU 訓練。
- [ ] **推論與腳本**:
    - [ ] 完善 BDE 預測器。
    - [ ] 建立統一的 CLI 入口。
    - [ ] 專案打包。
