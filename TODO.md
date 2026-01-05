# TODO List - 專案改進計畫

## 數據處理與擴充 (Data Processing & Extensions)
- [ ] **多重數據源輸入**: 擴展數據加載功能，支持從多個檔案或多種格式（如 CSV, SDF, JSON）讀取數據。
- [ ] **數據預處理**: 在進行任何特徵提取前，應先將 SMILES 字符串標準化 (Canonicalization)。
- [ ] **支援超大型數據集**: 考慮將 `BDEDataset` 從 `InMemoryDataset` 轉換為標準的 PyG `Dataset`，以支持處理超出記憶體容量的數據集。
- [ ] **數據增強 (Data Augmentation)**: 導入如 SMILES 隨機化等策略，以提升模型的泛化能力。

## 實驗管理與 ML Ops (Experiment Management & ML Ops)
- [ ] **儲存完整的預測結果**: 在評估階段，應將訓練、驗證、測試集的完整預測結果（包含原始 SMILES 和碎片資訊）儲存下來。
- [ ] **儲存運行配置**: 為確保可重現性，在每次訓練運行時，應將當次使用的 `config.json` 複製並保存到該次運行的目錄中。
- [ ] **集成超參數優化 (HPO)**: 引入如 Optuna 或 Ray Tune 等框架，以自動化探索最佳超參數。
- [ ] **集成實驗追蹤平台**: 引入如 MLflow, TensorBoard, 或 W&B 等工具，以系統化地管理、可視化和比較所有實驗結果。
- [ ] **實施模型版本管理**: 建立更完善的模型版本控制機制，確保訓練過程的可追溯性。

## 模型與特徵工程 (Model & Featurization)
- [ ] **模塊化 Featurizer**: 將特徵提取器抽象化，使其可通過配置文件輕鬆替換或組合。
- [ ] **支持多種 GNN 骨幹**: 透過配置，允許動態選擇除 MPNN 之外的其他 GNN 模型架構。

## 性能優化 (Performance Optimization)
- [ ] **整合大規模計算優化**: 整合更多針對大規模計算的性能優化策略，例如更高效的批次處理（Batching）。
- [ ] **集成 `torch.compile`**: 利用 PyTorch 2.x 的優化功能來加速模型訓練和推斷。
- [ ] **實現多 GPU 訓練**: 考慮加入多 GPU 訓練功能，以加速在大型模型或數據集上的訓練過程。

## 推論與腳本 (Inference & Scripts)
- [ ] **完善 BDE 預測器**: 實現或修復 BDE 預測器的功能，確保能正確處理 SMILES 列表輸入、去重、以及潛在的 `fragment_bond_indices` 問題。
- [ ] **優化 `create_training_template.py`**: 引入命令行參數，使其能從命令行接收 SMILES 輸入並指定輸出路徑。
- [ ] **優化 `predict.py`**:
    - 簡化模型加載邏輯，允許通過 `--run_dir` 參數直接指向訓練運行目錄。
    - 提供多種輸出格式選項（如 CSV, JSON）。
- [ ] **建立統一的 CLI 入口**: (長期建議) 使用 Click 或 Typer 等工具，將所有腳本的功能整合到一個統一的命令行界面中。
- [ ] **專案打包**: (長期建議) 將專案打包為可安裝的 Python 套件，以避免手動修改 `sys.path`。

## 程式碼重構與清理 (Code Refactoring & Cleanup)
- [ ] **移除重複的 `evaluate` 函數**: 在代碼中存在兩個 `evaluate` 函數，需要移除其中一個以保持代碼整潔。

## 測試與品質保證 (Testing & Quality Assurance)
- [ ] **擴充單元測試覆蓋率**: 為核心的數據處理、特徵提取和模型組件編寫更全面的單元測試。
- [ ] **建立整合測試**: 建立一個小型的端到端測試，自動運行一個完整的訓練和預測流程，以確保所有模塊能正確協同工作。
- [ ] **導入靜態分析與 Linter**: 引入如 `ruff`, `mypy` 等工具，並考慮整合到 pre-commit hook 中，以自動化程式碼風格檢查和型別檢查，確保代碼品質。
