# Project Requirement Document (PRD)

## 1. 專案目標 (Problem Statement)
將既有的化學鍵解離能 (BDE) 預測模型（原基於 Keras/TensorFlow 1.x 的 `pstjohn/bde_model_methods`）重構並移植到現代主流的 **PyTorch + PyTorch Geometric (PyG)** 架構。旨在解決舊版環境架設困難、依賴過時、維護不易的問題，並為後續研究提供一個乾淨、現代化的 Codebase。

## 2. 目標受眾 (Target Audience)
* 化學資訊學 (Cheminformatics) 研究人員
* 機器學習工程師
* 需要預測分子 BDE 的開發者（包括未來的維護者）

## 3. 成功標準 (Success Criteria)
* **架構還原度**：新模型需能接受 SMILES 字串，並產出與原版邏輯一致的圖結構特徵（特別是雙向邊的特徵處理）。
* **可訓練性**：建立完整的 PyTorch 訓練迴圈，Loss 能正常收斂。
* **再現性 (Reproducibility)**：在相同測試數據集上，新模型的 BDE 預測結果應與原論文基準誤差在可接受範圍內（MAE 接近原版水準）。
* **易用性**：代碼結構清晰，分為 `src/` (核心邏輯) 與 `run/` (執行腳本)，不依賴過時的 `nfp` 庫。

## 4. 範疇排除 (Out of Scope)
* **不進行模型架構改良**：本專案僅做「移植」，不引入 Transformer、Attention 機制或更換 GNN 骨幹。
* **不進行超參數優化**：沿用原作者設定的 Hidden Dimension (128)、層數等參數，不進行額外的 Grid Search。
* **不開發圖形介面 (GUI/Web)**：產出物僅包含 Python 訓練與推論腳本。