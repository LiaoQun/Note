# ALFABET Repository AI Analysis
*   **Repository:** [NREL/alfabet](https://deepwiki.com/NREL/alfabet)
*   **System Architecture (bde\_model\_methods):** [Link](https://deepwiki.com/pstjohn/bde_model_methods#system-architecture)

## Related Implementations & Developers
ALFABET is a specialized model.  Limited community-driven refactoring or "unofficial optimizations" exist.  Modern implementations of similar architectures often use Chemprop (PyTorch-based MPNN framework). Robert Paton, an ALFABET author, is active in this area.

*   **Chemprop (PyTorch):** Recommended for training similar BDE models using a modern PyTorch architecture. Supports comparable Message Passing mechanisms.


# Gaussian Method Reference
*   [Gaussian Tutorial - Open Shell System](http://blog.molcalx.com.cn/2017/12/15/gaussian-tutorial-open-shell-system.html)


# Alfabet BDE project refactor to torch
可參考性優先度: alfabet資料夾 > bde_model_train > nfp==0.3.0
### 一、 ALFABET 重構策略 (Refactoring Strategy)
**核心原則**：**不要修復原始訓練代碼** (Don't fix the legacy training loop)。僅需提取其「數據處理邏輯」，並用現代 PyTorch 重寫。

#### 執行路徑圖 (4 Checkpoints)：
1.  **檢核點 1：特徵工程解剖 (Featurization Autopsy)**
    * **任務**：閱讀原始 `preprocessing.py`。
    * **目標**：確認 Atom Features (原子特徵) 與 Bond Features (鍵特徵) 的具體定義與維度（例如：原子特徵長度是 42 還是 133？）。
2.  **檢核點 2：建立特徵轉換器 (The Converter)**
    * **任務**：撰寫 `smiles_to_graph` 函數。
    * **目標**：輸入 SMILES，輸出符合 PyTorch 格式的 Tensor (`x`, `edge_index`, `edge_attr`)。需確保數值邏輯與原作一致。
3.  **檢核點 3：數據管道 (The Pipeline)**
    * **任務**：建立 `Dataset` 與 `DataLoader`。
    * **目標**：處理 Graph Batching (將小圖拼成大圖)，確保數據能批量送入 GPU。
4.  **檢核點 4：模型搭建 (The Brain)**
    * **任務**：使用 PyTorch 實作 MPNN 層。
    * **目標**：復刻 $h_v^{t+1} = \text{ReLU}(W \cdot \text{concat}(h_v^t, \sum m_{vw}))$ 的數學邏輯。

---

### 二、 GCNN 與 MPNN 架構解析
**核心概念**：
* **繼承關係**：MPNN 是父類別 (Parent Class)，GCNN 是子類別 (Child Class)。
* **關鍵差異**：在於對 **「邊 (Edge/Bond)」** 的處理方式。

| 特性 | GCNN (傳統/簡化版) | MPNN (ALFABET 使用) |
| :--- | :--- | :--- |
| **關注點** | 節點 (原子) | 節點 + 邊 (原子 + 化學鍵) |
| **鍵的處理** | 通常視為權重或僅用作連接索引 | **顯式特徵輸入** (單/雙/三鍵會影響運算) |
| **BDE 適用性** | 低 (無法區分鍵的強弱) | **高** (BDE 高度依賴鍵的類型) |

---

### 三、 兩模型整合方案 (Integration Plan)
**目標**：將現有的 MP/BP GCNN 模型與新重構的 BDE MPNN 模型合併。

#### 1. 工程層級：統一數據管道 (最優先)
* **做法**：建立一個「超級特徵產生器」。
* **邏輯**：取兩者特徵的**聯集 (Union)**。確保同一個 SMILES 產生的 Graph Data 可以同時被兩個模型讀取，避免重複計算。

#### 2. 模型層級：共用骨幹 (Shared Backbone)
* **架構**：Y 字型架構 (Multi-task Learning)。
    * **Shared Encoder** (MPNN Layers): 負責理解化學結構，輸出 Node Embeddings。
    * **Head A (MP/BP)**: Global Pooling -> MLP -> 預測分子性質。
    * **Head B (BDE)**: Edge Concatenation -> MLP -> 預測鍵性質。
* **條件**：你的 GCNN 必須升級為能處理 `edge_attr` 的 MPNN 層，才能支援 BDE 預測。

---

### 下一步行動建議 (Next Step)

請將這份摘要複製到你的 IDE Agent 中作為 **System Prompt** 或 **Context**，然後開始執行 **「檢核點 1」**：

> **"找出原始 ALFABET 代碼中的特徵定義，並分析其輸入輸出維度。"**