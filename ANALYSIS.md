# 模型表徵視覺化分析 (Embedding Visualization Analysis)

本文件旨在說明如何使用 `scripts/visualize_embeddings.py` 腳本，對訓練完成的模型進行深度分析。這個工具可以幫助我們「看見」模型在不同 message passing 層學到的化學鍵表徵 (bond representations)，從而理解模型的學習過程與效果。

## 1. 目的 (Purpose)

圖神經網路 (GNN) 常被視為一個「黑盒子」。我們知道它能做出準確的預測，但我們不清楚它內部到底學到了什麼。這個視覺化工具的主要目的就是**打開這個黑盒子**。

透過觀察模型在不同深度（例如第 1, 3, 6 層）為不同化學鍵產生的內部向量（Embeddings），我們可以：
- **評估模型的學習品質**：如果模型學得好，來自相似化學環境的鍵（例如所有的 C-H 鍵）在高維空間中應該會彼此靠近，形成「群聚 (cluster)」，而與其他類型的鍵（例如 C-C 或 C-O 鍵）保持距離。
- **診斷模型問題**：如果不同類型的鍵混雜在一起，代表模型在該層次下還無法有效地區分它們。
- **理解特徵演化**：比較不同層的視覺化結果，可以看出模型是如何從一開始較混亂的表徵，逐步學習到更有區分性的、化學意義更明確的表徵。

## 2. 原理 (Principles)

這個腳本的運作原理包含以下幾個關鍵步驟：

### a. 提取中間層 Embeddings
我們的 `BDEModel` 在 `forward` 方法中被設計成可以回傳中間層的輸出。腳本會載入一個訓練好的模型，並在進行預測時啟用 `return_intermediate=True` 參數。這使得模型不僅回傳最終結果，還回傳一個包含指定層（預設為 1, 3, 6）的 bond state 向量的字典。這些向量就是化學鍵在該層的「高維表徵」。

### b. 降維 (Dimensionality Reduction)
每個 bond embedding 都是一個高維向量（例如 128 維），我們無法直接將其畫在 2D 平面上。因此，腳本採用了一個兩階段的降維方法：
1.  **PCA (主成分分析)**：先將維度從 128 維降至 50 維。這一步能有效過濾雜訊，並保留主要的變異資訊，同時加速後續的 t-SNE 計算。
2.  **t-SNE (t-分布隨機鄰居嵌入)**：接著將 50 維的向量投影到 2D 空間。t-SNE 是一種非線性降維演算法，其核心思想是在 2D 平面上重新排列點，使得在高維空間中彼此靠近的點，在 2D 平面上也盡可能地靠近。這使得我們能夠直觀地觀察數據的群聚結構。

### c. 標註與繪圖
在得到每個化學鍵的 2D 座標後，腳本會：
1.  **標註鍵類型**：使用 RDKit 分析每個鍵，為其生成一個易於理解的標籤（如 "C-H", "C-C", "C-O"）。
2.  **繪圖**：使用 `matplotlib` 和 `seaborn` 繪製散點圖。圖中每個點代表一個化學鍵，其**位置**由 t-SNE 的輸出決定，其**顏色**由它的化學鍵類型標籤決定。

最終，我們會得到三張並排的圖，分別對應第 1, 3, 6 層的 embedding 視覺化結果。

## 3. 如何使用 (Usage)

您可以透過以下指令來執行此腳本。

### 基本指令
```bash
python scripts/visualize_embeddings.py --run_dir <path_to_your_run_directory>
```
- `--run_dir` 是唯一必要的參數，指向一個包含 `config.json` 和模型權重檔案 (`.pt`) 的訓練輸出目錄。

### 可選參數
- `--data_path <path>`: 指定用來取樣的數據來源檔案。預設為 `examples/test_data.csv.gz`。
- `--num_samples <int>`: 指定要從數據中取樣多少個化學鍵進行分析。預設為 `2000`。數量越多，圖越密集，計算也越慢。
- `--output_path <path>`: 指定輸出圖檔的儲存路徑。如果未指定，預設會儲存在 `--run_dir` 目錄下的 `embedding_visualization.svg`。

### 範例
```bash
# 對我們用 ChemPropFeaturizer 訓練的模型進行分析
python scripts/visualize_embeddings.py --run_dir "training_runs/20260128_013848"
```
執行完畢後，您會在 `training_runs/20260128_013848/` 目錄下找到 `embedding_visualization.svg` 圖檔。