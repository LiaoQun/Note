step1:
1. AtomFeaturizer BondFeaturizer具體來說可以參考 @alfabet\preprocessor.py，裡面有函數使用到。
2. Tokenizer 的詞彙表有先保存了一個預先定義的版本在 @etc\preprocessor.json，預設目前只使用到這個靜態列表。但之後會有擴充需求，請設計一個當訓練新的資料集時，可以動態更新詞彙表的機制。訓練後保存成json格式。

step3:
先保留問題，我之後回答

step4:
2. 輸出層 (Output Layer) 的輸入是邊特徵沒錯，這個模型主要是想要預測邊的特徵。
3. 維度註解依照你的建議

step5:
跟你想的一樣

step6:
需要設定random seed確保我可以重現實驗。train/vali/test=80/10/10，請幫我隨機打亂資料集。