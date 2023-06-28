# fate-pipeline-component
Components in fate pipeline

## 流程

- 將本地 data 上傳至 HDFS 或使用 bind_table 的方式來將 data 綁到 table 中
- 建立 Pipeline class，並在底下
- 建立 Reader component 讀取資料
- 建立 DataTransform component 做資料前處理
- 建立 Model component，創建 ML 模型
- 建立 Eval component
- Pipeline.compile ()
- Pipeline.fit()，開始訓練模型

## 
