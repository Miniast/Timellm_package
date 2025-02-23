# Get Start

1. 将代码clone到本地

2. 环境安装。推荐环境如下

   - python 3.11

   - torch >= 2.2 ，参考torch官网教程

   - 其他包要求和安装参考

     ```bash
     pip install -r requirements.txt
     ```

3. 下载llama2-7b-hf模型。修改代码中 `models/Timellm.py` 文件中的以下行

   ```c++
   # 替换路径
   MODEL_PATH = '/home/aiseon/storage/public_data/Models/Llama-2-7b-hf'
   ```

4. 下载模型预训练权重checkpoint。放在 `checkpoints/Capacity_Timellm` 路径下


5. 准备数据集。将原始的xlsx文件放入 `dataset` 中，运行

   ```c++
   python data_prepare.py
   ```

   在 `dataset` 下会生成4个文件：`total.csv` , `train_data_index.csv` , `val_data_index.csv` , `test_data_index.csv`

6. 根据硬件配置，修改 `scripts` 下的对应sh文件（训练/验证/预测）并运行。通常只需要修改 `num_process` 和 `batch_size`

7. 结果将在 `result` 目录下生成

