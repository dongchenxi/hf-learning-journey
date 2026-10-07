1. load_dataset("csv/json", data_files=...) 可以把本地 CSV/JSON 加载成 Hugging Face Dataset。

2. data_files 用于指定需要加载的数据文件。

3. DatasetDict 用于组织 train、validation、test 等 Dataset split。

4. Train 用于训练并更新参数，Validation 用于训练过程中评估和选择，Test 用于最终评估。

5. train_test_split(test_size=..., seed=...) 可以把 Dataset 随机划分为 train/test，其中 test_size 控制 test 大小，seed 用于保证划分可复现。