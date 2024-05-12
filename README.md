# ECAPAModel

## 运行

### 训练

- 配置环境

```
pip install -r requirements.txt 或者
conda activate server_dev
```

- 训练

```
python trainECAPAModel.py --save_path <save_path>
```

### 测试

```
python testECAPAModel.py --eval
```



## 模型解释

### 训练

- `trainECAPAModel.py` 训练脚本

- `ECAPAModel.py`  调用模型

- `model.py`  模型
  - ECAPA_TDNN -> AAMsoftmax

### 测试

- `testECAPAModel.py`  测试脚本

  修改参数：

  - `train_list`、`train_path`、`eval_list`、`eval_path`、`musan_path`噪声、`rir_path`混响、`save_path`
  - `initial_model` 需要测试的模型（required）
  - `ECAPAMModel.py`  调用模型脚本中`test_network`   里`modelname`每次要更新



### 数据加载

- `dataloader.py`  数据加载脚本
  - 实现数据增强，加入噪声、混响，注意匹配所有wav文件时的路径问题
  - `get_label(label)`  提取标签 `fake-0 genuine-1`












