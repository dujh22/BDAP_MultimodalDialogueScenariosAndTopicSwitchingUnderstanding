# 多模态对话场景与话题切换

## 数据集

### 统计数据

|       | 剪辑    | 话语      | 场景     | 话题     |
| ----- |-------|---------|--------|--------|
| train | 8,025 | 200,073 | 11,440 | 20,996 |
| valid | 416   | 10,532  | 615    | 1,270  |
| test  | 403   | 10,260  | -      | -      |

### 环境配置

请按照requirements.txt文件进行配置，其中可以参考的部分有：

python 3.7

pip install packaging==21.3

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch-ignite

pip install transformers==4.29.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

### 参数修改

在核心的运行代码run.py中，为了适配您的GPU硬件环境，请修改以下内容：

parser.add_argument("--device", type=str, default="cuda:0   # 从cuda改为自己机器的情况

### 指令

```python
python run.py \
--train_path inputs/preprocessed/MDSS_train.json \
--valid_path inputs/preprocessed/MDSS_valid.json \
--test_path inputs/preprocessed/MDSS_test.json \
--train_batch_size 8 \
--lr 1e-5 \
--gradient_accumulation_steps 2\
--model_checkpoint bert-base-uncased \
--ft 1 \ # 1: fitune 0: train from scratch
--exp_set _baseline \
--video 0 \ # 0: session identification 1: scene identification
--gpuid 0 \
--test_each_epoch 1 \ # test for each epoch

```
注意去掉指令中的注释，如下：
```python
python run.py \
--train_path inputs/preprocessed/MDSS_train.json \
--valid_path inputs/preprocessed/MDSS_valid.json \
--test_path inputs/preprocessed/MDSS_test.json \
--train_batch_size 8 \
--lr 1e-5 \
--gradient_accumulation_steps 2 \
--model_checkpoint bert-base-uncased \
--ft 1 \ 
--exp_set _baseline \
--video 0 \ 
--gpuid 0 \
--test_each_epoch 1 
```
### 结果

| Task         | F1    |
| ------------ | ----- |
| 场景切换辨别 | 35.40 |
| 话题切换辨别 | 42.15 |

### 说明

项目仅展示了核心代码（可运行），删除了几乎全部的中间文件、结果文件和依赖文件，请对于依赖文件自行补充到文件结构中。

- bert-base-uncased：https://huggingface.co/bert-base-uncased/tree/main
- 视频特征可以从 https://pan.baidu.com/s/1pMKGY6Vkiy7N3YzD041nQA?pwd=9n53下载，将特征集合解压到 inputs/features/resnet，同理fast_rcnn

### 相关贡献

| 序     | 名   | 算法/模型分工       | 项目分工                      |
| ------ |-----|---------------|---------------------------|
| leader | DJH | BERTology、多模态 | 研究框架、赛题分析、优化思考、算法设计与实现、汇总 |
| 2      | LPF | faster-rcnn   | 赛题分析、算法设计与实现                   |
| 3      | ZYY | ResNet        | 算法设计与实现、优化思考                   |
| 4      | HRZ | 二分类           | 算法设计与实现                   |