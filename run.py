import collections  # 处理集合类型的数据
import json  # JSON数据的解析和处理
import logging  # 日志记录
import os  # 与操作系统进行交互
from argparse import ArgumentParser  # 解析命令行参数
from pprint import pformat  # 格式化输出Python对象

import torch  # 构建和训练神经网络
import torch.nn.functional as F  # 各种非线性函数和操作
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear  # 训练过程中的进度条和学习率调整
# 将训练过程记录到Tensorboard日志中
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from ignite.engine import Engine, Events  # 构建训练引擎和定义事件
from ignite.handlers import ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver  # 模型保存和加载
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup  # 创建带有预热的学习率调度器
from ignite.metrics import RunningAverage  # 计算平均值
from torch.utils.data import DataLoader  # 加载数据
from tqdm import tqdm  # 显示进度条
from transformers import AdamW, BertTokenizer  # BERT模型的优化器和分词器
from transformers.file_utils import CONFIG_NAME  # 配置文件的名称
from transformers.models.bert.configuration_bert import BertConfig  # BERT模型的配置

from data.seg_resnetnsp_dataset import DataSet, collate_fn, get_dataset  # 加载数据集
from model.resnetnspBert import BertForSegClassification, BertForVisSegClassification


logger = logging.getLogger(__file__)  # 创建一个logger对象，用于记录日志，__file__表示当前脚本的文件名

def average_distributed_scalar(scalar, args):
    # 如果处于分布式训练环境中，则对节点上的标量进行平均（在分布式评估中使用）
    if args.local_rank == -1:
        return scalar
    # 创建一个torch.tensor对象，将标量转换为张量，并将其放置在指定的设备上，然后除以分布式训练的设备数
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    # 对所有节点上的张量进行求和操作
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    # 返回标量张量的数值部分
    return scalar_t.item()

# 加载数据
def get_data_loaders_new(args, tokenizer):
    # 训练集
    print('start load train data.')
    train_data = get_dataset(tokenizer, args.train_path)
    # 验证集
    print('start load valid data.')
    valid_data = get_dataset(tokenizer, args.valid_path)
    # 测试集
    print('start load test data.')
    test_data = get_dataset(tokenizer, args.test_path)
    # 视频数据以及数据的进一步加载
    if args.video:
        with open(args.feature_path) as jh:
            feature = json.load(jh)
        train_dataset = DataSet(train_data, tokenizer, feature)
        valid_dataset = DataSet(valid_data, tokenizer, feature)
        test_dataset = DataSet(test_data, tokenizer, feature)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=2, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=0, shuffle=False,
                                  collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=0, shuffle=False,
                                 collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    else:
        train_dataset = DataSet(train_data, tokenizer, None)
        valid_dataset = DataSet(valid_data, tokenizer, None)
        test_dataset = DataSet(test_data, tokenizer, None)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=2, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=0, shuffle=False,
                                  collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=0, shuffle=False,
                                 collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
    return train_loader, valid_loader, test_loader


def train():
    # 超参数定义
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="inputs/preprocessed/MDSS_train.json",
                        help="Path of the trainset")
    parser.add_argument("--valid_path", type=str, default="inputs/preprocessed/MDSS_valid.json",
                        help="Path of the validset")
    parser.add_argument("--test_path", type=str, default="inputs/preprocessed/MDSS_test.json",
                        help="Path of the testset")
    parser.add_argument("--feature_path", type=str, default="inputs/MDSS_clipid2frames.json",
                        help="Path of the feature")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    # 梯度累积更新
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    # 梯度裁减范数
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--gpuid", type=str, default='0', help='select proper gpu id')
    parser.add_argument("--model", type=str, default='bert', help='Pretrained Model name')
    parser.add_argument('--video', type=int, default=0, help='if use video: 1 use 0 not')
    # 输出结果命名的后缀：在模型名字后面添加后缀用于后续结果的区分
    parser.add_argument('--exp_set', type=str, default='_test')
    parser.add_argument('--model_checkpoint', type=str, default="./bert-base-uncased")

    parser.add_argument('--test_each_epoch', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ft', type=int, default=1, choices=[0, 1], help='1: finetune bert 0: train from scratch')
    parser.add_argument('--warmup_init', type=float, default=1e-07)
    parser.add_argument('--warmup_duration', type=float, default=5000)
    args = parser.parse_args()

    args.valid_batch_size = args.train_batch_size
    args.test_batch_size = args.train_batch_size
    args.model = 'bert'
    exp_set = args.exp_set
    args.exp = args.model + exp_set
    args.log_path = 'ckpts/' + args.exp + '/'
    args.tb_path = 'tb_logs/' + args.exp + '/'
    if args.device == 'cuda':
        args.device = 'cuda:' + args.gpuid

    # 模型选择
    if args.model == 'bert':
        args.model_checkpoint = args.model_checkpoint
    else:
        raise ValueError('NO IMPLEMENTED MODEL!')

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.tb_path):
        os.makedirs(args.tb_path)
    # 输出运行日志
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    # 使用BertTokenizer作为分词器
    tokenizer_class = BertTokenizer
    if args.video:
        # 如果args.video为True，则使用BertForVisSegClassification模型类
        model_class = BertForVisSegClassification
        if not args.ft: # 如果不进行微调（fine-tuning），则使用默认的BertConfig配置
            bert_config = BertConfig()
            model = model_class(bert_config)  # 创建BertForVisSegClassification模型对象
        else:
            # 从预训练模型加载BertForVisSegClassification模型
            model = model_class.from_pretrained(args.model_checkpoint)
    else:
        model_class = BertForSegClassification  # 如果args.video为False，则使用BertForSegClassification模型类
        if not args.ft: # 如果不进行微调（fine-tuning），则使用默认的BertConfig配置
            bert_config = BertConfig()
            # 创建BertForSegClassification模型对象
            model = model_class(bert_config)
        else:
            # 从预训练模型加载BertForSegClassification模型
            model = model_class.from_pretrained(args.model_checkpoint)

    # 从预训练模型加载分词器对象
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    # 将模型移动到指定的设备上
    model.to(args.device)
    # 创建AdamW优化器，将模型参数传递给优化器，并设置学习率为args.lr
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # 将args.pad设置为模型配置中的pad_token_id属性值
    args.pad = model.config.pad_token_id
    # 加载数据集
    logger.info("Prepare datasets for Bert")
    train_loader, valid_loader, test_loader = get_data_loaders_new(args, tokenizer)

    # 定义训练函数和训练器的更新函数
    def update(engine, batch):
        dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs, \
            feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
        # 从batch中解包获取所需的变量
        if args.video == 0:
            # 如果args.video为0，表示不使用视频特征
            dialog_ids = dialog_ids.to(args.device)  # 将对话id转移到指定的设备上
            dialog_type_ids = dialog_type_ids.to(args.device)  # 将对话类型id转移到指定的设备上
            dialog_mask = dialog_mask.to(args.device)  # 将对话掩码转移到指定的设备上
            session_label_ids = session_label_ids.to(args.device)  # 将会话标签id转移到指定的设备上
            session_indexs = [sess.to(args.device) for sess in session_indexs]  # 将每个会话索引转移到指定的设备上
        else:
            # 如果args.video不为0，表示使用视频特征
            feature_ids = feature_ids.to(args.device)  # 将特征id转移到指定的设备上
            feature_type_ids = feature_type_ids.to(args.device)  # 将特征类型id转移到指定的设备上
            feature_mask = feature_mask.to(args.device)  # 将特征掩码转移到指定的设备上
            scene_indexs = [scene.to(args.device) for scene in scene_indexs]  # 将每个场景索引转移到指定的设备上
            scene_label_ids = scene_label_ids.to(args.device)  # 将场景标签id转移到指定的设备上

        # optimize Bert
        model.train(True)
        if args.video == 0:
            bsz = 16
            loss = model(dialog_ids[:bsz], dialog_mask[:bsz], dialog_type_ids[:bsz],
                         labels=session_label_ids[:bsz], seg_indexs=session_indexs[:bsz])[0]
        else:
            loss = model(feature_ids, feature_mask, feature_type_ids, labels=scene_label_ids, seg_indexs=scene_indexs)[
                0]
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    def valid(engine, batch):
        model.train(False)  # 将模型设为评估模式
        f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False,
                                  threshold=0.5)  # 创建F1ScoreMetric指标对象
        f1_metric = f1_metric.to(args.device)  # 将指标对象移动到指定的设备上
        with torch.no_grad():
            dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs, \
                feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
            if args.video == 0:
                dialog_ids = dialog_ids.to(args.device)  # 将对话id转移到指定的设备上
                dialog_type_ids = dialog_type_ids.to(args.device)  # 将对话类型id转移到指定的设备上
                dialog_mask = dialog_mask.to(args.device)  # 将对话掩码转移到指定的设备上
                session_indexs = [sess.to(args.device) for sess in session_indexs]  # 将每个会话索引转移到指定的设备上
                label = session_label_ids.to(args.device)  # 将会话标签id转移到指定的设备上
                logits = model(dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[
                    0]  # 调用模型进行前向传播计算，得到logits
            else:
                feature_ids = feature_ids.to(args.device)  # 将特征id转移到指定的设备上
                feature_type_ids = feature_type_ids.to(args.device)  # 将特征类型id转移到指定的设备上
                feature_mask = feature_mask.to(args.device)  # 将特征掩码转移到指定的设备上
                scene_indexs = [scene.to(args.device) for scene in scene_indexs]  # 将每个场景索引转移到指定的设备上
                label = scene_label_ids.to(args.device)  # 将场景标签id转移到指定的设备上
                logits = model(feature_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[
                    0]  # 调用模型进行前向传播计算，得到logits
            prob = F.softmax(logits, dim=1)  # 对logits进行softmax激活，得到概率
            f1_metric.update(prob[:, 1], label)  # 更新F1指标
            f1 = f1_metric.compute()  # 计算F1指标的值
        return f1.item()  # 返回F1指标的值（转换为Python标量）

    trainer = Engine(update)
    validator = Engine(valid)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: validator.run(valid_loader))

    if args.ft:
        # 如果进行微调（fine-tuning），则使用PiecewiseLinear调度器，根据预定义的学习率时间表进行学习率衰减
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    else:
        # 如果不进行微调（fine-tuning），则先使用PiecewiseLinear调度器进行学习率衰减，
        # 然后使用create_lr_scheduler_with_warmup创建具有预热的调度器
        torch_lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (
            args.n_epochs * len(train_loader) - args.warmup_duration, 0.0)])
        scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler, warmup_start_value=args.warmup_init,
                                                    warmup_duration=args.warmup_duration)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # 准备度量指标
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x).attach(validator, "f1")

    pbar = ProgressBar(persist=True)  # 创建进度条对象用于训练过程的进度显示，设置为持久化显示
    pbar.attach(trainer, metric_names=["loss"])  # 将进度条对象绑定到训练器上，并指定显示的指标为"loss"
    val_pbar = ProgressBar(persist=True)  # 创建进度条对象用于验证过程的进度显示，设置为持久化显示
    val_pbar.attach(validator, metric_names=['f1'])  # 将进度条对象绑定到验证器上，并指定显示的指标为"f1"

    tb_logger = TensorboardLogger(log_dir=args.tb_path)  # 创建TensorboardLogger对象，设置日志保存路径为args.tb_path
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                     event_name=Events.ITERATION_COMPLETED)  # 将TensorboardLogger对象与训练器绑定，指定日志处理程序和触发事件
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer),
                     event_name=Events.ITERATION_STARTED)  # 将TensorboardLogger对象与训练器绑定，指定日志处理程序和触发事件
    tb_logger.attach(validator, log_handler=OutputHandler(tag='validation', metric_names=["f1"],
                                                          global_step_transform=global_step_from_engine(trainer)),
                     event_name=Events.EPOCH_COMPLETED)  # 将TensorboardLogger对象与验证器绑定，指定日志处理程序和触发事件

    checkpoint_handler = ModelCheckpoint(args.log_path, 'checkpoint', n_saved=args.n_epochs, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
        'mymodel': getattr(model, 'module', model)})  # 在每个 epoch 完成时调用 checkpoint_handler 保存模型
    # "getattr" 处理分布式封装
    # 将 'mymodel'（模型名称）与 getattr(model, 'module', model) 绑定，确保在分布式训练中正确处理模型的获取

    torch.save(args, args.log_path + 'model_training_args.bin')  # 保存训练参数至文件model_training_args.bin
    getattr(model, 'module', model).config.to_json_file(os.path.join(args.log_path, CONFIG_NAME))  # 将模型配置保存为JSON文件
    tokenizer.save_vocabulary(args.log_path)  # 保存分词器的词汇表

    best_score = Checkpoint.get_default_score_fn('f1')  # 定义最佳模型评分函数为F1指标
    best_model_handler = Checkpoint(
        {'mymodel': getattr(model, 'module', model)},
        filename_prefix='best',
        save_handler=DiskSaver(args.log_path, create_dir=True, require_empty=False),
        score_name='f1',
        score_function=best_score,
        global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
        filename_pattern='{filename_prefix}_{global_step}_{score_name}={score}.{ext}'
    )
    validator.add_event_handler(Events.COMPLETED, best_model_handler)  # 在验证器完成事件时保存最佳模型

    if args.test_each_epoch:
        @trainer.on(Events.EPOCH_COMPLETED)
        def test():
            model.train(False)  # 将模型设为评估模式
            result = collections.defaultdict(list)  # 创建一个默认空列表的字典，用于存储测试结果
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='test'):  # 在测试数据加载器上进行迭代，显示进度条描述为'test'
                    dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs, \
                        feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
                    if args.video == 0:
                        dialog_ids = dialog_ids.to(args.device)  # 将对话id转移到指定的设备上
                        dialog_type_ids = dialog_type_ids.to(args.device)  # 将对话类型id转移到指定的设备上
                        dialog_mask = dialog_mask.to(args.device)  # 将对话掩码转移到指定的设备上
                        session_indexs = [sess.to(args.device) for sess in session_indexs]  # 将每个会话索引转移到指定的设备上
                        logits = model(dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[
                            0]  # 调用模型进行前向传播计算，得到logits
                        probs = F.softmax(logits, dim=1)  # 对logits进行softmax激活，得到概率
                        preds = torch.argmax(probs, dim=1)  # 根据概率预测最可能的类别
                        for vid, pre in zip(vid_lst, preds):
                            result[vid].append(pre.item())  # 将预测结果添加到结果字典中

                    else:
                        feature_ids = feature_ids.to(args.device)  # 将特征id转移到指定的设备上
                        feature_type_ids = feature_type_ids.to(args.device)  # 将特征类型id转移到指定的设备上
                        feature_mask = feature_mask.to(args.device)  # 将特征掩码转移到指定的设备上
                        scene_indexs = [scene.to(args.device) for scene in scene_indexs]  # 将每个场景索引转移到指定的设备上
                        logits = model(feature_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[
                            0]  # 调用模型进行前向传播计算，得到logits
                        probs = F.softmax(logits, dim=1)  # 对logits进行softmax激活，得到概率
                        preds = torch.argmax(probs, dim=1)  # 根据概率预测最可能的类别
                        for vid, pre in zip(vid_lst, preds):
                            result[vid].append(pre.item())  # 将预测结果添加到结果字典中

            output_dir = 'results/{}/'.format(args.exp)  # 设置输出目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # 如果输出目录不存在，则创建该目录
            if args.video:
                with open(os.path.join(output_dir, 'scene_res_{}.json'.format(trainer.state.epoch)), 'w') as jh:
                    json.dump(result, jh)  # 将结果字典以JSON格式写入文件，文件名格式为'scene_res_{epoch}.json'
            else:
                with open(os.path.join(output_dir, 'session_res_{}.json'.format(trainer.state.epoch)), 'w') as jh:
                    json.dump(result, jh)  # 将结果字典以JSON格式写入文件，文件名格式为'session_res_{epoch}.json'

    # 运行训练
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # 在主进程中：关闭Tensorboard记录器并重命名最后一个检查点（以便使用OpenAIGPTModel.from_pretrained方法方便重新加载）
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        tb_logger.close()  # 关闭Tensorboard记录器


if __name__ == "__main__":
    # 仅训练
    train()
