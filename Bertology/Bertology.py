# coding=utf-8
"""PyTorch BERT model. """

import math  # 数学计算
import os  # 操作文件和目录
import warnings  # 警告处理
import collections  # 高级容器数据类型
from typing import Optional, Tuple, Union  # 类型提示的相关模块

import torch  # 构建和训练神经网络
import torch.utils.checkpoint  # 支持模型的 Checkpoint 机制
from torch import nn  # 构建神经网络层
from torch.nn import CrossEntropyLoss, MSELoss  # 交叉熵损失函数和均方误差损失函数

from transformers.activations import ACT2FN  # 激活函数映射表
from transformers.file_utils import (
    ModelOutput,                            # 模型输出的基类
    add_code_sample_docstrings,             # 用于添加代码示例文档字符串的函数
    add_start_docstrings,                   # 用于添加起始文档字符串的函数
    add_start_docstrings_to_model_forward,  # 用于添加前向传播起始文档字符串的函数
    replace_return_docstrings,              # 用于替换返回值文档字符串的函数
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,      # 包含过去和交叉注意力的基础模型输出类
    BaseModelOutputWithPoolingAndCrossAttentions,   # 包含汇聚和交叉注意力的基础模型输出类
    CausalLMOutputWithCrossAttentions,              # 包含因果语言模型和交叉注意力的模型输出类
    MaskedLMOutput,                                 # 包含掩码语言模型的模型输出类
    MultipleChoiceModelOutput,                      # 多项选择模型的模型输出类
    NextSentencePredictorOutput,                    # 下一个句子预测模型的模型输出类
    QuestionAnsweringModelOutput,                   # 问答模型的模型输出类
    SequenceClassifierOutput,                       # 序列分类器的模型输出类
    TokenClassifierOutput,                          # 标记分类器的模型输出类
)
from transformers.modeling_utils import (
    PreTrainedModel,                    # 预训练模型的基类
    apply_chunking_to_forward,          # 在前向传播中应用分块计算的函数
    find_pruneable_heads_and_indices,   # 找到可剪枝注意力头的函数
    prune_linear_layer,                 # 剪枝线性层的函数
)
from transformers.utils import logging  # 日志记录相关的工具函数
from transformers import BertConfig, BertPreTrainedModel  # BERT 配置类和预训练模型基类



logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]

class BertVisEmbeddings(nn.Module):
    """构建从词嵌入、位置嵌入和标记类型嵌入到总嵌入的模块。"""

    def __init__(self, config):
        super().__init__()
        self.vis_embeddings = nn.Linear(1000, config.hidden_size)  # 图像嵌入线性层，将1000维嵌入转换为隐藏大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  # 位置嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # 标记类型嵌入层

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Layer Normalization 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Dropout 层

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))  # 注册位置嵌入的位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")  # 获取位置嵌入类型，默认为"absolute"

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()[:-1]  # 获取输入张量的形状（除去最后一个维度）
        else:
            input_shape = inputs_embeds.size()[:-1]  # 获取输入嵌入的形状（除去最后一个维度）

        seq_length = input_shape[1]  # 获取序列长度

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
            # 根据过去的键值长度和序列长度计算位置编码

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            # 创建与输入形状相同的全零张量作为标记类型编码，默认设备与位置编码相同

        if inputs_embeds is None:
            inputs_embeds = self.vis_embeddings(input_ids)
            # 如果输入嵌入为空，则将输入张量通过图像嵌入层进行嵌入转换
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 根据标记类型编码获取标记类型嵌入

        embeddings = inputs_embeds + token_type_embeddings
        # 将输入嵌入和标记类型嵌入相加得到总嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            # 如果位置嵌入类型为"absolute"，则根据位置编码获取位置嵌入
            embeddings += position_embeddings
            # 将位置嵌入加到总嵌入上
        embeddings = self.LayerNorm(embeddings)
        # 应用 Layer Normalization 层
        embeddings = self.dropout(embeddings)
        # 应用 Dropout 层
        return embeddings


class BertEmbeddings(nn.Module):
    """从词嵌入、位置嵌入和标记类型嵌入构建嵌入模块。"""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 词嵌入层，将词索引映射为词嵌入向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 位置嵌入层，根据位置索引映射为位置嵌入向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 标记类型嵌入层，根据标记类型索引映射为标记类型嵌入向量

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Layer Normalization 层，用于归一化嵌入向量
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Dropout 层，用于随机丢弃部分嵌入向量

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # 注册位置嵌入的位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 获取位置嵌入类型，默认为"absolute"

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()  # 获取输入张量的形状
        else:
            input_shape = inputs_embeds.size()[:-1]  # 获取输入嵌入的形状（除去最后一个维度）

        seq_length = input_shape[1]  # 获取序列长度

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            # 根据过去的键值长度和序列长度计算位置编码

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            # 创建与输入形状相同的全零张量作为标记类型编码，默认设备与位置编码相同

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            # 如果输入嵌入为空，则将输入张量通过词嵌入层进行嵌入转换
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 根据标记类型编码获取标记类型嵌入

        embeddings = inputs_embeds + token_type_embeddings
        # 将输入嵌入和标记类型嵌入相加得到总嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        # 检查隐藏大小是否为注意力头的倍数

        self.num_attention_heads = config.num_attention_heads  # 注意力头的数量
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有注意力头的总大小

        self.query = nn.Linear(config.hidden_size, self.all_head_size)  # 查询线性层
        self.key = nn.Linear(config.hidden_size, self.all_head_size)  # 键线性层
        self.value = nn.Linear(config.hidden_size, self.all_head_size)  # 值线性层

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # Dropout 层，用于注意力概率的随机丢弃
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")  # 获取位置嵌入类型，默认为"absolute"
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入数
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
            # 距离嵌入层，用于相对位置编码

        self.is_decoder = config.is_decoder  # 判断是否为解码器

    # 将输入张量进行形状变换，以便后续计算注意力得分。通过改变张量的形状，将最后两个维度重新排列，并返回重新排列后的张量
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # 计算混合查询层

        is_cross_attention = encoder_hidden_states is not None
        # 判断是否为交叉注意力模块

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 根据输入情况计算键值层

        query_layer = self.transpose_for_scores(mixed_query_layer)
        # 计算查询层

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
            # 如果是解码器，保存先前的键值状态，用于后续计算

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 计算注意力得分，通过矩阵乘法实现query与key的点积

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
            # 计算相对位置编码相关的注意力得分

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
                # 相对位置编码方式为"relative_key"时，计算相对位置得分并加到注意力得分上
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
                # 相对位置编码方式为"relative_key_query"时，分别计算查询层和键层的相对位置得分并加到注意力得分上

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 对注意力得分进行缩放，除以注意力头的大小的平方根

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            # 应用注意力掩码

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # 将注意力得分进行 softmax 操作，得到注意力权重

        attention_probs = self.dropout(attention_probs)
        # 应用 Dropout

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            # 应用头部掩码

        context_layer = torch.matmul(attention_probs, value_layer)
        # 计算上下文层，通过注意力权重与值层的矩阵乘法得到

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 调整上下文层的形状，使其符合预期形状

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # 输出包括上下文层和注意力权重

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

# 定义BERT模型中的Self-Attention层的输出操作
# 它包含一个全连接层、LayerNorm层和Dropout层，用于处理Self-Attention层的输出，并将其与输入张量相加
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)  # 创建BertSelfAttention层
        self.output = BertSelfOutput(config)  # 创建BertSelfOutput层
        self.pruned_heads = set()  # 记录已剪枝的注意力头索引集合

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数和记录已剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 添加注意力权重到输出结果中
        return outputs



class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)  # 线性变换层
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]  # 激活函数
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)  # 激活函数
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)  # 线性变换层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Layer Normalization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # 线性变换
        hidden_states = self.dropout(hidden_states)  # Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # Layer Normalization
        return hidden_states



class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 前馈网络的分块大小
        self.seq_len_dim = 1  # 序列长度维度
        self.attention = BertAttention(config)  # 自注意力机制
        self.is_decoder = config.is_decoder  # 是否为解码器层
        self.add_cross_attention = config.add_cross_attention  # 是否添加交叉注意力机制
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)  # 交叉注意力机制
        self.intermediate = BertIntermediate(config)  # 中间层
        self.output = BertOutput(config)  # 输出层

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 解码器的单向自注意力缓存键/值元组在位置 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]  # 自注意力输出

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]  # 当前键/值
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意权重，则添加自注意力

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # 交叉注意力缓存键/值元组在过去键/值元组的位置 3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]  # 交叉注意力输出
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意权重，则添加交叉注意力

            # 将交叉注意力缓存添加到现有键/值的位置 3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs  # 将层输出添加到输出元组中

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs  # 返回输出元组

# 分块进行前向传播
def feed_forward_chunk(self, attention_output):
    """
    Args:
        attention_output (torch.Tensor): 注意力层的输出张量。
    Returns:
        torch.Tensor: 经过前向传播后的张量。
    """
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        """
        Bert编码器的初始化函数。

        Args:
            config (BertConfig): Bert模型的配置信息。
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建多个BertLayer层，并存储在ModuleList中

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Bert编码器的前向传播函数。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态张量。
            attention_mask (torch.Tensor, optional): 注意力掩码张量。默认为None。
            head_mask (List[torch.Tensor], optional): 注意力头掩码列表。默认为None。
            encoder_hidden_states (torch.Tensor, optional): 编码器的隐藏状态张量。默认为None。
            encoder_attention_mask (torch.Tensor, optional): 编码器的注意力掩码张量。默认为None。
            past_key_values (Tuple[Tuple[torch.Tensor]], optional): 保存了先前的键值对的元组。默认为None。
            use_cache (bool, optional): 是否使用缓存。默认为None。
            output_attentions (bool, optional): 是否输出注意力权重。默认为False。
            output_hidden_states (bool, optional): 是否输出隐藏状态。默认为False。
            return_dict (bool, optional): 是否返回字典形式的输出。默认为True。

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: 编码器的输出张量或元组形式的输出。

        """
        all_hidden_states = () if output_hidden_states else None
        # 用于存储每一层的隐藏状态，如果不输出隐藏状态则设置为None
        all_self_attentions = () if output_attentions else None
        # 用于存储每一层的自注意力权重，如果不输出注意力权重则设置为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 用于存储每一层的交叉注意力权重，如果不输出交叉注意力权重或模型不包含交叉注意力则设置为None

        next_decoder_cache = () if use_cache else None
        # 如果使用缓存则初始化一个空的缓存元组，否则设置为None
        for i, layer_module in enumerate(self.layer):
            # 遍历每一层的BertLayer模块
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的头掩码，如果没有头掩码则设置为None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            # 获取当前层的先前键值对，如果没有先前键值对则设置

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                # 检查是否启用渐进检查点并且当前处于训练模式

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False
                    # 如果同时启用渐进检查点和使用缓存，则发出警告并将use_cache设置为False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
                # 使用渐进检查点执行当前层的前向传播

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                # 直接执行当前层的前向传播

            hidden_states = layer_outputs[0]
            # 更新隐藏状态为当前层的输出
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
                # 如果使用缓存，则将当前层的缓存添加到next_decoder_cache中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果输出注意力权重，则将当前层的自注意力权重添加到all_self_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
                    # 如果模型包含交叉注意力且输出注意力权重，则将当前层的交叉注意力权重添加到all_cross_attentions中

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # 如果输出隐藏状态，则将最后一层的隐藏状态添加到all_hidden_states中

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
            # 如果不返回字典形式的输出，则以元组的形式返回隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力权重、所有交叉注意力权重，过滤掉为None的项

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        # 否则，以BaseModelOutputWithPastAndCrossAttentions类的实例形式返回包含最后一层隐藏状态、下一个解码器缓存、所有隐藏状态、


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
        # 通过简单地选择与第一个标记对应的隐藏状态来"汇聚"模型。
        # 将第一个标记的隐藏状态经过线性层和激活函数处理得到汇聚输出。

class BertSegPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, seg_indexs):
        seg_mean_lst = []
        for ii in range(len(seg_indexs)):
            seg_mean_lst.append(hidden_states[ii, seg_indexs[ii]].mean(dim=0))
            # 针对每个样本，计算分段索引对应的隐藏状态的均值，并将其添加到列表中
        seg_mean_tensor = torch.stack(seg_mean_lst, dim=0)
        pooled_output = self.dense(seg_mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
        # 将分段均值张量经过线性层和激活函数处理得到汇聚输出

class BertModel(BertPreTrainedModel):
    """
    Bert模型可以作为编码器（仅使用自注意力）或解码器使用，在后一种情况下，在自注意力层之间添加了交叉注意力层，
    该架构遵循Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Lukasz Kaiser和Illia Polosukhin在“Attention is all you need”一文中描述的架构。

    要作为解码器使用，需要将模型的配置中的`is_decoder`参数设置为`True`。
    要在Seq2Seq模型中使用，模型需要通过将`is_decoder`和`add_cross_attention`参数设置为`True`来初始化；然后在前向传递中需要传入`encoder_hidden_states`参数。
    """

    def __init__(self, config, add_pooling_layer=True, add_seg_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.seg_pooler = BertSegPooler(config) if add_seg_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        """
        返回输入词嵌入层。
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        设置输入词嵌入层。
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型中的注意力头。heads_to_prune：一个字典，键为层号，值为要在该层中剪枝的注意力头列表。参见基类PreTrainedModel。
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            seg_indexs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        前向传递方法，用于模型的推理过程。

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                输入序列的token id。
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                注意力掩码，用于指示哪些位置需要参与注意力计算。取值为：

                - 0：表示对应位置的token被掩盖（masked），
                - 1：表示对应位置的token未被掩盖。
            token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                输入序列的token类型id。
            position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                输入序列的位置id。
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
                注意力头（Attention Heads）的掩码，用于指定哪些注意力头需要被屏蔽。
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                输入序列的嵌入表示。
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                编码器最后一层的隐藏状态序列。如果模型被配置为解码器，则用于交叉注意力（cross-attention）。
            encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                编码器输入的填充标记的注意力掩码。如果模型被配置为解码器，则用于交叉注意力（cross-attention）。
            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`, `optional`):
                包含注意力块的预计算键（key）和值（value）隐藏状态。可用于加速解码过程。

                如果使用了 :obj:`past_key_values`，用户可以选择只输入最后的 :obj:`decoder_input_ids`（那些没有将它们的过去键值状态传递给模型的部分），形状为 :obj:`(batch_size, 1)`，而不是所有的 :obj:`decoder_input_ids`，形状为 :obj:`(batch_size, sequence_length)`。
            use_cache (:obj:`bool`, `optional`):
                如果设置为 :obj:`True`，将返回 :obj:`past_key_values` 键值状态，并可用于加速解码（参见 :obj:`past_key_values`）。
            seg_indexs (:obj:`list`, `optional`):
                分段索引列表。用于分段池化（Segment Pooling）。
            output_attentions (:obj:`bool`, `optional`):
                是否返回注意力权重。如果设置为 :obj:`True`，则注意力权重将作为输出之一返回。
            output_hidden_states (:obj:`bool`, `optional`):
                是否返回所有隐藏状态。如果设置为 :obj:`True`，则所有隐藏状态将作为输出之一返回。
            return_dict (:obj:`bool`, `optional`):
                是否返回 :obj:`BaseModelOutputWithPastAndCrossAttentions` 类型的结果字典。如果设置为 :obj:`False`，则返回元组类型的结果。

        Returns:
            :obj:`BaseModelOutputWithPastAndCrossAttentions` or :obj:`tuple`:
                如果 :obj:`return_dict=True`，则返回 :obj:`BaseModelOutputWithPastAndCrossAttentions` 类型的结果字典，包含以下字段：

                - last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                    最后一层的隐藏状态序列。
                - past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`):
                    用于加速解码过程的过去键（key）和值（value）隐藏状态。
                - hidden_states (:obj:`tuple(torch.FloatTensor)`, optional):
                    所有隐藏状态序列，如果 :obj:`output_hidden_states=True`。
                - attentions (:obj:`tuple(torch.FloatTensor)`, optional):
                    所有自注意力权重序列，如果 :obj:`output_attentions=True`。
                - cross_attentions (:obj:`tuple(torch.FloatTensor)`, optional):
                    所有交叉注意力权重序列，如果 :obj:`output_attentions=True` 且模型配置为解码器。

                如果 :obj:`return_dict=False`，则返回元组类型的结果，包含上述字段的对应项。
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # 如果没有指定output_attentions，则使用配置中的output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # 如果没有指定output_hidden_states，则使用配置中的output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果没有指定return_dict，
        # 则使用配置中的use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache  # 如果模型是decoder模型，
            # 使用指定的use_cache或配置中的use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
                # 不允许同时指定input_ids和inputs_embeds
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape  # 获取input_ids的形状信息
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape  # 获取inputs_embeds的形状信息
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")  # 必须指定input_ids或inputs_embeds

        device = input_ids.device if input_ids is not None else inputs_embeds.device  # 获取设备信息

        past_key_values_length = past_key_values[0][0].shape[
            2] if past_key_values is not None else 0  # 获取past_key_values的长度信息

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)),
                                        device=device)  # 如果没有提供attention_mask，则创建一个全1的张量，
            # 形状为(batch_size, seq_length + past_key_values_length)，使用与输入相同的设备
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long,
                                         device=device)  # 如果没有提供token_type_ids，则创建一个全0的张量，
            # 形状为input_shape，数据类型为long，使用与输入相同的设备

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape,
                                                                                 device)  # 获取扩展后的attention_mask

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape,
                                                    device=device)  # 如果没有提供encoder_attention_mask，则创建一个全1的张量，
                # 形状为encoder_hidden_shape，使用与输入相同的设备
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)  # 反转encoder_attention_mask的值
        else:
            encoder_extended_attention_mask = None  # 如果不是decoder模型或没有提供encoder_hidden_states，
            # 则设置encoder_extended_attention_mask为None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)  # 获取头部掩码

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )  # 嵌入层处理输入

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 编码器处理嵌入输出

        sequence_output = encoder_outputs[0]  # 获取编码器输出的序列
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None  # 如果存在pooler，则对序列进行池化
        seg_pooled_output = self.seg_pooler(sequence_output,
                                            seg_indexs) if self.seg_pooler is not None else None  # 如果存在seg_pooler，则对序列进行分段池化
        if seg_pooled_output is not None:
            pooled_output = seg_pooled_output  # 如果seg_pooled_output存在，则使用seg_pooled_output作为池化输出

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]  # 返回序列输出、池化输出以及其他编码器输出

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )  # 返回包含序列输出、池化输出和其他编码器输出的命名元组


class BertVisModel(BertPreTrainedModel):
    """
    该模型可以作为一个编码器（仅使用自注意力）或解码器来运行。如果将模型配置为解码器，则在自注意力层之间添加了一层交叉注意力层，遵循《Attention is
    all you need》中描述的架构。详见：https://arxiv.org/abs/1706.03762

    要将模型配置为解码器，需要使用配置参数`is_decoder`设置为True。要在Seq2Seq模型中使用该模型，需要同时将`is_decoder`和`add_cross_attention`设置为True；
    此时需要将编码器的隐藏状态作为输入传递给前向传播函数。
    """

    def __init__(self, config, add_pooling_layer=True, add_seg_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertVisEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.seg_pooler = BertSegPooler(config) if add_seg_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.vis_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.vis_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        修剪模型中的注意力头。heads_to_prune: {layer_num: 要修剪的该层中要删除的注意力头列表}，参见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            seg_indexs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        前向传播函数，用于将输入向前传递给模型进行推理。

        Args:
            input_ids (:obj:`torch.LongTensor`, `optional`):
                输入序列的token ID张量，形状为(batch_size, sequence_length)。
            attention_mask (:obj:`torch.FloatTensor`, `optional`):
                注意力遮蔽张量，形状为(batch_size, sequence_length)。
                在编码器输入的填充token索引上避免执行注意力操作。如果模型配置为解码器，则在交叉注意力中使用该掩码。
                掩码取值范围为``[0, 1]``：
                - 1表示**未被掩蔽**的token，
                - 0表示**被掩蔽**的token。
            token_type_ids (:obj:`torch.LongTensor`, `optional`):
                分段类型ID张量，形状为(batch_size, sequence_length)。
            position_ids (:obj:`torch.LongTensor`, `optional`):
                位置ID张量，形状为(batch_size, sequence_length)。
            head_mask (:obj:`torch.FloatTensor`, `optional`):
                头掩码张量，形状为(num_heads,)或(num_hidden_layers x num_heads)。
                将其转换为形状为(num_hidden_layers x batch x num_heads x seq_length x seq_length)的头掩码。
            inputs_embeds (:obj:`torch.FloatTensor`, `optional`):
                输入嵌入张量，形状为(batch_size, sequence_length, embedding_dim)。
            encoder_hidden_states (:obj:`torch.FloatTensor`, `optional`):
                编码器最后一层的隐藏状态序列，形状为(batch_size, sequence_length, hidden_size)。
                如果模型配置为解码器，则在交叉注意力中使用。
            encoder_attention_mask (:obj:`torch.FloatTensor`, `optional`):
                编码器输入的填充token索引的遮蔽张量，形状为(batch_size, sequence_length)。
                如果模型配置为解码器，则在交叉注意力中使用。遮蔽取值范围为``[0, 1]``：
                - 1表示**未被掩蔽**的token，
                - 0表示**被掩蔽**的token。
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`):
            预先计算的注意力块的键和值隐藏状态的张量元组。可用于加速解码过程。
            如果使用了`past_key_values`，用户可以选择只输入最后一个解码器输入ID（不包含给该模型的过去键值状态），
            其形状为(batch_size, 1)，而不是全部解码器输入ID，其形状为(batch_size, sequence_length)。
        use_cache (:obj:`bool`, `optional`):
            如果设置为True，则返回past_key_values键值状态，可用于加速解码过程（参见`past_key_values`）。
        seg_indexs (:obj:`list` of :obj:`torch.LongTensor`, `optional`):
            分段索引列表。每个分段索引张量的形状为(batch_size, num_segments)。
        output_attentions (:obj:`bool`, `optional`):
            是否输出注意力权重。如果设置为True，则返回注意力权重张量，默认为None。
        output_hidden_states (:obj:`bool`, `optional`):
            是否输出隐藏状态序列。如果设置为True，则返回隐藏状态张量序列，默认为None。
        return_dict (:obj:`bool`, `optional`):
            是否返回一个字典作为输出。如果设置为True，则返回一个命名元组BaseModelOutputWithPoolingAndCrossAttentions，
            否则返回一个元组。默认为None。

        Returns:
            (:obj:`tuple` of :obj:`torch.FloatTensor` or :obj:`torch.Tensor`):
            模型的输出。根据参数return_dict的设置，返回一个元组或命名元组BaseModelOutputWithPoolingAndCrossAttentions。

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()[:-1]  # 获取输入形状
            batch_size, seq_length = input_shape  # 获取批次大小和序列长度
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # 获取输入形状
            batch_size, seq_length = input_shape  # 获取批次大小和序列长度
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device  # 获取设备

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)),
                                        device=device)  # 创建全1的注意力掩码
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)  # 创建全0的分段类型ID张量

        # 我们可以自己提供形状为[batch_size, from_seq_length, to_seq_length]的自注意力遮罩
        # 在这种情况下，我们只需要将它在所有头部上进行广播
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # 如果为交叉注意力提供了形状为2D或3D的注意力遮罩
        # 我们需要将其广播为[batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # 如果需要，准备头部遮罩
        # head_mask中的1.0表示保留该头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入的head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask被转换为形状为[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 嵌入层输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        # 编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        seg_pooled_output = self.seg_pooler(sequence_output, seg_indexs) if self.seg_pooler is not None else None
        if seg_pooled_output is not None:
            pooled_output = seg_pooled_output

        # 如果不使用return_dict，则返回元组形式的输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 返回以BaseModelOutputWithPoolingAndCrossAttentions命名的输出
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForSegClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False, add_seg_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        seg_indexs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            seg_indexs=seg_indexs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            lpos = labels.view(-1) == 1
            lneg = labels.view(-1) == 0
            pp, nn = 1, 1
            wp = (pp / float(pp + nn)) * lpos / (lpos.sum() + 1e-5)
            wn = (nn / float(pp + nn)) * lneg / (lneg.sum() + 1e-5)
            w = wp + wn
            loss = (w*loss).sum()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForVisSegClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertVisModel(config, add_pooling_layer=False, add_seg_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        seg_indexs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            seg_indexs=seg_indexs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:

            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            lpos = labels.view(-1) == 1
            lneg = labels.view(-1) == 0
            pp, nn = 1, 1
            wp = (pp / float(pp + nn)) * lpos / (lpos.sum() + 1e-5)
            wn = (nn / float(pp + nn)) * lneg / (lneg.sum() + 1e-5)
            w = wp + wn
            loss = (w*loss).sum()
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

