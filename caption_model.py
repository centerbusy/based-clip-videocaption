from fastcore.all import *
from paddlenlp.transformers import GPTLMHeadModel, GPTModel, GPTTokenizer
from paddle.nn import functional as nnf
import paddle
import paddle.nn as nn
from typing import Tuple, Optional, Union

import skimage.io as io
import PIL.Image
from IPython.display import Image
from paddlenlp.transformers import GPTLMHeadModel, GPTTokenizer
from paddlenlp.transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np


@patch_to(GPTModel)
def forward_embed(self,
                  input_embed,
                  position_ids=None,
                  attention_mask=None,
                  use_cache=False,
                  cache=None):
    self.checkpoints = []
    embedding_output = input_embed

    # TODO, use registered buffer
    length = paddle.shape(input_embed)[-2]
    if cache is not None:
        cache_length = paddle.shape(cache[0].k)[2]
        length = length + cache_length
    else:
        cache_length = 0
    causal_mask = self.bias[:, :, cache_length:length, :length]

    if attention_mask is not None:
        if attention_mask.dtype != paddle.int64:
            attention_mask = paddle.cast(attention_mask, dtype=paddle.int64)
        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - (attention_mask & causal_mask)) * -1e4
        # attention_mask = (1.0 - attention_mask) * -1e4

    else:
        attention_mask = (1.0 - causal_mask) * -1e4
    # The tensor returned by triu not in static graph.
    attention_mask.stop_gradient = True

    encoder_outputs = self.decoder(embedding_output,
                                   memory=None,
                                   tgt_mask=attention_mask,
                                   use_cache=use_cache,
                                   cache=cache)
    self.checkpoints.extend(self.decoder.checkpoints)
    return encoder_outputs


@patch_to(GPTLMHeadModel)
def forward_embed(self,
                  input_embed,
                  position_ids=None,
                  attention_mask=None,
                  use_cache=False,
                  cache=None):
    outputs = self.gpt.forward_embed(input_embed,
                                     position_ids=position_ids,
                                     attention_mask=attention_mask,
                                     use_cache=use_cache,
                                     cache=cache)

    if use_cache:
        encoder_outputs, cached_kvs = outputs[:2]
    else:
        encoder_outputs = outputs

    logits = self.lm_head(encoder_outputs)

    if use_cache:
        return logits, cached_kvs
    else:
        return logits


class MLP(nn.Layer):

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Layer):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Layer):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias_attr=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias_attr=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape([b, n, self.num_heads, c // self.num_heads])
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape([b, m, 2, self.num_heads, c // self.num_heads])
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = paddle.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        # attention = attention.softmax(axis=2)
        attention = nnf.softmax(attention, axis=2)
        out = paddle.einsum('bnmh,bmhd->bnhd', attention, values).reshape([b, n, c])
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Layer):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Layer = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Layer):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Layer = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.LayerList(layers)


class TransformerMapper(nn.Layer):

    def forward(self, x):
        x = self.linear(x).reshape([x.shape[0], self.clip_length, -1])
        prefix = self.prefix_const.unsqueeze(0).expand([x.shape[0], *self.prefix_const.shape])
        prefix = paddle.concat((x, prefix), axis=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        # self.prefix_const = nn.Parameter(paddle.randn(prefix_length, dim_embedding), requires_grad=True)
        pa_param = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        self.prefix_const = paddle.create_parameter(shape=[prefix_length, dim_embedding],
                                                    dtype='float32',
                                                    attr=pa_param)


class ClipCaptionModel(nn.Layer):

    def get_dummy_token(self, batch_size: int, device: paddle.device) -> paddle.Tensor:
        return paddle.zeros(batch_size, self.prefix_length, dtype=paddle.int64, device=device)

    def forward(self, tokens: paddle.Tensor, prefix: paddle.Tensor, mask: Optional[paddle.Tensor] = None,
                labels: Optional[paddle.Tensor] = None):
        embedding_text = self.gpt.embeddings(tokens)  # 没有position ids
        prefix_projections = self.clip_project(prefix).reshape([-1, self.prefix_length, self.gpt_embedding_size])
        embedding_cat = paddle.concat((prefix_projections, embedding_text), axis=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = paddle.concat((dummy_token, tokens), axis=1)
        # out = self.gpt.forward_embed(embedding_cat, labels=labels, attention_mask=mask)
        out = self.gpt.forward_embed(embedding_cat, attention_mask=mask)

        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 768,
                 num_layers: int = 8):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPTLMHeadModel.from_pretrained('gpt2-medium-en')
        self.gpt.eval()
        self.gpt_embedding_size = 1024
        self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                              clip_length, num_layers)


prefix_length = 10


def generate_beam(
        model,
        tokenizer,
        beam_size: int = 5,
        prompt=None,
        embed=None,
        entry_length=67,
        temperature=1.0,
        stop_token: str = ".",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)["input_ids"][0]
    tokens = None
    scores = None
    seq_lengths = paddle.ones([beam_size])
    is_stopped = paddle.zeros([beam_size], dtype=paddle.bool)
    with paddle.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = paddle.to_tensor(tokenizer.encode(prompt)["input_ids"])
                tokens = tokens.unsqueeze(0)
                generated = model.gpt.embedding(tokens)
        for i in range(entry_length):
            outputs = model.gpt.forward_embed(generated)
            # logits = outputs.logits
            logits = outputs
            # logits = nnf.softmax(outputs,axis = -1)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = nnf.softmax(logits, axis=-1).log()
            # logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand([beam_size, *generated.shape[1:]])
                next_tokens, scores = next_tokens.transpose([1, 0]), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand([beam_size, *tokens.shape[1:]])
                    tokens = paddle.concat((tokens, next_tokens), axis=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.reshape([-1]).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = paddle.concat((tokens, next_tokens), axis=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source.numpy().tolist()]
            next_token_embed = model.gpt.embeddings(next_tokens.squeeze()).reshape(
                [generated.shape[0], 1, -1]
            )
            generated = paddle.concat((generated, next_token_embed), axis=1)
            # is_stopped = is_stopped + next_tokens.equal(stop_token_index).squeeze()
            temp_a = next_tokens.equal(paddle.full_like(next_tokens, stop_token_index)).astype("float32").squeeze()
            is_stopped = paddle.any(paddle.stack([is_stopped.astype("float32"), temp_a], axis=0).astype("bool"), axis=0)
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts
