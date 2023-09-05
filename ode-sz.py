import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ode.layer_history import CreateLayerHistory
from ode.utils1 import nn_seq_ms, train, evaluate, predict_future, Lossfun
from ode.multihead_attention import MultiheadAttention
from ode.relative_multihead_attention import RelativeMultiheadAttention
from torchsummary import summary


# In[2]:


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input, incremental_state=None, timestep=None):
        bs, seq_len = input.shape[:2]
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
            )
        self.weights = self.weights.type_as(self._float_tensor)
        return self.weights.view(bs, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)


# In[3]:


def PositionalEmbedding(num_embeddings, embedding_dim):
    return SinusoidalPositionalEmbedding(embedding_dim, num_embeddings)


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


# In[4]:


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


# In[5]:


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.calculate_num = args.enc_calculate_num
        if args.max_relative_length == -1:
            self.self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout,
            )
        else:
            self.self_attn = RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout,
            )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])
        self.gate_linear = nn.Linear(2 * self.embed_dim, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        runge_kutta_list = []
        residual = x
        for step_size in range(int(self.calculate_num)):
            x = self.maybe_layer_norm(0, x, before=True)
            x, _ = self.self_attn(query=x, key=x, value=x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            runge_kutta_list.append(x)
            if int(self.calculate_num) == 4:
                if step_size == 0 or step_size == 1:
                    x = residual + 1 / 2 * x
                    x = residual + x
                elif step_size == 2:
                    x = residual + x
            elif int(self.calculate_num) == 3:
                if step_size == 0:
                    x = residual + 1 / 2 * x
                elif step_size == 1:
                    x = residual - runge_kutta_list[1] + 2 * x
            elif int(self.calculate_num) == 2:
                x = residual + x
        if self.calculate_num == 4:
            # RK4-block
            x = residual + 1 / 6 * (
                    runge_kutta_list[0] + 2 * runge_kutta_list[1] + 2 * runge_kutta_list[2] + runge_kutta_list[3])

            # learnable alpha for RK4-block
            # x = residual + self.alpha[0]*runge_kutta_list[0] + self.alpha[1]*runge_kutta_list[1] + self.alpha[2]*runge_kutta_list[2] + self.alpha[3]*runge_kutta_list[3]
        elif self.calculate_num == 3:
            # RK3-block
            x = residual + (runge_kutta_list[0] + runge_kutta_list[1] + runge_kutta_list[2])
        elif self.calculate_num == 2:
            # learnbale coefficients for RK2-block with gated
            # alpha = torch.sigmoid(self.gate_linear(torch.cat((runge_kutta_list[0], runge_kutta_list[1]), dim=-1)))
            # x = residual + alpha * runge_kutta_list[0] + (1 - alpha) * runge_kutta_list[1]
            # RK2-block
            x = residual + 1 / 2 * (runge_kutta_list[0] + runge_kutta_list[1])
            # learnable coefficients with initialized 1
            # x = residual + self.alpha[0] * runge_kutta_list[0] + self.alpha[1] * runge_kutta_list[1]
        elif self.calculate_num == 1:
            x = residual + runge_kutta_list[0]

        elif self.calculate_num == 2.1:
            x = residual + runge_kutta_list[0] + runge_kutta_list[1]
        else:
            raise ValueError("invalid caculate num!")

        # x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        for step_size in range(int(self.calculate_num)):
            x = self.maybe_layer_norm(1, x, before=True)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.relu_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            runge_kutta_list.append(x)
            if int(self.calculate_num) == 4:
                if step_size == 0 or step_size == 1:
                    # x = residual + 1 / 2 * x
                    x = residual + x
                elif step_size == 2:
                    x = residual + x
            elif int(self.calculate_num) == 3:
                if step_size == 0:
                    x = residual + 1 / 2 * x
                elif step_size == 1:
                    x = residual - runge_kutta_list[1] + 2 * x
            elif int(self.calculate_num) == 2:
                x = residual + x
        if self.calculate_num == 4:
            # RK4-block
            x = residual + 1 / 6 * (
                    runge_kutta_list[0] + 2 * runge_kutta_list[1] + 2 * runge_kutta_list[2] + runge_kutta_list[3])

            # learnable alpha for RK4-block
            # x = residual + self.alpha[0]*runge_kutta_list[0] + self.alpha[1]*runge_kutta_list[1] + self.alpha[2]*runge_kutta_list[2] + self.alpha[3]*runge_kutta_list[3]
        elif self.calculate_num == 3:
            # RK3-block
            # x = residual + 1 / 6 * (runge_kutta_list[0] + 4 * runge_kutta_list[1] + runge_kutta_list[2])
            x = residual + (runge_kutta_list[0] + runge_kutta_list[1] + runge_kutta_list[2])
        elif self.calculate_num == 2:
            # learnbale coefficients for RK2-block with gated
            alpha = torch.sigmoid(self.gate_linear(torch.cat((runge_kutta_list[0], runge_kutta_list[1]), dim=-1)))
            x = residual + alpha[0] * runge_kutta_list[0] + alpha[1] * runge_kutta_list[1]
        elif self.calculate_num == 1:
            x = residual + runge_kutta_list[0]

        elif self.calculate_num == 2.1:
            x = residual + runge_kutta_list[0] + runge_kutta_list[1]

        elif self.calculate_num == 0:
            x = residual
        else:
            raise ValueError("invalid caculate num!")
        # x = residual + x

        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x



# In[6]:


class ODETransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_dim = args.embedding_dim
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = nn.Linear(args.input_dim, self.embedding_dim)
        self.embed_scale = math.sqrt(self.embedding_dim)
        self.dropout = args.dropout
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, self.embedding_dim,
        )

        # create encoder layer history
        self.history = CreateLayerHistory(args, is_encoder=True)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(self.embedding_dim)
        self.calculate_num = args.enc_calculate_num
        self.gate_linear = nn.Linear(2 * self.embedding_dim, 1)

    def forward(self, src_tokens):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # print('src的形状', src_tokens.size())
        src_tokens = src_tokens.unsqueeze(-1)
        if self.history is not None:
            self.history.clean()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        if self.embed_positions is not None:
            a = self.embed_positions(src_tokens)

            x += a

        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # add emb into history
        if self.history is not None:
            self.history.add(x)

        # encoder layers
        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()
            runge_kutta_list = []
            residual = x
            for step_size in range(int(self.calculate_num)):
                x = layer(x)
                runge_kutta_list.append(x)
                if int(self.calculate_num) == 4:
                    if step_size == 0 or step_size == 1:
                        # x = residual + 1 / 2 * x
                        x = residual + x
                    elif step_size == 2:
                        x = residual + x
                elif int(self.calculate_num) == 3:
                    if step_size == 0:
                        x = residual + 1 / 2 * x
                    elif step_size == 1:
                        x = residual - runge_kutta_list[1] + 2 * x
                elif int(self.calculate_num) == 2:
                    x = residual + x
            if self.calculate_num == 4:
                # RK4-block
                # x = residual + 1 / 6 * (
                # runge_kutta_list[0] + 2 * runge_kutta_list[1] + 2 * runge_kutta_list[2] + runge_kutta_list[3])
                x = residual + (
                        runge_kutta_list[0] + 2 * runge_kutta_list[1] + 2 * runge_kutta_list[2] + runge_kutta_list[3])

                # learnable alpha for RK4-block
                # x = residual + self.alpha[0]*runge_kutta_list[0] + self.alpha[1]*runge_kutta_list[1] + self.alpha[2]*runge_kutta_list[2] + self.alpha[3]*runge_kutta_list[3]
            elif self.calculate_num == 3:
                # RK3-block
                x = residual + (runge_kutta_list[0] + runge_kutta_list[1] + runge_kutta_list[2])
            elif self.calculate_num == 2:
                # learnbale coefficients for RK2-block with gated
                alpha = torch.sigmoid(self.gate_linear(torch.cat((runge_kutta_list[0], runge_kutta_list[1]), dim=-1)))
                x = residual + alpha * runge_kutta_list[0] + (1 - alpha) * runge_kutta_list[1]
                # RK2-block
                # x = residual + 1/2 * (runge_kutta_list[0] + runge_kutta_list[1])
                # learnable coefficients with initialized 1
                # x = residual + self.alpha[0] * runge_kutta_list[0] + self.alpha[1] * runge_kutta_list[1]
            elif self.calculate_num == 1:
                x = residual + runge_kutta_list[0]

            elif self.calculate_num == 2.1:
                x = residual + runge_kutta_list[0] + runge_kutta_list[1]

            elif self.calculate_num == 0:
                x = residual
            else:
                raise ValueError("invalid caculate num!")

            if self.history is not None:
                self.history.add(x)

        if self.history is not None:
            x = self.history.pop()

        if self.normalize:
            x = self.layer_norm(x)

        return x  # T x B x C





# In[7]:


class ODETransformerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = ODETransformerEncoder(args)
        # self.decoder = ODETransformerDecoder(args)
        self.seq_len = args.seq_len
        self.encoder_embed_dim = args.encoder_embed_dim
        self.batch_size = args.batch_size
        # 卷积处理多变量
        self.conv1 = nn.Conv1d(self.seq_len , self.seq_len, 7)
        # self.conv2 = nn.Conv1d(self.seq_len, self.seq_len, 2)
        # self.conv3 = nn.Conv1d(self.seq_len, self.seq_len, 2)

        self.fc = nn.Linear(args.encoder_embed_dim * args.seq_len, 1024)
        self.ffc = nn.Linear(1024, 1)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()

    def forward(self, src_tokens):
        # src_tokens = self.conv1(src_tokens)

        # src_tokens = self.conv2(src_tokens)
        #
        # src_tokens = self.conv3(src_tokens)

        src_tokens = src_tokens.squeeze(-1)

        encoder_out = self.silu(self.encoder(src_tokens)).transpose(0, 1).view(self.batch_size, self.seq_len * self.encoder_embed_dim)
        # print(encoder_out.size())
        decoder_out = self.silu(self.fc(encoder_out))
        decoder_out = self.silu(self.ffc(decoder_out))

        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dropout', default=0, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--relu-dropout', default=0, type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-dim', default=128, type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', default=128, type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', default=6, type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', default=8, type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-history-type', default=None, type=int, metavar='N',
                            help='')
        parser.add_argument('--seed', default=1, type=int, metavar='N',
                            help='pseudo random number generator seed')
        parser.add_argument('--max-source-positions', default=16, type=int, metavar='N',
                            help='')
        parser.add_argument('--max-target-positions', default=16, type=int, metavar='N',
                            help='')
        parser.add_argument('--input-dim', default=1, type=int, metavar='N',
                            help='')
        parser.add_argument('--seq-len', default=24, type=int, metavar='N',
                            help='')
        parser.add_argument('--embedding-dim', default=128, type=int, metavar='N',
                            help='')
        parser.add_argument('--max-relative-length', default=16, type=int, metavar='N',
                            help='')
        parser.add_argument('--attention-dropout', default=0, type=float, metavar='N',
                            help='')
        parser.add_argument('--enc-calculate-num', default=2, type=int, metavar='N',
                            help='')
        parser.add_argument('--device', default='0', type=int, metavar='N',
                            help='')
        parser.add_argument('--batch-size', default='128', type=int, metavar='N',
                            help='')


filepath = os.path.join(os.getcwd(), 'output')
# lop = 1
parser = argparse.ArgumentParser()
ODETransformerModel.add_args(parser)
num_cal = 4
model = ODETransformerModel(parser.parse_known_args()[0])
model.load_state_dict(torch.load("best_model_ett.pt"))
model.to(device=0)
input_window = 24
batch_size = 128
epochs = 30
lr = 0.00001
criterion = Lossfun()
# summary(model, input_size=(8, 5))

# train_data, val_data, test_data, train_mean, train_std = nn_seq_ms(batch_size, input_window)
train_data, val_data, test_data, all_data = nn_seq_ms(batch_size, input_window)
# train_data, val_data, test_data = nn_seq_mm(batch_size, 96, input_window)
# train_data, val_data, test_data, train_mean, train_std = nn_seq_us(batch_size, input_window)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
best_model = None
loss_train_lst = []
loss_val_lst = []
loss_test_lst = []
reach_count = 0
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    loss_train = train(model, optimizer, scheduler, criterion, train_data, epoch)
    loss_train_lst.append(np.mean(loss_train))
    val_loss_mean = evaluate(model, criterion, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f}'
          .format(epoch, (time.time() - epoch_start_time), val_loss_mean))
    print('-' * 89)

    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        torch.save(model.state_dict(), 'best_model_ett.pt')
        reach_count = 0
    else:
        reach_count += 1

    with torch.no_grad():
        for step_t, (t_x, t_y) in enumerate(val_data):
            t_x = t_x.to(0)
            t_y = t_y.to(0)
            validation_out = model(t_x)
            validation_batch_loss = criterion(validation_out, t_y)
    loss_val_lst.append(validation_batch_loss.item())
    # early stopping
    with torch.no_grad():
        for step_t, (t_x, t_y) in enumerate(test_data):
            t_x = t_x.to(0)
            t_y = t_y.to(0)
            test_out = model(t_x)
            test_batch_loss = criterion(test_out, t_y)
    loss_test_lst.append(test_batch_loss.item())

    if reach_count == 200:
        print('Early stopped at epoch', epoch)
        break
    # torch.save(model.state_dict(), 'best_model.pt')
    scheduler.step()

# print(loss_train1)
plt.figure(figsize=(9, 5))
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(fontproperties='Times New Roman', fontsize=14)
plt.yticks(fontproperties='Times New Roman', fontsize=14)
plt.xlabel('Epoch', fontproperties='Times New Roman', fontsize=14)
plt.ylabel('Log10-Loss', fontproperties='Times New Roman', fontsize=14)
plt.plot(np.log10(np.array(loss_train_lst)), color='blue', label='train_loss')
plt.plot(np.log10(np.array(loss_val_lst)), label='val_loss')
plt.plot(np.log10(np.array(loss_test_lst)), color='purple', label='test_loss')
plt.legend(ncol=1, loc='best', fontsize=14)
plt.savefig(filepath + '/loss.png')
# plt.show()
#
loss_train_lst = pd.DataFrame(loss_train_lst)
loss_train_lst.columns = ['loss_train']
loss_val_lst = pd.DataFrame(loss_val_lst)
loss_val_lst.columns = ['loss_val']
loss_test_lst = pd.DataFrame(loss_test_lst)
loss_test_lst.columns = ['loss_test']
loss1 = loss_train_lst.join(loss_val_lst, how='outer')
loss_all = loss1.join(loss_test_lst, how='outer')
loss_all.to_csv(filepath + '/loss.csv')

#
st = 'train'
train, train_truth = predict_future(model, train_data, st, 'ODETransformer')
train = pd.DataFrame(train)
train.columns = ['train']
train_truth = pd.DataFrame(train_truth)
train_truth.columns = ['truth_train']
train_result = train.join(train_truth, how='outer')
train_result.to_csv(filepath + '/train_result.csv')

st = 'val'
val, val_truth = predict_future(model, val_data, st, 'ODETransformer')
val = pd.DataFrame(val)
val.columns = ['val']
val_truth = pd.DataFrame(val_truth)
val_truth.columns = ['truth_val']
val_result = val.join(val_truth, how='outer')
val_result.to_csv(filepath + '/val_result.csv')
st = 'test'
pred, truth = predict_future(model, test_data, st, 'ODETransformer')
pred = pd.DataFrame(pred)
pred.columns = ['test']
truth = pd.DataFrame(truth)
truth.columns = ['truth_test']
test_result = pred.join(truth, how='outer')
test_result.to_csv(filepath + '/test_result.csv')
