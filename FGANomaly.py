#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
import os
from time import time
from copy import deepcopy

import torch as t
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from torch.nn import MSELoss, BCELoss
import numpy as np
from sklearn.metrics import mean_squared_error
import pennylane as qml
import math

from utils import seed_all, metrics_calculate, AdaWeightedLoss


seed_all(2021)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        """
        Attention layer to compute weighted sum of encoder hidden states.

        Args:
            hidden_dim: Dimension of the hidden states.
        """
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_hidden, decoder_hidden):
        """
        Compute attention weights and context vector.

        Args:
            encoder_hidden: Encoder hidden states, shape [batch_size, seq_len, hidden_dim]
            decoder_hidden: Decoder hidden state, shape [batch_size, hidden_dim]

        Returns:
            context: Weighted sum of encoder hidden states, shape [batch_size, hidden_dim]
            weights: Attention weights, shape [batch_size, seq_len]
        """
        # Compute attention scores: score(hi) = W2·tanh(W1·hi)
        scores = self.W2(self.tanh(self.W1(encoder_hidden)))  # [batch_size, seq_len, 1]
        weights = self.softmax(scores.squeeze(-1))  # [batch_size, seq_len]
        # Compute context vector: weighted sum of encoder hidden states
        context = t.bmm(weights.unsqueeze(1), encoder_hidden).squeeze(1)  # [batch_size, hidden_dim]
        return context, weights

class RNNEncoder(nn.Module): # RNN编码器
    """
    An implementation of Encoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):
        """
        args:
            inp_dim: dimension of input value,输入数据的维度
            z_dim: dimension of latent code,潜在编码的维度
            hidden_dim: dimension of fully connection layer,全连接层的维度
            rnn_hidden_dim: dimension of rnn cell hidden states,RNN单元隐藏状态的维度
            num_layers: number of layers of rnn cell,RNN单元的维度
            bidirectional: whether use BiRNN ,cell是否使用双层RNN
            cell: one of ['lstm', 'gru', 'rnn'],RNN单元类型
        """
        super(RNNEncoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        #定义一个全连接层L1，将输入维度映射到隐藏维度
        self.linear1 = nn.Linear(inp_dim, hidden_dim)
        #定义全连接层L2,将RNN单元隐藏维度映射到潜在编码维度
        self.rnn_hidden_dim_mult = 2 if bidirectional else 1
        # 判断RNN类型
        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                                rnn_hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                                rnn_hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_dim,
                                rnn_hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional)
        self.linear2 = nn.Linear(self.rnn_hidden_dim * self.rnn_hidden_dim_mult, z_dim)

    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        self.rnn.flatten_parameters() #优化RNN参数的内存布局
        # inp shape: [seq_len, bsz, inp_dim]更改维度顺序，适应输入格式
        inp = inp.permute(1, 0, 2)
        #linear1处理输入，使用tanh激活函数
        rnn_inp = t.tanh(self.linear1(inp))
        rnn_out, _ = self.rnn(rnn_inp)
        # 通过线性层 linear2 处理RNN输出，并将维度置换回 [bsz, seq_len, z_dim]。
        z = self.linear2(rnn_out).permute(1, 0, 2)
        # Return both z and rnn_out (for attention)
        rnn_out = rnn_out.permute(1, 0, 2)  # [bsz, seq_len, rnn_hidden_dim * mult]
        return z,rnn_out


class RNNDecoder(nn.Module):
    """
    An implementation of Decoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):
        """
        args:
            Reference argument annotations of RNNEncoder.
        """
        super(RNNDecoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.rnn_hidden_dim_mult = 2 if bidirectional else 1

        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                                rnn_hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                                rnn_hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_dim,
                                rnn_hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional)
        self.linear2 = nn.Linear(self.rnn_hidden_dim * self.rnn_hidden_dim_mult, inp_dim)
        self.attention = Attention(self.rnn_hidden_dim * self.rnn_hidden_dim_mult)
        # 新增线性层，将 context 维度映射到 hidden_dim
        self.context_linear = nn.Linear(self.rnn_hidden_dim * self.rnn_hidden_dim_mult, hidden_dim)

    def forward(self,z,encoder_hidden):
        # z shape: [bsz, seq_len, z_dim]
        self.rnn.flatten_parameters()
        z = z.permute(1, 0, 2)
        rnn_inp = t.tanh(self.linear1(z))
        seq_len, bsz, _ = rnn_inp.shape
        rnn_outs = []
        # Initialize hidden state (optional: use encoder's final hidden state)
        for i in range(seq_len):
            # Get current input
            curr_inp = rnn_inp[i]  # [bsz, hidden_dim]
            # Compute attention context
            context, _ = self.attention(encoder_hidden, curr_inp)  # [bsz, rnn_hidden_dim * mult]
            # Combine input and context (optional: use context directly or concatenate)
            context_mapped = t.tanh(self.context_linear(context))  # [bsz, hidden_dim]
            rnn_inp_t = context_mapped.unsqueeze(0)  # [1, bsz, rnn_hidden_dim * mult]
            # RNN step
            rnn_out, _ = self.rnn(rnn_inp_t)
            rnn_outs.append(rnn_out)
        rnn_out = t.cat(rnn_outs, dim=0)  # [seq_len, bsz, rnn_hidden_dim * mult]
        re_x = self.linear2(rnn_out).permute(1, 0, 2)  # [bsz, seq_len, inp_dim]
        return re_x


class RNNAutoEncoder(nn.Module):
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):

        super(RNNAutoEncoder, self).__init__()

        self.encoder = RNNEncoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                    num_layers, bidirectional=bidirectional, cell=cell)
        self.decoder = RNNDecoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                    num_layers, bidirectional=bidirectional, cell=cell)

    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        z,encoder_hidden= self.encoder(inp)
        re_inp = self.decoder(z,encoder_hidden)
        return re_inp, z

class QuantumDiscriminator(nn.Module):
    def __init__(self,inp_dim,min_qubits=4,max_qubits=10,blocks=3):
        """
                Quantum Discriminator based on Variational Quantum Circuit.

                Args:
                    inp_dim: 输入特征维度 (df)
                    n_qubits: 量子比特数，默认为 10
                    blocks: 变分层数，默认为 3
       """
        super(QuantumDiscriminator,self).__init__()
        self.inp_dim = inp_dim
        self.blocks = blocks

        self.n_qubits = max(min_qubits,min(max_qubits,math.ceil(math.log2(inp_dim))))
        self.target_dim = 2 ** self.n_qubits
#       breakpoint()

        #将输入维度映射到量子态维度
        self.linear_pre = nn.Linear(inp_dim,self.target_dim)

        #量子电路参数
        self.weights = nn.Parameter(
            t.randn(blocks,self.n_qubits,3,dtype=t.float32)* 0.01,
            requires_grad=True
        )

        # Define Pennylane quantum device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

    def qml(self,x):
        """
                量子电路前向传播。

                Args:
           n.py         x: 输入张量，形状 [seq, 2^n]，dtype=float32
                Returns:
                    输出张量，形状 [seq]，概率值
        """
        #创建量子电路
        x_norm = x / t.norm(x, dim=1, keepdim=True)

        # Define the quantum circuit
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # Initialize quantum state from inputs
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)

            # Variational circuit
            for j in range(self.blocks):
                for k in range(self.n_qubits):
                    qml.RX(weights[j, k, 0], wires=k)
                    qml.RZ(weights[j, k, 1], wires=k)
                for k in range(self.n_qubits - 1):
                    qml.CZ(wires=[k, k + 1])  # Using CZ instead of ZZ for simplicity
            # Measure expectations
            z_exp = [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]
            x_exp = [qml.expval(qml.PauliX(k)) for k in range(self.n_qubits)]
            return z_exp + x_exp

        # 为整个批次计算电路输出
        result = circuit(x_norm, self.weights)
        # 转换为 PyTorch 张量并对量子比特的期望值求和
        result_tensor = t.stack(result, dim=1).to(dtype=t.float32)  # 形状 [batch_size, 2 * n_qubits]
        output = t.sum(result_tensor, dim=1)  # 形状 [batch_size]

        return t.sigmoid(output)


    def forward(self,inp):
        """
                前向传播。
                Args:
                    inp: 输入张量，形状 [seq, inp_dim]
                Returns:
                    输出张量，形状 [seq]
                """
        # 前处理：将 [seq, inp_dim] 转换为 [seq, 2^n]
        x = t.tanh(self.linear_pre(inp))  # [seq, 1024]
        # 量子电路计算
        probs = self.qml(x)  # [seq]
        return probs



#多层(MLP)感知机判别器
class MLPDiscriminator(nn.Module):
    def __init__(self, inp_dim, hidden_dim):
        super(MLPDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            #定义输入层到隐藏层的线性变换
            nn.Linear(inp_dim, hidden_dim),
            nn.Tanh(),
            #定义隐藏层到输出层的线性变换
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            #定义隐藏层到输出层的线性变换
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
      #  breakpoint()
        #print("name1=",inp.shape)
        seq, df = inp.shape
        c = self.dis(inp)
        #将结果重塑为seq的形状并返回
        return c.view(seq)


class FGANomalyModel(object):
    def __init__(self, ae, dis_ar, data_loader, **kwargs):
        self.params = kwargs
        self.print_param()
        self.print_model(ae, dis_ar)

        self.device = kwargs['device']
        self.lr = kwargs['lr']
        self.epoch = kwargs['epoch']
        self.window_size = kwargs['window_size']
        self.early_stop = kwargs['early_stop']
        self.early_stop_tol = kwargs['early_stop_tol']
        self.if_scheduler = kwargs['if_scheduler']

        self.adv_rate = kwargs['adv_rate']
        self.dis_ar_iter = kwargs['dis_ar_iter']

        self.weighted_loss = kwargs['weighted_loss']
        self.strategy = kwargs['strategy']

        self.ae = ae.to(self.device)
        self.dis_ar = dis_ar.to(self.device)
        self.data_loader = data_loader

        self.mse = MSELoss()
        self.bce = BCELoss()
        self.ada_mse = AdaWeightedLoss(self.strategy)

        self.ae_optimizer = Adam(params=self.ae.parameters(), lr=self.lr)
        self.ae_scheduler = lr_scheduler.StepLR(optimizer=self.ae_optimizer,
                                                step_size=kwargs['scheduler_step_size'],
                                                gamma=kwargs['scheduler_gamma'])
        self.ar_optimizer = Adam(params=self.dis_ar.parameters(), lr=self.lr)
        self.ar_scheduler = lr_scheduler.StepLR(optimizer=self.ar_optimizer,
                                                step_size=kwargs['scheduler_step_size'],
                                                gamma=kwargs['scheduler_gamma'])

        self.cur_step = 0
        self.cur_epoch = 0
        self.best_ae = None
        self.best_dis_ar = None
        self.best_val_loss = np.inf
        self.val_loss = None
        self.early_stop_count = 0
        self.re_loss = None
        self.adv_dis_loss = None
        self.time_per_epoch = None

    def train(self):
        print('*' * 20 + 'Start training' + '*' * 20)
        for i in range(self.epoch):
            self.cur_epoch += 1
            print(f"Epoch {self.cur_epoch}/{self.epoch} starts")
            self.train_epoch()
            self.validate()

            print(f"[Epoch {self.cur_epoch}] Training Loss: {self.re_loss:.5f}, "
                  f"Validation Loss: {self.val_loss:.5f}, Adversarial Loss: {self.adv_dis_loss:.5f}, "
                  f"Time per Epoch: {self.time_per_epoch:.5f} seconds")

            if self.val_loss < self.best_val_loss and self.best_val_loss - self.val_loss >= 1e-4:
                self.best_val_loss = self.val_loss
                self.best_ae = deepcopy(self.ae)
                self.best_dis_ar = deepcopy(self.dis_ar)
                self.save_best_model()
                self.early_stop_count = 0
                print("Saved best model")
            elif self.early_stop:
                self.early_stop_count += 1
                if self.early_stop_count > self.early_stop_tol:
                    print('*' * 20 + 'Early stop' + '*' * 20)
                    return
            else:
                pass

            print('[Epoch %d/%d] current training loss is %.5f, val loss is %.5f, adv loss is %.5f, '
                    'time per epoch is %.5f' % (i+1, self.epoch, self.re_loss, self.val_loss,
                                                self.adv_dis_loss, self.time_per_epoch))

    def train_epoch(self):
        start_time = time()
        print("Training for this epoch...")
        for x, _ in self.data_loader['train']:
            self.cur_step += 1
            x = x.to(self.device)

            for _ in range(self.dis_ar_iter):
                self.dis_ar_train(x)
            self.ae_train(x)

            if self.cur_step % 100 == 0:  # Every 100 steps print progress
                print(f"Step {self.cur_step}: Reconstruction Loss: {self.re_loss:.5f}, "
                      f"Adversarial Loss: {self.adv_dis_loss:.5f}")

        end_time = time()
        self.time_per_epoch = end_time - start_time
        if self.if_scheduler:
            self.ar_scheduler.step()
            self.ae_scheduler.step()

    def dis_ar_train(self, x):
        self.ar_optimizer.zero_grad()

        x = x.requires_grad_(True)
        re_x, z = self.ae(x)
        soft_label, hard_label = self.value_to_label(x, re_x)

        # 确保 re_x 也启用梯度追踪（如果 ae 的输出没有自动启用）
        re_x = re_x.requires_grad_(True)

        actual_normal = x[t.where(hard_label == 0)]
        re_normal = re_x[t.where(hard_label == 0)]

        # 验证张量是否启用梯度追踪
        assert actual_normal.requires_grad, "actual_normal does not require grad"
        assert re_normal.requires_grad, "re_normal does not require grad"

        actual_target = t.ones(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)
        re_target = t.zeros(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)

        re_logits = self.dis_ar(re_normal)
        actual_logits = self.dis_ar(actual_normal)

        # 调试 logits
        print("re_logits:", re_logits.shape, re_logits.min(), re_logits.max())
        print("actual_logits:", actual_logits.shape, actual_logits.min(), actual_logits.max())

        re_dis_loss = self.bce(input=re_logits, target=re_target)
        actual_dis_loss = self.bce(input=actual_logits, target=actual_target)

        dis_loss = re_dis_loss + actual_dis_loss

        # 验证损失
        print("dis_loss requires_grad:", dis_loss.requires_grad)

        dis_loss.backward()
        self.ar_optimizer.step()

        # Print discriminator loss every 100 steps
        if self.cur_step % 100 == 0:
            print(f"Discriminator Loss at Step {self.cur_step}: {dis_loss.item():.5f}")

    def dis_ar_train_no_filter(self, x):
        self.ar_optimizer.zero_grad()

        bsz, seq, fd = x.shape
        re_x, z = self.ae(x)

        re_x = re_x.contiguous().view(bsz * seq, fd)
        x = x.contiguous().view(bsz * seq, fd)

        actual_target = t.ones(size=(x.shape[0],), dtype=t.float, device=self.device)
        re_target = t.zeros(size=(re_x.shape[0],), dtype=t.float, device=self.device)

        re_logits = self.dis_ar(re_x)
        actual_logits = self.dis_ar(x)

        re_dis_loss = self.bce(input=re_logits, target=re_target)
        actual_dis_loss = self.bce(input=actual_logits, target=actual_target)

        dis_loss = re_dis_loss + actual_dis_loss
        dis_loss.backward()
        self.ar_optimizer.step()

    def ae_train(self, x):
        bsz, seq, fd = x.shape
        self.ae_optimizer.zero_grad()

        re_x, z = self.ae(x)

        # reconstruction loss
        if self.weighted_loss:
            self.re_loss = self.ada_mse(re_x, x, self.cur_step)
        else:
            self.re_loss = self.mse(re_x, x)

        # adversarial loss
        ar_inp = re_x.contiguous().view(bsz*seq, fd)
        actual_target = t.ones(size=(ar_inp.shape[0],), dtype=t.float, device=self.device)
        re_logits = self.dis_ar(ar_inp)
        self.adv_dis_loss = self.bce(input=re_logits, target=actual_target)

        loss = self.re_loss + self.adv_dis_loss * self.adv_rate
        loss.backward()
        self.ae_optimizer.step()

        # Print autoencoder loss every 100 steps
        if self.cur_step % 100 == 0:
            print(f"Autoencoder Loss at Step {self.cur_step}: Reconstruction Loss: {self.re_loss:.5f}, "
                  f"Adversarial Loss: {self.adv_dis_loss:.5f}")

    def validate(self):
        self.ae.eval()
        re_values = self.value_reconstruction_val(self.data_loader['val'], self.window_size)
        self.val_loss = mean_squared_error(y_true=self.data_loader['val'][:len(re_values)], y_pred=re_values)
        self.ae.train()
        print(f"Validation Loss: {self.val_loss:.5f}")


    def test(self, load_from_file=False):
        if load_from_file:
            self.load_best_model()

        self.best_ae.eval()

        test_x, test_y = self.data_loader['test']
        re_values = self.value_reconstruction_val(test_x, self.window_size, val=False)

        values = test_x[:len(re_values)]
        labels = test_y[:len(re_values)]
        metrics_calculate(values, re_values, labels)
        self.save_result(values, re_values, labels)

    def value_reconstruction_val(self, values, window_size, val=True):
        piece_num = len(values) // window_size
        reconstructed_values = []
        for i in range(piece_num):
            raw_values = values[i * window_size:(i + 1) * window_size, :]
            # Convert raw_values to numpy.ndarray for efficiency
            raw_values_np = np.array(raw_values)
            # Convert numpy.ndarray to PyTorch tensor
            raw_values_tensor = t.tensor(raw_values_np, dtype=t.float32).to(self.device)

            raw_values_tensor = raw_values_tensor.unsqueeze(0)  # 从 [window_size, fd] 到 [1, window_size, fd]

            if val:
                reconstructed_value_, z = self.ae(raw_values_tensor)
            else:
                reconstructed_value_, z = self.best_ae(raw_values_tensor)

            reconstructed_value_ = reconstructed_value_.squeeze().detach().cpu().tolist()
            reconstructed_values.extend(reconstructed_value_)
        return np.array(reconstructed_values)

    def value_to_label(self, values, re_values):
        with t.no_grad():
            errors = t.sqrt(t.sum((values - re_values) ** 2, dim=-1))
            error_mean = t.mean(errors, dim=-1)[:, None]
            error_std = t.std(errors, dim=-1)[:, None] + 1e-6
            z_score = (errors - error_mean) / error_std
            z_score = z_score * (1 - 1 / self.cur_epoch)

            soft_label = t.sigmoid(z_score)
            rand = t.rand_like(soft_label)
            hard_label = (soft_label > rand).float()
            return soft_label, hard_label

    def save_best_model(self):
        if not os.path.exists(self.params['best_model_path']):
            os.makedirs(self.params['best_model_path'])

        t.save(self.best_ae, os.path.join(self.params['best_model_path'],
                                            'ae_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))
        t.save(self.best_dis_ar, os.path.join(self.params['best_model_path'],
                                                'dis_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))

    def load_best_model(self):
        self.best_ae = t.load(os.path.join(self.params['best_model_path'], 'ae.pth'))
        self.best_dis_ar = t.load(os.path.join(self.params['best_model_path'], 'dis_ar.pth'))

    def save_result(self, values, re_values, labels):
        if not os.path.exists(self.params['result_path']):
            os.makedirs(self.params['result_path'])

        with h5py.File(os.path.join(self.params['result_path'], 'result_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.h5'), 'w') as f:
            f['values'] = values
            f['re_values'] = re_values
            f['labels'] = labels

    def print_param(self):
        print('*'*20+'parameters'+'*'*20)
        for k, v in self.params.items():
            print(k+' = '+str(v))
        print('*' * 20 + 'parameters' + '*' * 20)

    def print_model(self, ae, dis_ar):
        print(ae)
        print(dis_ar)

class QuantumAutoEncoder(nn.Module):
    def __init__(self, inp_dim, n_qubits=4, blocks=3, latent_n_qubits=2, device='cpu'):
        super(QuantumAutoEncoder, self).__init__()
        # Handle device as either string or torch.device
        if isinstance(device, str):
            self.device = t.device(device)
        else:
            self.device = device
        self.inp_dim = inp_dim
        self.n_qubits = n_qubits
        self.blocks = blocks
        self.latent_n_qubits = latent_n_qubits

        # Classical pre-processing layer
        self.linear_pre = nn.Linear(inp_dim, 2 ** n_qubits).to(self.device)

        # Quantum circuit parameters
        self.encoder_weights = nn.Parameter(
            t.randn(blocks, n_qubits, 3, dtype=t.float32) * 0.01,
            requires_grad=True
        ).to(self.device)
        self.decoder_weights = nn.Parameter(
            t.randn(blocks, n_qubits, 3, dtype=t.float32) * 0.01,
            requires_grad=True
        ).to(self.device)

        # Quantum device
        self.q_device = 'lightning.gpu' if t.cuda.is_available() and self.device.type == 'cuda' else 'default.qubit'
        self.dev = qml.device(self.q_device, wires=n_qubits)

    def qml_circuit(self, x, weights, is_encoder=True):
        x_norm = x / t.norm(x, dim=1, keepdim=True)

        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            for j in range(self.blocks):
                for k in range(self.n_qubits):
                    qml.RX(weights[j, k, 0], wires=k)
                    qml.RZ(weights[j, k, 1], wires=k)
                for k in range(self.n_qubits - 1):
                    qml.CZ(wires=[k, k + 1])
            if is_encoder:
                return [qml.expval(qml.PauliZ(k)) for k in range(self.latent_n_qubits)]
            else:
                return [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]

        # 应用 broadcast_expand 变换
        circuit = qml.transforms.broadcast_expand(circuit)
        result = circuit(x_norm, weights)
        return t.stack(result, dim=1).to(dtype=t.float32, device=self.device)

    def forward(self, x):
        x = t.tanh(self.linear_pre(x))
        latent = self.qml_circuit(x, self.encoder_weights, is_encoder=True)
        latent_padded = t.zeros((x.shape[0], self.n_qubits), device=self.device)
        latent_padded[:, :self.latent_n_qubits] = latent
        recon = self.qml_circuit(latent_padded, self.decoder_weights, is_encoder=False)
        reconstructed = t.tanh(self.linear_pre(recon))
        return reconstructed, latent
