#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as t
from utils import load_data, seed_all
from FGANomaly import FGANomalyModel, RNNAutoEncoder, MLPDiscriminator,QuantumDiscriminator
import os



seed_all(2021)


params = {
    'data_prefix': 'SMAP',#或者MSL,SMD,SWaT
    'val_size': 0.3, #划分30%的数据为验证集
    'batch_size': 256, #批次大小
    'stride': 1, #步长为1
    'window_size': 120,  #时间窗口120

    'z_dim': 10,
    'hidden_dim': 100,  # 50 for msl, smap and swat, 100 for wadi
    'rnn_hidden_dim': 100,  # 50 for msl, smap and swat, 100 for wadi
    'num_layers': 1,
    'bidirectional': True,
    'cell': 'gru',  # 'lstm' for msl, smap and swat, 'gru' for wadi

    'device': t.device('cuda:0' if t.cuda.is_available() else 'cpu'),
    'lr': 3e-4,
    'if_scheduler': True,  # whether use lr scheduler
    'scheduler_step_size': 5,
    'scheduler_gamma': 0.5,

    'epoch': 100,
    'early_stop': True,
    'early_stop_tol': 10,

    'weighted_loss': True,
    'strategy': 'linear',

    'adv_rate': 0.01,
    'dis_ar_iter': 1,

    'best_model_path': None,
    'result_path': None,
}

params['best_model_path'] = os.path.join('rnn_output','attention', params['data_prefix'], 'best_model')  # 后更新
params['result_path'] = os.path.join('rnn_output', 'attention',params['data_prefix'])  # 后更新
print(f"Using device: {params['device']}")

def main():
    data = load_data(data_prefix=params['data_prefix'],
                    val_size=params['val_size'],
                    window_size=params['window_size'],
                    stride=params['stride'],
                    batch_size=params['batch_size'],
                    dataloder=True)

    model = FGANomalyModel(ae=RNNAutoEncoder(inp_dim=data['nc'],
                                            z_dim=params['z_dim'],
                                            hidden_dim=params['hidden_dim'],
                                            rnn_hidden_dim=params['rnn_hidden_dim'],
                                            num_layers=params['num_layers'],
                                            bidirectional=params['bidirectional'],
                                            cell=params['cell']),
                            #dis_ar=MLPDiscriminator(inp_dim=data['nc'],hidden_dim=params['hidden_dim']),
                           dis_ar=QuantumDiscriminator(inp_dim=data['nc'], blocks=3),
                           data_loader=data, **params)
    model.train()
    model.test()


if __name__ == '__main__':
    for ss in ['log', 'linear', 'nlog', 'quadratic']:
        params['strategy'] = ss
        for ar in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            params['adv_rate'] = ar
            main()
