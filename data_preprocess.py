import ast
import csv
import pandas as pd
import numpy as np
import pickle
import os
import json
from pickle import dump

# from tfsnippet.utils import makedirs

output_folder = 'processed'
# makedirs(output_folder, exist_ok=True)


#接受4个参数：数据集类别，具体的文件名，数据集名称，数据集所在的根目录路径
def load_and_save(category, filename, dataset, dataset_folder):
    #读取本地文件，先拼接文件路径，然后将从文本文件读取的数据（通常为CSV)返回numpy数组
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32, #数据转为32位浮点数
                         delimiter=',')  #由逗号分离
    print(dataset, category, filename, temp.shape)
    #将 temp 数组保存为 pickle 文件，以便后续快速加载（比 CSV 读取更快）。
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)

#根据输入的dataset参数加载不同的数据集，处理数据与标签，保存为pickle文件
def load_data(dataset):

    if dataset == 'SMAP' or dataset == 'MSL':
        
        dataset_folder = 'data'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]    #一行行的列表
        res = sorted(res, key=lambda k: k[0]) #按列表中第一个元素排序
        """
        将labeled_anomalies.csv文件中的内容分解为
        [
            ['P-1', 'SMAP', '[[2149, 2349], [4536, 4844], [3539, 3779]]', '[contextual, contextual, contextual]', '8505'],
            ['S-1', 'SMAP', '[[5300, 5747]]', '[point]', '7331']
        ]
        """

        label_folder = os.path.join(dataset_folder, 'test_label')
        # makedirs(label_folder, exist_ok=True)  #递归创建目录
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])  #对字符串进行类型转换
            length = int(row[-1])   #-1表示最后一行的数据
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels) #转换成数组
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join('processed', dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)
        """
        对文件的处理结果：
        对于 dataset = 'SMAP':
        P-1:8505 个标签，异常区间 [2149:2350], [4536:4845], [3539:3780] 为 True。
        S-1:7331 个标签，异常区间 [5300:5748] 为 True。
        总标签：长度 15836 的布尔数组。
        """

        #加载和保存测试数据
        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)
            """
            category = 'train'：data 累积所有训练数据。
                P-1：加载 data/train/P-1.npy
                S-1：加载 data/train/S-1.npy
            category = 'test'：data 累积所有测试数据。
                P-1：加载 data/test/P-1.npy
                S-1：加载 data/test/S-1.npy
                到SMAP_train.pkl 或 SMAP_test.pkl文件中
            """

        for c in ['train', 'test']:
            concatenate_and_save(c)

    elif dataset == 'SWaT':
        dataset_folder = 'SWaT'
        # Load attack data (discard normal data as per experiment setup)
        attack_file = os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.xlsx')
        attack = pd.read_excel(attack_file, header=1)

        # Process timestamp and labels
        #attack['Timestamp'] = pd.to_datetime(attack[' Timestamp'])
        labels = [float(label != 'Normal') for label in attack['Normal/Attack'].values]
        labels = np.array(labels)

        # Remove unnecessary columns
        del attack['Normal/Attack']
        del attack[' Timestamp']

        # Handle missing values and ensure float type
        attack = attack.fillna(method='ffill').fillna(method='bfill')
        attack = attack.astype(float)

        # Evenly split into training and testing sets
        total_samples = len(attack)
        split_idx = total_samples // 2
        train_data = attack.iloc[:split_idx].values
        test_data = attack.iloc[split_idx:].values
        test_labels = labels[split_idx:]

        # Save processed data
        print(f"{dataset} train {train_data.shape}")
        print(f"{dataset} test {test_data.shape}")
        print(f"{dataset} test_label {test_labels.shape}")
        with open(os.path.join(output_folder, f"{dataset}_train.pkl"), "wb") as f:
            dump(train_data,f)
        with open(os.path.join(output_folder, f"{dataset}_test.pkl"), "wb") as f:
            dump(test_data, f)
        with open(os.path.join(output_folder, f"{dataset}_test_label.pkl"), "wb") as f:
            dump(test_labels, f)

    elif dataset == 'WADI':
        dataset_folder = './dataset/wadi_raw_data'
        attack_data_file = os.path.join(dataset_folder, 'WADI_attackdata.csv')

        # 读取攻击数据
        attack = pd.read_csv(attack_data_file)

        # 打印原始Date和Time列以调试
        print("First two rows of Date and Time:")
        print(attack[['Date', 'Time']].head(2))

        # 合并Date和Time列为时间戳
        try:
            attack['Timestamp'] = pd.to_datetime(attack['Date'] + ' ' + attack['Time'],
                                                 format='%m/%d/%Y %I:%M:%S.%f %p')
        except Exception as e:
            print(f"Error parsing Date and Time: {e}")
            raise

        print("First two timestamps:")
        print(attack['Timestamp'].head(2))
        print("Data time range:", attack['Timestamp'].min(), "to", attack['Timestamp'].max())

        # 读取攻击时间段
        with open(os.path.join(dataset_folder, 'time.json'), 'r') as f:
            event_log = json.load(f)

        # 初始化标签（默认全为0，即正常）
        labels = np.zeros(len(attack), dtype=bool)

        # 根据时间段生成标签
        for key, event in event_log.items():
            try:
                # 解析时间段
                date = pd.to_datetime(event[0], format='%Y/%m/%d')
                start_time = pd.to_datetime(f"{event[0]} {event[1]}", format='%Y/%m/%d %H:%M:%S')
                end_time = pd.to_datetime(f"{event[0]} {event[2]}", format='%Y/%m/%d %H:%M:%S')

                # 标记时间段内的数据为异常（1）
                labels[(attack['Timestamp'] >= start_time) & (attack['Timestamp'] <= end_time)] = True
            except Exception as e:
                print(f"Error processing event {key}: {e}")
                continue

        # 验证标签生成
        print('Total number of anomalies:', labels.sum())
        if labels.sum() == 0:
            print("Warning: No anomalies detected. Check time formats or date ranges in time.json")

        # 删除不必要的列
        data_columns = [col for col in attack.columns if col not in ['Row', 'Date', 'Time', 'Timestamp']]
        attack_data = attack[data_columns]

        # 处理缺失值并转换为numpy数组
        attack_data = np.nan_to_num(np.asarray(attack_data, dtype=np.float32))

        # 划分训练集和测试集
        split_index = len(attack_data) // 2
        train_data = attack_data[:split_index]
        test_data = attack_data[split_index:]
        test_labels = labels[split_index:]

        # 打印数据形状以验证
        print('WADI train', train_data.shape)
        print('WADI test', test_data.shape)
        print('WADI test_label', test_labels.shape)
        print('Number of anomalies in test labels:', test_labels.sum())

        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 保存为pkl文件
        with open(os.path.join(output_folder, 'WADI_train.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(output_folder, 'WADI_test.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        with open(os.path.join(output_folder, 'WADI_test_label.pkl'), 'wb') as f:
            pickle.dump(test_labels, f)

    elif dataset == 'SMD':
        dataset_folder = 'SMD'
        train_data = []
        test_data = []
        test_labels = []

        # 遍历所有机器
        machines = [os.listdir(os.path.join(dataset_folder, 'train'))[0]]
        for machine in machines:
            machine_id = machine.replace('.txt','')
            # 加载训练数据
            train = pd.read_csv(os.path.join(dataset_folder, 'train', machine), header=None)
            train = train.fillna(method='ffill').fillna(method='bfill').astype(np.float32)
            train_data.append(train.values)
            # 加载测试数据和标签
            test = pd.read_csv(os.path.join(dataset_folder, 'test', f'{machine_id}.txt'), header=None)
            test = test.fillna(method='ffill').fillna(method='bfill').astype(np.float32)
            test_data.append(test.values)
            label = pd.read_csv(os.path.join(dataset_folder, 'test_label', f'{machine_id}.txt'), header=None)
            test_labels.append(label.values.flatten().astype(np.int32))

        # 合并数据
        train_data = np.vstack(train_data)
        test_data = np.vstack(test_data)
        test_labels = np.concatenate(test_labels)

        print(dataset, 'train', train_data.shape)
        print(dataset, 'test', test_data.shape)
        print(dataset, 'test_label', test_labels.shape)


        with open(os.path.join(output_folder, dataset + "_train.pkl"), "wb") as f:
            dump(train_data, f)
        with open(os.path.join(output_folder, dataset + "_test.pkl"), "wb") as f:
            dump(test_data, f)
        with open(os.path.join(output_folder, dataset + "_test_label.pkl"), "wb") as f:
            dump(test_labels, f)



    elif dataset == 'MITDB':
        import wfdb
        id  = 100
        record = wfdb.rdrecord('./mitdb/mitdb/' + str(id), sampfrom=0, sampto=650000, physical=False)
        annotation = wfdb.rdann('./mitdb/mitdb/' + str(id), 'atr')
        # 生成所有数据标签
        ventricular_signal = record.d_signal
        beat_types = annotation.symbol
        beat_positions = annotation.sample
        # 对数据进行裁剪----保证数据的可测试
        length_set = 30000
        ventricular_signal = ventricular_signal[0:length_set]
        beat_types_temp = []
        beat_positions_temp = []
        for i in range(len(beat_positions)):
            if beat_positions[i] < length_set:
                beat_types_temp.append(beat_types[i])
                beat_positions_temp.append(beat_positions[i])
            else:
                break
        beat_types = beat_types_temp
        beat_positions = beat_positions_temp
        labels = []
        for i in range(len(ventricular_signal)):
            labels.append(True)
        for i in range(len(beat_types)):
            if beat_types[i] != "N":
                if i == 0:
                    for j in range(beat_positions[i+1]):
                        labels[j] = False
                elif i == len(beat_types)-1:
                    for j in range(beat_positions[i-1],len(ventricular_signal)):
                        labels[j] = False
                else:
                    for j in range(beat_positions[i-1],beat_positions[i+1]):
                        labels[j] = False
        labels = np.asarray(labels)  # 转换成数组
        # 对所有数据进行7:3划分训练集和测试集，其中训练集没有标签，测试集有标签
        k = int(len(ventricular_signal) * 0.7)
        train_data = ventricular_signal[0:k, :]
        test_data = ventricular_signal[k:-1, :]
        labels = labels[k:-1]
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)
        print(dataset, "train_data", train_data.shape)
        print(dataset, "test_data", test_data.shape)
        with open(os.path.join(output_folder, dataset + "_" + "train" + ".pkl"), "wb") as file:
            dump(train_data, file)
        with open(os.path.join(output_folder, dataset + "_" + "test" + ".pkl"), "wb") as file:
            dump(test_data, file)

    elif dataset == 'WRIST':
        import wfdb
        id  = "s1_low_resistance_bike"
        record = wfdb.rdrecord('./PPG/' + str(id), physical=False, channels=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14])
        annotation = wfdb.rdann('./PPG/' + str(id), 'atr')
        # 生成所有数据标签
        ventricular_signal = record.d_signal
        beat_types = annotation.symbol
        beat_positions = annotation.sample
        # 对数据进行裁剪----保证数据的可测试
        length_set = 10000
        ventricular_signal = ventricular_signal[0:length_set]
        beat_types_temp = []
        beat_positions_temp = []
        for i in range(len(beat_positions)):
            if beat_positions[i] < length_set:
                beat_types_temp.append(beat_types[i])
                beat_positions_temp.append(beat_positions[i])
            else:
                break
        beat_types = beat_types_temp
        beat_positions = beat_positions_temp
        labels = []
        for i in range(len(ventricular_signal)):
            labels.append(True)
        for i in range(len(beat_types)):
            if beat_types[i] != "N":
                if i == 0:
                    for j in range(beat_positions[i+1]):
                        labels[j] = False
                elif i == len(beat_types)-1:
                    for j in range(beat_positions[i-1],len(ventricular_signal)):
                        labels[j] = False
                else:
                    for j in range(beat_positions[i-1],beat_positions[i+1]):
                        labels[j] = False
        labels = np.asarray(labels)  # 转换成数组
        # 对所有数据进行7:3划分训练集和测试集，其中训练集没有标签，测试集有标签
        k = int(len(ventricular_signal) * 0.7)
        train_data = ventricular_signal[0:k, :]
        test_data = ventricular_signal[k:-1, :]
        labels = labels[k:-1]
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)
        print(dataset, "train_data", train_data.shape)
        print(dataset, "test_data", test_data.shape)
        with open(os.path.join(output_folder, dataset + "_" + "train" + ".pkl"), "wb") as file:
            dump(train_data, file)
        with open(os.path.join(output_folder, dataset + "_" + "test" + ".pkl"), "wb") as file:
            dump(test_data, file)

#
if __name__ == '__main__':
    datasets = ['SMAP', 'MSL', 'SWaT', 'WADI', 'MITDB', 'WRIST','SMD']
    commands = ['SMD']
    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
    else:
        print("""
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMAP', 'MSL', 'SWaT', 'WADI','SMD']
        """)
