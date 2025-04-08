import pandas as pd
import json
import numpy as np

with open('./result/capacity_val_result.json') as f:
    data = json.load(f)

def smape_loss(pred, true):
    # assume pred and true are numpy arrays sized (n,)
    return np.mean(np.abs(pred - true) / (np.abs(pred) + np.abs(true)))

records_loss = []
batch_size = 2
for record in data:
    device_batch, pred_batch, true_batch = record['device'], record['pred'], record['true']

    for index in range(batch_size):
        device = device_batch[index]
        pred = pred_batch[index]
        true = true_batch[index]
        # pred, true: list shaped (96,2), each element is a list of [cpu, mem]
        cpu_pred = np.array(pred)[:, 0]
        mem_pred = np.array(pred)[:, 1]
        cpu_true = np.array(true)[:, 0]
        mem_true = np.array(true)[:, 1]

        cpu_smape = smape_loss(cpu_pred, cpu_true)
        mem_smape = smape_loss(mem_pred, mem_true)
        cpu_mae = np.mean(np.abs(cpu_pred - cpu_true))
        mem_mae = np.mean(np.abs(mem_pred - mem_true))
        cpu_mse = np.mean((cpu_pred - cpu_true) ** 2)
        mem_mse = np.mean((mem_pred - mem_true) ** 2)

        records_loss.append({
            'device': device,
            'cpu_smape': cpu_smape,
            'cpu_mae': cpu_mae,
            'cpu_mse': cpu_mse,
            'mem_smape': mem_smape,
            'mem_mae': mem_mae,
            'mem_mse': mem_mse
        })

df = pd.DataFrame(records_loss)
# df.to_csv('pred_result_2025-01-23.csv', index=False)
# count smape loss levels: 0.01, 0.02, ... , 0.19, 0.20, 0.30, 0.40, 0.50, others
smape_levels = [0.01 * i for i in range(20)] + [0.20, 0.30, 0.40, 0.50]
cpu_smape_counts = []
mem_smape_counts = []
for level in smape_levels:
    cpu_smape_counts.append(df[df['cpu_smape'] <= level].shape[0])
    mem_smape_counts.append(df[df['mem_smape'] <= level].shape[0])

cpu_smape_counts.append(df.shape[0])
mem_smape_counts.append(df.shape[0])

# erase the first element of smape_levels
df_cpu = pd.DataFrame({
    'smape_level': smape_levels[1:] + ['>0.50'],
    'count': np.diff(cpu_smape_counts),
    'count_ratio': np.diff(cpu_smape_counts) / df.shape[0],
    'count_cumsum': cpu_smape_counts[1:],
    'sum_ratio': np.array(cpu_smape_counts[1:]) / df.shape[0]
})
df_mem = pd.DataFrame({
    'smape_level': smape_levels[1:] + ['>0.50'],
    'count': np.diff(mem_smape_counts),
    'count_ratio': np.diff(mem_smape_counts) / df.shape[0],
    'count_cumsum': mem_smape_counts[1:],
    'sum_ratio': np.array(mem_smape_counts[1:]) / df.shape[0]
})
df_cpu['count_ratio'] = df_cpu['count_ratio'].map(lambda x: '{:.2%}'.format(x))
df_mem['count_ratio'] = df_mem['count_ratio'].map(lambda x: '{:.2%}'.format(x))
df_cpu['sum_ratio'] = df_cpu['sum_ratio'].map(lambda x: '{:.2%}'.format(x))
df_mem['sum_ratio'] = df_mem['sum_ratio'].map(lambda x: '{:.2%}'.format(x))
print(df_cpu)
print(df.describe())