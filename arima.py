import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA

# from sklearn.metrics import mean_squared_error, mean_absolute_error

# 你可以根据需要修改 (p, d, q) 的默认阶次
ORDER = (1, 1, 1)


def train_arima_model(train_series, order=(1, 1, 1)):
    """
    训练一个 ARIMA 模型并返回拟合结果
    train_series: pandas Series，训练用的单变量序列
    order: (p, d, q)
    """
    model = ARIMA(train_series, order=order)
    result = model.fit()
    return result


def evaluate_predictions(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    smape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    return mae, mse, smape


def main():
    # 假设你已有一个 total.csv，含有 columns: [device, date, cpu, memory, name]
    file_path = './dataset/total.csv'
    df = pd.read_csv(file_path)
    device_num = 134
    total_loss = {
        'device': [],
        'cpu_mae': [],
        'cpu_mse': [],
        'cpu_smape': [],
        'mem_mae': [],
        'mem_mse': [],
        'mem_smape': [],
    }
    for device_id in range(device_num):
        if device_id == 104 or device_id == '104':
            continue
        df_device = df[df['device'] == device_id].copy()
        df_device['date'] = pd.to_datetime(df_device['date'], unit='ms')
        df_device.sort_values(by='date', inplace=True)
        df_device.reset_index(drop=True, inplace=True)

        data_cpu = df_device['cpu']
        data_mem = df_device['memory']
        split_index = int(len(data_cpu) * 0.7)
        train_cpu = data_cpu[:split_index]
        test_cpu = data_cpu[split_index:]
        train_mem = data_mem[:split_index]
        test_mem = data_mem[split_index:]

        cpu_model = train_arima_model(train_cpu, order=ORDER)
        pred_length = len(test_cpu)
        cpu_forecast = cpu_model.forecast(steps=pred_length)
        cpu_mae, cpu_mse, cpu_smape = evaluate_predictions(test_cpu.values, cpu_forecast.values)

        mem_model = train_arima_model(train_mem, order=ORDER)
        mem_forecast = mem_model.forecast(steps=pred_length)
        mem_mae, mem_mse, mem_smape = evaluate_predictions(test_mem.values, mem_forecast.values)

        # if cpu_mse > 200 or mem_mse > 200:
        #     print(f"Device {device_id} has large error, skipping...")
        #     continue
        total_loss['device'].append(device_id)
        total_loss['cpu_smape'].append(cpu_smape)
        total_loss['cpu_mae'].append(cpu_mae)
        total_loss['cpu_mse'].append(cpu_mse)
        total_loss['mem_smape'].append(mem_smape)
        total_loss['mem_mae'].append(mem_mae)
        total_loss['mem_mse'].append(mem_mse)
        print(f"Device {device_id}:")
        print(f"CPU MAE: {cpu_mae:.4f}, MSE: {cpu_mse:.4f}, SMAPE: {cpu_smape:.4f}")
        print(f"Memory MAE: {mem_mae:.4f}, MSE: {mem_mse:.4f}, SMAPE: {mem_smape:.4f}")

    result_df = pd.DataFrame(total_loss)

    for key in total_loss:
        total_loss[key] = np.mean(total_loss[key])

    total_loss['device'] = 'total'
    result_df = pd.concat([result_df, pd.DataFrame([total_loss])], ignore_index=True)
    print("Total Losses:")
    print(f"CPU MAE: {total_loss['cpu_mae']:.4f}")
    print(f"CPU MSE: {total_loss['cpu_mse']:.4f}")
    print(f"CPU SMAPE: {total_loss['cpu_smape']:.4f}")
    print(f"Memory MAE: {total_loss['mem_mae']:.4f}")
    print(f"Memory MSE: {total_loss['mem_mse']:.4f}")
    print(f"Memory SMAPE: {total_loss['mem_smape']:.4f}")
    result_df.to_csv('./result/arima_result.csv', index=False)


if __name__ == "__main__":
    main()
