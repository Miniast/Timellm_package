import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from loguru import logger


def create_sequences(data, seq_length, pred_length):
    sequences = []
    sample_nums = len(data) - seq_length - pred_length + 1
    for i in range(sample_nums):
        past_seq = data[i: i + seq_length]
        future_seq = data[i + seq_length: i + seq_length + pred_length]
        sequences.append((past_seq, future_seq))
    return sequences


def get_data_loader(device_id, df, batch_size, data_type, seq_length, pred_length):
    # only remain the 'cpu' and 'memory' col
    data_raw = df.loc[df['device'] == device_id, ['cpu', 'memory']].values
    seq_data = create_sequences(data_raw, seq_length, pred_length)
    X = torch.tensor(np.array([s[0] for s in seq_data]), dtype=torch.float32)
    Y = torch.tensor(np.array([s[1] for s in seq_data]), dtype=torch.float32)
    shuffle_flag = data_type == 'train'
    # 0.8 for train, 0.2 for test
    split = int(len(X) * 0.7)
    train_set = TensorDataset(X[:split], Y[:split])
    val_set = TensorDataset(X[split:], Y[split:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_flag)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MultiStepLSTM, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(output_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, future_steps):
        encoder_outputs, (hidden, cell) = self.encoder_lstm(x)
        decoder_input = x[:, -1, :]
        outputs = []
        for _ in range(future_steps):
            decoder_input = decoder_input.unsqueeze(1)
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            prediction = self.fc(decoder_output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))
            decoder_input = prediction
        return torch.cat(outputs, dim=1)


def smape_loss(y_true, y_pred):
    epsilon = 1e-8
    return torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true) + epsilon))


def train(model, train_loader, test_loader, epochs, device, pred_length, criterion, optimizer, save_path):
    model.to(device)
    best_test_loss = float('inf')
    patience = 6
    trigger_times = 0

    logger.info(f'Sample Number: {len(train_loader.dataset)}')
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, future_steps=pred_length)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch, future_steps=pred_length)
                loss = criterion(output, y_batch)
                test_loss += loss.item() * x_batch.size(0)
        test_loss /= len(test_loader.dataset)

        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"模型已保存至 {save_path}")
            trigger_times = 0
        else:
            trigger_times += 1
            logger.info(f"Early stop计数: {trigger_times}/{patience}")
            if trigger_times >= patience:
                logger.info("Early stop，停止训练。")
                break


def test(model, test_loader, device, criterion, pred_length):
    model.to(device)
    model.eval()
    # test_mse_loss, test_mae_loss, test_smape_loss = 0.0, 0.0, 0.0
    cpu_mae, cpu_mse, cpu_smape = 0.0, 0.0, 0.0
    mem_mae, mem_mse, mem_smape = 0.0, 0.0, 0.0
    mae_loss_fn = nn.L1Loss()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # print(x_batch.shape)
            output = model(x_batch, future_steps=pred_length)
            loss = criterion(output, y_batch)
            # x_batch: [16,96,2]
            cpu_true = output[:, :, 0:1]
            mem_true = output[:, :, 1:2]
            cpu_pred = y_batch[:, :, 0:1]
            mem_pred = y_batch[:, :, 1:2]
            # test_mse_loss += loss.item() * x_batch.size(0)
            # test_mae_loss += mae_loss_fn(output, y_batch).item() * x_batch.size(0)
            # test_smape_loss += smape_loss(output, y_batch).item() * x_batch.size(0)
            cpu_smape += smape_loss(cpu_true, cpu_pred).item() * x_batch.size(0)
            mem_smape += smape_loss(mem_true, mem_pred).item() * x_batch.size(0)

    # test_mse_loss /= len(test_loader.dataset)
    # test_mae_loss /= len(test_loader.dataset)
    # test_smape_loss /= len(test_loader.dataset)
    cpu_smape /= len(test_loader.dataset)
    mem_smape /= len(test_loader.dataset)

    # logger.info(f"Test MSE Loss: {test_mse_loss:.6f}, MAE Loss: {test_mae_loss:.6f}, SMAPE Loss: {test_smape_loss:.6f}")
    # return test_mae_loss, test_mse_loss, test_smape_loss
    return cpu_smape, mem_smape


def main():
    seq_length = 96
    pred_length = 96
    batch_size = 16
    epochs = 100
    hidden_size = 512
    learning_rate = 0.001
    device_num = 134
    file_path = './dataset/total.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Use: {device}")

    df = pd.read_csv(file_path)
    total_result = []
    for device_id in range(device_num):
        if device_id == 104 or device_id == '104':
            continue
        # start_point = 105
        # if device_id < start_point:
        #     continue
        logger.info(f'Training device {device_id}')
        save_path = f'./model_results/{device_id}_model.pth'
        train_loader, val_loader = get_data_loader(device_id, df, batch_size, 'train', seq_length, pred_length)
        # test_loader = get_data_loader(file_path, batch_size, 'test', seq_length, pred_length)

        # Initialize model
        model = MultiStepLSTM(input_size=2, hidden_size=hidden_size, output_size=2, num_layers=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if os.path.exists(save_path):
            state_dict = torch.load(save_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded successfully of {save_path}.")

        # train(model, train_loader, val_loader, epochs, device, pred_length, criterion, optimizer, save_path)

        # mae, mse, smape = test(model, val_loader, device, criterion, pred_length)
        cpu_smape, mem_smape = test(model, val_loader, device, criterion, pred_length)
        total_result.append({
            'device': device_id,
            'cpu_smape': cpu_smape,
            'mem_smape': mem_smape
        })

    result_df = pd.DataFrame(total_result)
    result_df.to_csv('./result/lstm_result.csv', index=False)


if __name__ == "__main__":
    main()
    # logger.info('?')
