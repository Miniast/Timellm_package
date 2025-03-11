import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(data, seq_length, pred_length):
    sequences = []
    sample_nums = len(data) - seq_length - pred_length + 1
    for i in range(sample_nums):
        past_seq = data[i: i + seq_length]
        future_seq = data[i + seq_length: i + seq_length + pred_length]
        sequences.append((past_seq, future_seq))
    return sequences


def get_data_loader(file_path, batch_size, data_type, seq_length, pred_length):
    df = pd.read_csv(file_path)
    # train_index_df = pd.read_csv('./dataset/train_data_index.csv')
    # val_index_df = pd.read_csv('./dataset/val_data_index.csv')
    # data_raw = df.values
    # seq_data = create_sequences(data_raw, seq_length, pred_length)
    # X = torch.tensor(np.array([s[0] for s in seq_data]), dtype=torch.float32)
    # Y = torch.tensor(np.array([s[1] for s in seq_data]), dtype=torch.float32)
    # dataset = TensorDataset(X, Y)
    # shuffle_flag = data_type == 'train'
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag)
    # return loader
    device_num = df['device'].nunique()
    for device in range(device_num):
        # only remain the 'cpu' and 'memory' col
        data_raw =

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
    patience = 10
    trigger_times = 0

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

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存至 {save_path}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"触发早停计数: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("早停触发，停止训练。")
                break


def test(model, test_loader, device, criterion, pred_length):
    model.to(device)
    model.eval()
    test_mse_loss, test_mae_loss, test_smape_loss = 0.0, 0.0, 0.0
    mae_loss_fn = nn.L1Loss()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch, future_steps=pred_length)
            loss = criterion(output, y_batch)
            test_mse_loss += loss.item() * x_batch.size(0)
            test_mae_loss += mae_loss_fn(output, y_batch).item() * x_batch.size(0)
            test_smape_loss += smape_loss(output, y_batch).item() * x_batch.size(0)
    test_mse_loss /= len(test_loader.dataset)
    test_mae_loss /= len(test_loader.dataset)
    test_smape_loss /= len(test_loader.dataset)
    print(f"Test MSE Loss: {test_mse_loss:.6f}, MAE Loss: {test_mae_loss:.6f}, SMAPE Loss: {test_smape_loss:.6f}")


def main():
    seq_length = 96
    pred_length = 96
    batch_size = 16
    epochs = 100
    hidden_size = 512
    learning_rate = 0.001
    save_path = './best_model.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # train_loader = get_data_loader('./dataset/train.csv', batch_size, 'train', seq_length, pred_length)
    # test_loader = get_data_loader('./dataset/test.csv', batch_size, 'test', seq_length, pred_length)
    # train_loader, val_loader =

    model = MultiStepLSTM(input_size=4, hidden_size=hidden_size, output_size=4, num_layers=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, test_loader, epochs, device, pred_length, criterion, optimizer, save_path)

    # if os.path.exists(save_path):
    #     state_dict = torch.load(save_path, map_location=device, weights_only=True)
    #     model.load_state_dict(state_dict)
    #     print("已加载最佳模型进行测试。")
    #
    # test(model, test_loader, device, criterion, pred_length)


if __name__ == "__main__":
    main()
