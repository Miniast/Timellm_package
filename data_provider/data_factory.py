from data_provider.data_loader import DatasetCapacity
from torch.utils.data import DataLoader, Subset


def data_provider(args, data_path, flag):
    time_enc = 0 if args.embed != 'timeF' else 1
    args.freq = '15min'

    shuffle_flag = (flag != 'test')
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    data_set = DatasetCapacity(
        root_path=args.root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        time_enc=time_enc,
        freq=freq,
    )
    split_point = int(len(data_set) * 0.8)
    train_data = Subset(data_set, range(split_point))
    val_data = Subset(data_set, range(split_point, len(data_set)))
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return train_data, train_loader, val_data, val_loader
