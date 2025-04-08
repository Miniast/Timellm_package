import argparse
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 30)

# 指定 Excel 文件的路径
parser = argparse.ArgumentParser(description='Time-LLM data prepare')
parser.add_argument('--excel_file', type=str, default='./dataset/data.xlsx', help='excel file path')
args = parser.parse_args()


def process_df(input_df, name):
    input_df.columns = ['cpu', 'memory']

    cpu_df = input_df['cpu'].str.replace('"', '').str.replace(',', '').str.split(':', expand=True)
    memory_df = input_df['memory'].str.replace('"', '').str.replace(',', '').str.split(':', expand=True)

    cpu_time = cpu_df[0].to_list()
    cpu_time = [int(x) for x in cpu_time if not pd.isna(x)]
    cpu_time.sort()
    memory_time = memory_df[0].to_list()
    memory_time = [int(x) for x in memory_time if not pd.isna(x)]
    memory_time.sort()

    start_time = min(cpu_time[0], memory_time[0])
    end_time = max(cpu_time[-1], memory_time[-1])

    slot = 15 * 60 * 1000
    if (end_time - start_time) % slot != 0:
        raise ValueError("时间戳不是15分钟的整数倍")

    time_cols = pd.Series(list(range(start_time, end_time + slot, slot)))

    cpu_df.dropna(subset=0, inplace=True)
    memory_df.dropna(subset=0, inplace=True)
    cpu_df[0] = cpu_df[0].astype('int64')
    memory_df[0] = memory_df[0].astype('int64')

    time_cols = time_cols.to_frame()
    df = pd.merge(time_cols, cpu_df, left_on=0, right_on=0, how='left')
    df = pd.merge(df, memory_df, left_on=0, right_on=0, how='left')
    df.columns = ['date', 'cpu', 'memory']
    df['cpu'] = df['cpu'].astype('float64')
    df['memory'] = df['memory'].astype('float64')
    df['cpu'] = df['cpu'].interpolate(method='linear').round(decimals=2)
    df['memory'] = df['memory'].interpolate(method='linear').round(decimals=2)
    df['name'] = name

    return df


def read_excel(excel_file):
    # 读取 Excel 文件中所有的 sheet
    try:
        xls = pd.ExcelFile(excel_file)
    except FileNotFoundError:
        print(f"文件未找到: {excel_file}")
        exit(1)
    except Exception as e:
        print(f"读取 Excel 文件时出错: {e}")
        exit(1)

    sheets = sorted(xls.sheet_names)
    # 遍历所有 sheet，并保存为 CSV
    dfs = []
    for index, sheet_name in enumerate(sheets):
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df = process_df(df, sheet_name)

            # process_df 返回的df: ['date', 'cpu', 'memory', 'name'], 在最前列插入'device'列，值为index
            df.insert(0, 'device', index)
            dfs.append(df)
            print(f"处理 sheet '{sheet_name}' 完成")
        except Exception as e:
            print(f"处理 sheet '{sheet_name}' 时出错: {e}")
    output_csv = pd.concat(dfs)
    output_csv.to_csv('./dataset/total.csv', index=False)

    return dfs


def create_dataset(dfs):
    train_data = []
    val_data = []
    test_data = []
    pos_base = 0
    for df in dfs:
        device = f'{df["device"].values[0]}-{df["name"].values[0]}'
        # 在train mode下，每隔48行做pos记录，直到pos+96+96越界, 形成一个seq_begins数组[0, 48, 96, ...]
        seqs = [(device, pos + pos_base) for pos in range(0, len(df), 48) if pos + 192 <= len(df)]
        train_data.extend(seqs[:int(len(seqs) * 0.8)])
        val_data.extend(seqs[int(len(seqs) * 0.8):])
    
        # 在test mode下，每个df只取一个最后seq, 不需要96的pred部分
        test_data.append((device, pos_base + len(df) - 96))
        pos_base += len(df)
    train_data_df = pd.DataFrame(train_data)
    train_data_df.to_csv('./dataset/train_data_index.csv', index=False, header=False)
    val_data_df = pd.DataFrame(val_data)
    val_data_df.to_csv('./dataset/val_data_index.csv', index=False, header=False)

    test_data_df = pd.DataFrame(test_data)
    test_data_df.to_csv('./dataset/test_data_index.csv', index=False, header=False)

def main():
    # dfs = read_excel(args.excel_file)
    dfs = pd.read_csv('./dataset/total.csv')
    dfs = [group.copy() for _, group in dfs.groupby('device')]
    create_dataset(dfs)


if __name__ == '__main__':
    main()
