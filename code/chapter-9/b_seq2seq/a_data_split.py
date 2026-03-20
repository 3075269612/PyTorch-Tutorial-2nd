
"""
@file name  : a_data_split.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-19
@brief      : 训练、测试数据集划分
"""
import os
import random


def split_dataset(input_file, train_file, valid_file, split_ratio_):

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.shuffle(lines)

    split_index = int(len(lines) * split_ratio_)
    print("总共{}行，训练集:{}行， 测试集:{}行".format(len(lines), split_index, len(lines)-split_index))

    train_data = lines[:split_index]
    valid_data = lines[split_index:]

    with open(train_file, 'w', encoding='utf-8') as file:
        file.writelines(train_data)
    print(f"训练集保存成功，位于:{train_file}")

    with open(valid_file, 'w', encoding='utf-8') as file:
        file.writelines(valid_data)
    print(f"测试集保存成功，位于:{valid_file}")


if __name__ == '__main__':
    random.seed(42)
    import zipfile

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data", "cmn-eng")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 自动解压处理
    zip_path = os.path.join(base_dir, "cmn-eng.zip")
    path_raw = os.path.join(data_dir, "cmn.txt")
    
    if not os.path.exists(path_raw):
        if os.path.exists(zip_path):
            print(f"正在解压 {zip_path} ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
            print("解压完成")
        else:
            print(f"错误: 未找到 {path_raw} 且未找到压缩包 {zip_path}")
            # 尝试查找上一级目录
            zip_path_up = os.path.join(base_dir, "..", "cmn-eng.zip")
            if os.path.exists(zip_path_up):
                 print(f"正在解压 {zip_path_up} ...")
                 with zipfile.ZipFile(zip_path_up, 'r') as zf:
                    zf.extractall(data_dir)
            
    path_train = os.path.join(data_dir, "train.txt")
    path_test = os.path.join(data_dir, "test.txt")
    split_ratio = 0.8

    if os.path.exists(path_raw):
        split_dataset(path_raw, path_train, path_test, split_ratio)
    else:
        print("无法执行划分：找不到原数据文件 cmn.txt")
