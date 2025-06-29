import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ts2vec import TS2Vec
from torch.utils.tensorboard import SummaryWriter
import tasks
import os
from datetime import datetime
import datautils

plt.switch_backend('Agg')

def load_and_preprocess_data(train_ratio=0.8, val_ratio=0.2, random_seed=42):
    np.random.seed(random_seed)
    xls = pd.ExcelFile("output.xlsx")
    sheets = xls.sheet_names
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

    for sheet in sheets:
        data = pd.read_excel(xls, sheet_name=sheet, header=None)
        dataX = data.iloc[:, 1:].values.astype(np.float32)
        dataY = data.iloc[:, 0].values.astype(np.int64)
        N = len(dataY)

        indices = np.random.permutation(N)
        num_test = int(N * (1 - train_ratio))
        test_indices = indices[:num_test]
        train_val_indices = indices[num_test:]

        num_val = int(len(train_val_indices) * val_ratio)
        val_indices = train_val_indices[:num_val]
        train_indices = train_val_indices[num_val:]

        X_train.append(dataX[train_indices])
        y_train.append(dataY[train_indices])
        X_val.append(dataX[val_indices])
        y_val.append(dataY[val_indices])
        X_test.append(dataX[test_indices])
        y_test.append(dataY[test_indices])

    def concat_and_convert(arr_list):
        arr = np.concatenate(arr_list)
        return arr.reshape(arr.shape[0], arr.shape[1], 1)  # [batch, seq_len, features]

    X_train = concat_and_convert(X_train)
    X_val = concat_and_convert(X_val)
    X_test = concat_and_convert(X_test)

    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)
    y_test = np.concatenate(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

class TrainingMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        self.loss_history = []

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_embedding(self, features, labels, step):
        self.writer.add_embedding(features, metadata=labels, tag=f'embeddings/step_{step}', global_step=step)

    def log_confusion_matrix(self, y_true, y_pred, classes, step):
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        self.writer.add_figure('ConfusionMatrix', fig, step)
        plt.close()

    def close(self):
        self.writer.close()


def setup_logging(run_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join("training_logs", f"{timestamp}_{run_name}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def main():
    log_dir = setup_logging("TS2Vec")
    monitor = TrainingMonitor(log_dir)

    # 加载数据（
    train_data, train_labels, test_data, test_labels = datautils.load_UCR('ECG200')
    # 处理NaN值
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)

    print("train_data",train_data.shape)
    print("test_data",test_data.shape)

    print("train_labels",train_labels.shape)


    # X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    #
    # # 转换为numpy数组并处理形状
    # train_data = np.array(X_train)
    # test_data = np.array(X_test)
    #
    # test_labels = np.array(y_test)
    # train_labels = np.array(y_train)


    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS2Vec(
        input_dims=1,
        output_dims=256,
        hidden_dims=128,
        depth=6,
        device=device,
        batch_size=32,
        lr=0.001,
        max_train_length=512
    )

    # 梯度记录函数
    def log_gradients(step):
        for name, param in model._net.named_parameters():
            if param.grad is not None:
                monitor.log_histogram(f"Gradients/{name}", param.grad, step)

    # 训练回调
    def create_callback(monitor, log_dir):
        def _callback(model, epoch_loss):
            monitor.loss_history.append(epoch_loss)
            monitor.log_scalar('Loss/Train', epoch_loss, model.n_epochs)
            log_gradients(model.n_epochs)

            if model.n_epochs % 10 == 0:
                model.save(os.path.join(log_dir, f'model_epoch{model.n_epochs}.pkl'))

            if model.n_epochs % 5 == 0 and len(monitor.loss_history) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(monitor.loss_history, alpha=0.3)
                if len(monitor.loss_history) >= 5:
                    smoothed = np.convolve(monitor.loss_history, np.ones(5) / 5, mode='valid')
                    plt.plot(range(4, len(monitor.loss_history)), smoothed)
                plt.savefig(os.path.join(log_dir, 'loss_curve.png'), dpi=120)
                plt.close()

        return _callback

    # 训练
    print("\nStarting training...")
    loss_log = model.fit(
        train_data,
        n_epochs=10,
        verbose=True,
        after_epoch_callback=create_callback(monitor, log_dir)
    )

    # 评估阶段
    print("\nEvaluating...")
    y_score, eval_res = tasks.eval_classification(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        eval_protocol='linear'
    )

    # 处理不同维度的预测分数
    if y_score.ndim == 1:  # 二分类决策值
        y_pred = (y_score > 0).astype(int)
    else:  # 多分类概率
        y_pred = np.argmax(y_score, axis=1)

    # 记录结果
    monitor.log_confusion_matrix(test_labels, y_pred, np.unique(test_labels), 0)
    monitor.log_scalar('Metrics/Accuracy', eval_res['acc'], 0)
    monitor.log_scalar('Metrics/AUPRC', eval_res['auprc'], 0)

    # 保存嵌入
    sample_features = model.encode(test_data[:100], encoding_window='full_series')
    monitor.log_embedding(sample_features, test_labels[:100], 0)

    # 保存日志
    with open(os.path.join(log_dir, 'results.txt'), 'w') as f:
        f.write(f"Final Loss: {monitor.loss_history[-1]:.4f}\n")
        f.write(f"Accuracy: {eval_res['acc']:.4f}\n")
        f.write(f"AUPRC: {eval_res['auprc']:.4f}\n")

    monitor.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()