import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_processed.processed import setting_time_as_an_index, non_binary_columns

ddos_data = pd.read_csv(r'F:\lstm_model\csv\processed_train.csv')
n_past, n_future = 20, 2                                                #前者输入数据样本数，后者表示预测未来的标签
trainX=[]                                                               #存储样本特征
trainY=[]                                                               #存储输入目标标签
for i in range(n_past,len(ddos_data)-n_future+1):                       #构建滑动窗口
    trainX.append(ddos_data.iloc[i-n_past:i, :])
    trainY.append(ddos_data.iloc[i+n_future-1:i+n_future]['Label'])
trainX, trainY=np.array(trainX), np.array(trainY)                       #将列表转换为NumPy数组，便于深度学习模型处理
print(f'trainX shape=={trainX.shape}')
print(f'trainY shape=={trainY.shape}')
model=Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])),
    LSTM(units=64, return_sequences=True),
    LSTM(units=32, return_sequences=False),
    Dropout(0.3),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=1e-4),loss=BinaryCrossentropy(),metrics=['accuracy'])
model.summary()
history_log=model.fit(trainX, trainY, epochs=50, batch_size=1024, validation_split=0.2,verbose=2, shuffle=False)

# 创建一个大图界面（一行两列）
plt.figure(figsize=(12, 5))  # 调整整体画布大小
'''
# 第一张图：Loss
plt.subplot(1, 2, 1)  # 1行2列中的第1个位置
loss = history_log.history['loss']
val_loss = history_log.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 第二张图：Accuracy
plt.subplot(1, 2, 2)  # 1行2列中的第2个位置
accuracy = history_log.history['accuracy']
val_accuracy = history_log.history['val_accuracy']
plt.plot(epochs, accuracy, 'g', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 自动调整子图间距
plt.tight_layout()
plt.show()'''


# 读取数据
data_path = r'F:\lstm_model\csv\processed_true.csv'
ddos_data_true = pd.read_csv(data_path)

# 将 Timestamp 设置为索引
ddos_data_true['Timestamp'] = pd.to_datetime(ddos_data_true['Timestamp'])
ddos_data_true.set_index('Timestamp', inplace=True)
print(ddos_data_true.index)
print(ddos_data_true.columns)

testX=[]
testY=[]
n_future=2
n_past=20
for i in range(n_past, len(ddos_data_true) - n_future + 1):
    testX.append(ddos_data_true.iloc[i - n_past:i, :].values)  # 使用 .values 获取数值数据
    testY.append(ddos_data_true.iloc[i + n_future - 1:i + n_future]['Label'].values)  # 使用 .values 获取数值数据
testX = np.array(testX)
testY = np.array(testY)
testX = testX.astype(np.float32)
testY = testY.astype(np.float32)
print(f'trainX shape=={testX.shape}')
print(f'trainY shape=={testY.shape}')

y_pred = model.predict(testX)
binary_predictions = (y_pred > 0.5).astype(int)

print("样本预测标签:", binary_predictions[:10].flatten())  # 应输出 [0 1 0 1 1 0 ...]
print("样本真实标签:", testY[:10].flatten())

# 计算精确率、召回率、F1等原有指标
accuracy = accuracy_score(testY, binary_predictions)
precision = precision_score(testY, binary_predictions)
recall = recall_score(testY, binary_predictions)
f1 = f1_score(testY, binary_predictions)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

total_traffic = len(testY)# 总流量数（测试样本总数）
print(f'Total Traffic: {total_traffic}')
attack_traffic = np.sum(testY)# 攻击流量数（正例数）
print(f'Attack Traffic: {attack_traffic}')
false_positives = np.sum((testY == 0) & (binary_predictions == 1))# 错误预测的正例数（FP，误检流量数）
print(f'False Positives (误检流量数): {false_positives}')
false_negatives = np.sum((testY == 1) & (binary_predictions == 0))# 错误预测的负例数（FN，漏检流量数）
print(f'False Negatives (漏检流量数): {false_negatives}')


import asyncio
import websockets
import json

timestamps = ddos_data_true.index[n_past + n_future - 1:]  # 获取对应时间戳
probabilities = y_pred.flatten().tolist()


async def send_stats(websocket, path):
    try:
        while True:
            if not websocket.open:
                break

            # 获取全部数据（不限制数量）
            stats = {
                "total_traffic": len(testY),
                "attack_traffic": int(np.sum(testY)),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                # 传输三种数据：预测概率、真实标签、预测标签
                "prob_data": [
                    {
                        "timestamp": str(timestamps[i]),
                        "probability": float(probabilities[i]),
                        "true_label": int(testY[i][0].item()),  # 使用.item()提取标量值
                        "pred_label": int(binary_predictions[i][0].item())
                    }
                    for i in range(len(timestamps))
                ]
            }
            await websocket.send(json.dumps(stats))
            await asyncio.sleep(3) # 降低发送频率
    except websockets.exceptions.ConnectionClosed:
        print("客户端正常断开")
    except Exception as e:
        print(f"推送异常: {str(e)}")

start_server = websockets.serve(send_stats, "172.27.52.106", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()


