'合并多个csv文件'
import pandas as pd
import os
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

# 设置数据集路径
data_dir = 'F:/Graduation project/ddos-data-set/cic201901/datasets'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f"Number of CSV files loaded: {len(csv_files)}")

# 加载数据
df_list = []
for file in csv_files:
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
combined_csv = pd.concat(df_list, ignore_index=True)
print(f"Shape of combined_df: {combined_csv.shape}")
#print(combined_df[' Label'].value_counts())
specific_value = 'WebDDoS'  #删除这个攻击类型
combined_csv = combined_csv[combined_csv[' Label'] != specific_value]
combined_csv.columns = combined_csv.columns.str.strip()#清楚列名中的空格
combined_csv.to_csv('F:\\lstm_model\\csv\\combined_csv.csv', index=False)

# 定义攻击类型
ddos_attack = ['Syn', 'TFTP', 'DrDoS_LDAP', 'DrDoS_UDP', 'UDP-lag', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_NetBIOS',
               'DrDoS_MSSQL', 'DrDoS_DNS', 'DrDoS_NTP']
ddos_dict = {
    atk: combined_csv[combined_csv['Label'] == atk].sort_values(by='Timestamp')
    for atk in ddos_attack
}
normal_df = combined_csv[combined_csv['Label'] == 'BENIGN'].sort_values(by='Timestamp')
combined_data = pd.concat(
    [normal_df] + list(ddos_dict.values()),  # 将字典的值转为列表并与 normal_df 合并
    ignore_index=True
)
print(f"Shape of final_df: {combined_data.shape}")
combined_data.to_csv(r'F:\lstm_model\csv\combined_data.csv', index=False)
# 抽取训练集和真实流量集
train_list = []
true_list = []

# 先处理 BENIGN：
benign_train = normal_df .iloc[:5000]           # 前5000条正常流量
benign_true  = normal_df .iloc[5000:6500]       # 接下来1500条正常流量
train_list.append(benign_train)
true_list.append(benign_true)

# 处理每种攻击类型
for atk, atk_df in ddos_dict.items():
    if len(atk_df) < 400:
        raise ValueError(f"攻击类型 {atk} 样本数不足，只有 {len(atk_df)} 条")
    # 前300条用于训练
    train_list.append(atk_df.iloc[:300])
    # 接下来100条用于真实流量模拟
    true_list.append(atk_df.iloc[300:400])

# 合并并输出
train_data = pd.concat(train_list, ignore_index=True)
true_data  = pd.concat(true_list,  ignore_index=True)

print(f"Shape of train_data: {train_data.shape}  # 应为 (5000+3300)=8300")
print(f"Shape of true_data:  {true_data.shape}  # 应为 (1500+1100)=2600")

# 保存到 CSV
train_data.to_csv(r'F:/lstm_model/csv/train_data.csv', index=False)
true_data.to_csv (r'F:/lstm_model/csv/true_data.csv',  index=False)
true_list = {
    0: "HOPOPT",
    6: "TCP",
    17: "UDP"
}
true_data['Protocol'] = true_data['Protocol'].map(true_list)

true_data = true_data[['Source IP', 'Source Port','Destination IP','Destination Port','Protocol','Label']]  # 筛选指定列
true_data.to_json(
    r'F:\ddos.system\ddosSystem03\entry\src\main\resources\rawfile\true_data.json',
    orient='records',   # 按记录格式保存
    lines=False,         # 确保为标准的 JSON 数组
    index=False          # 不保存索引
)
# 查看各类别计数
print("训练集各类样本数：")
print(train_data['Label'].value_counts())
print("真实流量集各类样本数：")
print(true_data['Label'].value_counts())
