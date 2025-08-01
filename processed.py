'利用的分类变量中的名义变量进行模型训练'
import pickle

import joblib
import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

#读取CSV数据
ddos_data=pd.read_csv('F:\\lstm_model\\csv\\train_data.csv')
train_columns=ddos_data.columns.tolist()     #读取列名
print(train_columns)
#print(ddos_data.info())#读取数据集的结构
print(ddos_data.describe())
ddos_data['Label'] = ddos_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
print(ddos_data['Label'].value_counts())

#print(ddos_data.isnull().sum())#查找缺失值
def normalize_timestamp(ddos_data):
    ddos_data['Timestamp'] = pd.to_datetime(ddos_data['Timestamp'])
    ddos_data.sort_values(by='Timestamp', inplace=True)
normalize_timestamp(ddos_data)
print(ddos_data['Timestamp'])

#查找数据集中值为唯一值的列
constant_columns=ddos_data.columns[ddos_data.nunique()==1]
print(constant_columns)
len(constant_columns)
are_equal = (ddos_data['Fwd Header Length'] == ddos_data['Fwd Header Length.1'])
all_equal = are_equal.all()
if all_equal:
    print("The two columns are the same.")
else:
    print("The two columns are not the same.")

# 删除空值列、无关列
columns_to_drop = [
        'Unnamed: 0', 'Unnamed: 0.1', 'Flow ID', 'Fwd Header Length.1', 'SimillarHTTP', 'Init_Win_bytes_forward',
        'Init_Win_bytes_backward',
       'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count',
       'PSH Flag Count', 'ECE Flag Count', 'Fwd Avg Bytes/Bulk',
       'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
       'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
ddos_data.drop(columns=columns_to_drop, axis=1, inplace=True)
print(ddos_data.columns)


le_label=LabelEncoder()
def encode_label_col(ddos_data):
    ddos_data['Label']=le_label.fit_transform(ddos_data['Label'])

le_ip = LabelEncoder()
def encoding_ip_srv_dst(ddos_data):
    all_ip=pd.concat([ddos_data['Source IP'],ddos_data['Destination IP']])
    le_ip.fit(all_ip)
    ddos_data['Source IP']=le_ip.transform(ddos_data['Source IP'])
    ddos_data['Destination IP']=le_ip.transform(ddos_data['Destination IP'])
encode_label_col(ddos_data)
encoding_ip_srv_dst(ddos_data)
print(ddos_data)


ddos_data_cp=ddos_data.drop('Timestamp',axis=1)

#进行相关性计算
ddos_data_cp.corr()
correlation_matrix = ddos_data_cp.corr()
highly_correlated = correlation_matrix.abs() > 0.9

correlated_features = [(feature1, feature2) for feature1 in correlation_matrix.columns
                       for feature2 in correlation_matrix.columns
                       if highly_correlated.loc[feature1, feature2] and feature1 != feature2]

printed_pairs = set()
for feature1, feature2 in correlated_features:
    if (feature2, feature1) not in printed_pairs:  # Check if the pair has already been printed
        print(f"{feature1} and {feature2} 相关性很高: {correlation_matrix.loc[feature1, feature2]}")
        printed_pairs.add((feature1, feature2))

corr_set1=['Flow Duration','Fwd IAT Total']
correlation_set1 = ddos_data[corr_set1 + ['Label']].corr()
label_correlation_set1 = correlation_set1['Label'].drop('Label')
print(label_correlation_set1)
ddos_data.drop(columns=['Fwd IAT Total'],inplace=True)

corr_set2=['Total Backward Packets','Total Length of Bwd Packets','Bwd Header Length','Subflow Bwd Packets','Subflow Bwd Bytes']
correlation_set2 = ddos_data[corr_set2 + ['Label']].corr()
label_correlation_set2 = correlation_set2['Label'].drop('Label')
print(label_correlation_set2)
ddos_data.drop(columns=['Total Backward Packets','Total Length of Bwd Packets','Subflow Bwd Packets','Subflow Bwd Bytes'],inplace=True)

corr_set3=['Total Length of Fwd Packets','act_data_pkt_fwd','Subflow Fwd Bytes']
correlation_set3 = ddos_data[corr_set3 + ['Label']].corr()
label_correlation_set3 = correlation_set3['Label'].drop('Label')
print(label_correlation_set3)
ddos_data.drop(columns=['act_data_pkt_fwd','Subflow Fwd Bytes'],inplace=True)

corr_set4=['Fwd Packet Length Min','Fwd Packet Length Mean','Min Packet Length','Average Packet Size','Avg Fwd Segment Size']
correlation_set4 = ddos_data[corr_set4 + ['Label']].corr()
label_correlation_set4 = correlation_set4['Label'].drop('Label')
print(label_correlation_set4)
ddos_data.drop(columns=['Fwd Packet Length Min','Fwd Packet Length Mean','Average Packet Size','Avg Fwd Segment Size'],inplace=True)

corr_set5=['Bwd Packet Length Max','Bwd Packet Length Std','Max Packet Length','Packet Length Std']
correlation_set5 = ddos_data[corr_set5 + ['Label']].corr()
label_correlation_set5 = correlation_set5['Label'].drop('Label')
print(label_correlation_set5)
ddos_data.drop(columns=['Bwd Packet Length Max','Bwd Packet Length Std','Max Packet Length'],inplace=True)

corr_set6=['Bwd Packet Length Mean','Packet Length Std','Avg Bwd Segment Size']
correlation_set6 = ddos_data[corr_set6 + ['Label']].corr()
label_correlation_set6 = correlation_set6['Label'].drop('Label')
print(label_correlation_set6)
ddos_data.drop(columns=['Packet Length Std','Avg Bwd Segment Size'])

corr_set7=['Flow Packets/s','Fwd Packets/s']
correlation_set7 = ddos_data[corr_set7 + ['Label']].corr()
label_correlation_set7 = correlation_set7['Label'].drop('Label')
print(label_correlation_set7)
#ddos_data.drop(columns=['Fwd Packets/s'])

corr_set8=['Flow IAT Mean','Fwd IAT Mean']
correlation_set8 = ddos_data[corr_set8 + ['Label']].corr()
label_correlation_set8 = correlation_set8['Label'].drop('Label')
print(label_correlation_set8)
ddos_data.drop(columns=['Fwd IAT Mean'])

corr_set9=['Flow IAT Std','Fwd IAT Std']
correlation_set9 = ddos_data[corr_set9 + ['Label']].corr()
label_correlation_set9 = correlation_set9['Label'].drop('Label')
print(label_correlation_set9)
ddos_data.drop(columns=['Flow IAT Std'])

corr_set10=['Flow IAT Max','Fwd IAT Std','Fwd IAT Max','Idle Mean','Idle Max','Idle Min']
correlation_set10 = ddos_data[corr_set10 + ['Label']].corr()
label_correlation_set10 = correlation_set10['Label'].drop('Label')
print(label_correlation_set10)
ddos_data.drop(columns=['Flow IAT Max','Fwd IAT Std','Fwd IAT Max','Idle Mean','Idle Max'])

corr_set11=['Flow IAT Min','Fwd IAT Min']
correlation_set11 = ddos_data[corr_set11 + ['Label']].corr()
label_correlation_set11 = correlation_set11['Label'].drop('Label')
print(label_correlation_set11)
ddos_data.drop(columns=['Flow IAT Min'])

corr_set12=['Bwd IAT Mean','Bwd IAT Std']
correlation_set12 = ddos_data[corr_set12 + ['Label']].corr()
label_correlation_set12 = correlation_set12['Label'].drop('Label')
print(label_correlation_set12)
ddos_data.drop(columns=['Bwd IAT Mean'])

corr_set13=['Bwd IAT Std','Bwd IAT Max']
correlation_set13 = ddos_data[corr_set13 + ['Label']].corr()
label_correlation_set13 = correlation_set13['Label'].drop('Label')
print(label_correlation_set13)
ddos_data.drop(columns=['Bwd IAT Std'])

corr_set14=['Fwd Header Length','min_seg_size_forward']
correlation_set14 = ddos_data[corr_set14 + ['Label']].corr()
label_correlation_set14 = correlation_set14['Label'].drop('Label')
print(label_correlation_set14)
ddos_data.drop(columns=['Fwd Header Length'])

corr_set15=['Active Mean','Active Min']
correlation_set15 = ddos_data[corr_set15 + ['Label']].corr()
label_correlation_set15 = correlation_set15['Label'].drop('Label')
print(label_correlation_set15)
ddos_data.drop(columns=['Active Min'])

ddos_data.drop(columns=['Subflow Fwd Packets','Min Packet Length','RST Flag Count'])
ddos_data.corr()

correlation_matrix = ddos_data_cp.corr()
label_correlation = correlation_matrix['Label']
print(label_correlation)

def setting_time_as_an_index(ddos_data):
    ddos_data.set_index('Timestamp', inplace=True)
setting_time_as_an_index(ddos_data)
ddos_data.info()

non_binary_columns=ddos_data.columns[ddos_data.nunique()>2]

scaler=StandardScaler()
def scale_num_features(ddos_data):
    ddos_data[non_binary_columns]=scaler.fit_transform(ddos_data[non_binary_columns])
is_infinite = np.any(np.isinf(ddos_data[non_binary_columns]), axis=0)
is_large = np.any(np.abs(ddos_data[non_binary_columns]) > np.finfo(np.float64).max, axis=0)
columns_with_issues = non_binary_columns[is_infinite]
print("Columns with Infinite or Large Values:", columns_with_issues)

ddos_data.drop(columns=['Flow Bytes/s','Flow Packets/s'], inplace=True)
non_binary_columns=ddos_data.columns[ddos_data.nunique()>2]
scale_num_features(ddos_data)
final_columns=ddos_data.columns
print(ddos_data.corr())
ddos_data.to_csv('F:\\lstm_model\\csv\\processed_train.csv', index=False)


data_path = r'F:\lstm_model\csv\true_data.csv'
true_df = pd.read_csv(data_path)
print(true_df.shape)
normalize_timestamp(true_df)
def dropping_const_and_red_columns(true_df):
    true_df.drop(['Unnamed: 0.1', 'Unnamed: 0','Flow ID','Fwd Header Length.1'],axis=1,inplace=True)
    true_df.drop(columns=constant_columns,inplace=True)
dropping_const_and_red_columns(true_df)

true_df['Label'] = true_df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

encode_label_col(true_df)
encoding_ip_srv_dst(true_df)
setting_time_as_an_index(true_df)

true_df=true_df[final_columns]
nonbinary_columns=true_df.columns[true_df.nunique()>2]
scale_num_features(true_df)
true_df.to_csv(r'F:\lstm_model\csv\processed_true.csv')

