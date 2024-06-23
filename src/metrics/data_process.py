import re
import numpy as np
import re
import pandas as pd
import os
import math


######################## Generate test_data_by_attack
def form_file(load_name, Attack_ID, Defense_ID, trainable_list, current_Q, dataset):
    '''
    load_name: ./exp_result  user-defined result file
    '''
    for attack_id in Attack_ID:
        print('=== For:', attack_id)
        for trainable in trainable_list:  # ,'1'

            data_dir = load_name + '/Q' + str(current_Q) + '/' + str(trainable) + '/'

            to_name = attack_id  # 'LRB'
            to_dir = 'test_data_by_attack/' + to_name + '/Q' + str(current_Q) + '/' + str(
                trainable) + '/'  # '/Q'+str(current_Q)+
            if not os.path.exists(to_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(to_dir)
            print('=======', to_dir)

            try:
                filenames = os.listdir(data_dir)  # target filenames
            except:
                print('No:', data_dir)
                continue

            # For Each Defense
            for file_name in filenames:

                defense_flag = 0
                for defense_id in Defense_ID:
                    if defense_id in file_name:
                        defense_flag = 1
                        print(defense_id, file_name)
                if defense_flag == 0:
                    continue

                content = []
                ######### select  ##########
                ignore_pattern = re.compile(r'ignore')

                matchPattern1 = re.compile(r'No_Attack')
                matchPattern2 = re.compile(r'ReplacementBackdoor')
                matchPattern3 = re.compile(r'NoisySample')
                matchPattern4 = re.compile(r'MissingFeature')

                matchPattern5 = re.compile(r'PassiveModelCompletion')  # PMC AMC
                matchPattern55 = re.compile(r'ActiveModelCompletion')  # PMC AMC
                matchPattern6 = re.compile(r'DirectLabelScoring')
                matchPattern7 = re.compile(r'BatchLabelReconstruction')

                matchPattern8 = re.compile(r'NormbasedScoring')
                matchPattern9 = re.compile(r'DirectionbasedScoring')

                matchPattern10 = re.compile(r'ResSFL')
                matchPattern11 = re.compile(r'GenerativeRegressionNetwork')
                ######### select  ##########
                attack_pattern_dict = {
                    'BLI': matchPattern7,
                    'DLI': matchPattern6,
                    'DS': matchPattern9,
                    'LRB': matchPattern2,
                    'PMC': matchPattern5,
                    'AMC': matchPattern55,
                    'MF': matchPattern4,
                    'No_Attack': matchPattern1,
                    'No_Attack_nsds': matchPattern1,
                    'NS': matchPattern8,
                    'NSB': matchPattern3,
                    'TBM': matchPattern10,
                    'GRN': matchPattern11
                }
                file = open(data_dir + file_name, 'r')
                while 1:
                    line = file.readline()
                    if not line:
                        break

                    if (ignore_pattern.search(line)):
                        continue

                    if (attack_pattern_dict[attack_id].search(line)):
                        content.append(line)
                        next_line = file.readline()
                        content.append(next_line)
                file.close()

                if content == []:
                    print(attack_id, file_name)

                file = open(to_dir + file_name, 'w')
                for i in content:
                    file.write(i)
                file.close()


trainable_list = ['0']
Attack_ID = ['No_Attack']
Defense_ID = ['None_None']
final_columns = ['K', 'bs', 'LR', 'num_class', 'Q', 'top_trainable', 'epochs',
                 'Commu', 'Main Task Accuracy', 'STD', 'stopping_commu(MB)', 'stopping_time', 'stopping_iter']
load_name = '../exp_result'
for current_Q in [1]:
    form_file(load_name, Attack_ID, Defense_ID, trainable_list, current_Q, load_name)


######################## Record raw mp/ap file

def get_avg(summary, total_df, trainable, Q, Defense):
    '''
    2.0, 1024.0, 0.01, 10.0, 1.0, 0.0, 30.0, 'BatchLabelReconstruction', '0.05',0.8874, 0.129883 ,CAE_0.0
    0       1      2     3    4    5     6           7                     8     9mp      10ap     11 
    '''
    if len(total_df) == 0:
        print(Defense, ': none file')
        return
    basic_info = list(total_df.iloc[0])[:7]  # 0-6

    assert Defense == list(total_df.iloc[0])[11]  # 11
    attack_name = list(total_df.iloc[0])[7]
    attack_param = list(total_df.iloc[0])[8]

    mp = total_df['main_acc'].mean()
    if attack_name in ['MissingFeature', 'NoisySample']:
        if max(list(total_df['attack_metric'])) <= 0:
            ap = 0
        else:
            ap_list = []
            for _ap in list(total_df['attack_metric']):
                if _ap >= 0:
                    ap_list.append(_ap)
            ap = sum(ap_list) / len(ap_list)
    else:
        ap = total_df['attack_metric'].mean()

    _info = basic_info + [Defense] + [attack_name, attack_param] + [mp, ap]

    summary.loc[len(summary.index)] = _info


current_Q = 1
dataset_name = 'mnist'
source_name = 'test_data_by_attack'
Attack_ID = ['BLI', 'DLI', 'DS', 'NS', 'No_Attack', 'No_Attack_nsds']
ALL_Attack = ['BatchLabelReconstruction', 'DirectLabelScoring', 'DirectionbasedScoring', 'NormbasedScoring', \
              'No_Attack', 'No_Attack_nsds']
final_columns = ['K', 'bs', 'LR', 'num_class', 'Q', 'top_trainable', 'epochs', 'Defense_name', \
                 'attack_name', 'attack_param', \
                 'Main Task Accuracy', 'Attack Performance']

for trainable in ['0']:
    summary = pd.DataFrame(columns=final_columns)
    for attack_id in Attack_ID:
        attack_data_dir = source_name + '/' + attack_id + '/'
        print(attack_data_dir)
        ########## READ FROM ################
        data_dir = attack_data_dir + 'Q' + str(current_Q) + '/' + trainable + '/'
        filenames = os.listdir(data_dir)  # attacks

        ignore_pattern = re.compile(r'ignore')
        # For Each Defense
        for file_name in filenames:
            defense_name = file_name.split(',')[0]
            # print('Defense:',defense_name)

            columns = ['K', 'bs', 'LR', 'num_class', 'Q', 'top_trainable', 'epochs', 'attack_name', 'attack_param', \
                       'main_acc', 'attack_metric']
            df = pd.DataFrame(columns=columns)

            f = open(data_dir + file_name, encoding='gbk')
            for line in f:
                if line == '' or line == '\n' or line[0] == '=':
                    continue

                if (ignore_pattern.search(line)):
                    continue

                info = line.split(',')

                if len(info) == 1:
                    print('Defense:', defense_name, attack_id)
                    print(line)
                    assert 1 > 2

                info = info[1].strip()
                final = info.split('|')
                for index in range(7):
                    final[index] = float(final[index])
                final[9] = float(final[9])

                # Some Specific Process for each attack
                if (final[7] == 'No_Attack'):

                    if int(final[3]) == 2:
                        final[7] = 'No_Attack_nsds'

                if len(final) == 12:
                    if (final[7] == 'GenerativeRegressionNetwork') or (final[7] == 'ResSFL'):
                        randmse = float(final[10])
                        mse = float(final[11])
                        final[10] = 1 - mse  # (randmse-mse)/randmse
                        final = final[:-1]
                    elif (final[7] == 'DirectionbasedScoring') or (final[7] == 'NormbasedScoring'):
                        acc = float(final[10])
                        auc = float(final[11])
                        if (final[7] == 'NormbasedScoring'):
                            acc = max(acc, 1 - acc)
                        final[10] = acc
                        final = final[:-1]
                    else:
                        print('error:')
                        print(final)
                else:
                    final[10] = float(final[10])

                if final[10] < 0:  # AP should be >0
                    final[10] = 0

                df.loc[len(df.index)] = final
            # add defense name
            df.insert(df.shape[1], 'DefenseName', defense_name)

            # get avg info for each attack
            get_avg(summary, df, int(trainable), current_Q, defense_name)  # add averaged info into summary

    # Save
    raw_file_name = 'record/' + dataset_name + '_Q' + str(current_Q) + '_mode' + str(trainable) + '_raw_record.xlsx'
    summary.to_excel(raw_file_name)
    print(raw_file_name)

################## Calculate DCS
TARGETED_BACKDOOR = ['ReplacementBackdoor']
UNTARGETED_BACKDOOR = ['NoisySample', 'MissingFeature']  # df2
LABEL_INFERENCE = ['BatchLabelReconstruction', 'DirectLabelScoring', 'NormbasedScoring', 'DirectionbasedScoring', \
                   'PassiveModelCompletion', 'ActiveModelCompletion']  # label_recovery
FEATURE_RECONSTRUCTION = ['GenerativeRegressionNetwork', 'ResSFL']  #
DEFENSE = ['MID_Active_1e-08', 'MID_Active_0.0', 'MID_Active_0.0001', \
           'MID_Active_0.1', 'MID_Active_0.01', 'MID_Active_1e-06', 'MID_Active_1.0', 'MID_Active_100',
           'MID_Active_10000', \
 \
           'GradPerturb_0.01', 'GradPerturb_0.1', 'GradPerturb_1.0', 'GradPerturb_10.0', \
 \
           'DistanceCorrelation_0.0001', 'DistanceCorrelation_0.01', 'DistanceCorrelation_0.1',
           'DistanceCorrelation_0.3', \
 \
           'DCAE_1.0', 'DCAE_0.5', 'DCAE_0.1', 'DCAE_0.0', \
           'CAE_0.0', 'CAE_0.5', 'CAE_0.1', 'CAE_1.0', \
 \
           'GradientSparsification_95.0', 'GradientSparsification_97.0', \
           'GradientSparsification_99.0', 'GradientSparsification_99.5', \
 \
           'LaplaceDP_0.0001', 'LaplaceDP_0.001', 'LaplaceDP_0.01', 'LaplaceDP_0.1', \
           'GaussianDP_0.0001', 'GaussianDP_0.01', 'GaussianDP_0.1', 'GaussianDP_0.001', \
           'None_None']


def cal_dcs(mp, ap, mp0, ap0, beta=0.5):
    d_ap = max(ap - ap0, 0)  # ap0 should < ap
    d_mp = max(mp0 - mp, 0)  # mp0 should > mp
    distance = math.sqrt((1 - beta) * (d_ap ** 2) + beta * (d_mp ** 2))
    dsc = 1 / (1 + distance)
    return dsc


def get_dcs(total_df, trainable, Q, defense_name, attack_name):
    '''
    total_df: summary
    'K', 'bs', 'LR', 'num_class', 'Q', 'top_trainable', 'epochs',
       'Defense_name', 'attack_name', 'attack_param', 'Main Task Accuracy',
       'Attack Performance'
    '''
    _df = total_df.loc[(total_df['attack_name'] == attack_name) & (total_df['Defense_name'] == defense_name) & (
                total_df['top_trainable'] == trainable) & (total_df['Q'] == Q)]

    if len(_df) == 0:
        print(defense_name, attack_name, 'none file')
        return
    mp = list(_df['Main Task Accuracy'])[0]
    ap = list(_df['Attack Performance'])[0]
    if len(list(_df['Main Task Accuracy'])) != 1:
        print(defense_name, ' ', attack_name, ' ', list(_df['Main Task Accuracy']))
    basic_info = list(_df.iloc[0][1:8])

    ###### Ideal Performance #######
    if attack_name in ['DirectionbasedScoring', 'NormbasedScoring', 'No_Attack_nsds']:
        vanilla = total_df.loc[
            (total_df['attack_name'] == 'No_Attack_nsds') & (total_df['Defense_name'] == 'None_None') & (
                        total_df['top_trainable'] == trainable) & (total_df['Q'] == Q)]
    else:
        vanilla = total_df.loc[(total_df['attack_name'] == 'No_Attack') & (total_df['Defense_name'] == 'None_None') & (
                    total_df['top_trainable'] == trainable) & (total_df['Q'] == Q)]
    if len(list(vanilla['Main Task Accuracy'])) == 0:
        print(attack_name, 'missing corresponding vannila case')
        assert 1 > 2
    mp0 = list(vanilla['Main Task Accuracy'])[0]
    ap0 = 0
    ###### Ideal Performance #######

    dcs = cal_dcs(mp, ap, mp0, ap0, beta=0.5)

    _info = basic_info + [mp, ap] + [defense_name] + [attack_name] + [dcs]
    report.loc[len(report.index)] = _info


for trainable in [0]:
    ################### Change Here ###################
    raw_file_name = 'record/' + dataset_name + '_Q' + str(current_Q) + '_mode' + str(trainable) + '_raw_record.xlsx'
    summary = pd.read_excel(raw_file_name)  # 'exp_result/'+load_name+'_raw_record.xlsx')
    print('Load Data from:', raw_file_name)
    ################### Change Here ###################

    final_columns = ['K', 'bs', 'LR', 'num_class', 'Q', 'top_trainable', 'epochs', 'mp', 'ap', 'Defense_name',
                     'Attack_name', 'DCS']
    report = pd.DataFrame(columns=final_columns)
    for defense_name in DEFENSE:
        for attack_name in ALL_Attack:
            get_dcs(summary, trainable, current_Q, defense_name, attack_name)
    dcs_report_name = 'record/' + dataset_name + '_Q' + str(current_Q) + '_mode' + str(trainable) + '_dcs.xlsx'
    report.to_excel(dcs_report_name, index=False)
    print(dcs_report_name)


###################### Calculate T-DCS & C-DCS
def average(value):
    if len(value) == 0:
        return -1
    # set T-DCS to -1, indicating we're not testing this type of attack

    clean_value = []
    for i in range(len(value)):
        if value[i] != -1:
            clean_value.append(value[i])
    result = sum(clean_value) / len(clean_value)

    return result


final_columns = ['K', 'bs', 'LR', 'num_class', 'Q', 'top_trainable', 'epochs', 'Defense_name', 'Defense_Param', \
                 'T-DCS_LI', 'T-DCS_FR', 'T-DCS_TB', 'T-DCS_NTB', 'C-DCS']  #
final_report = pd.DataFrame(columns=final_columns)


def get_final(total_df, trainable, Q, defense_name):
    dca_list = []
    for attack_name in TARGETED_BACKDOOR:
        _df = total_df.loc[(total_df['Attack_name'] == attack_name) & (total_df['Defense_name'] == defense_name) & (
                    total_df['top_trainable'] == trainable) & (total_df['Q'] == Q)]
        if len(_df) == 0:
            print(defense_name, attack_name, 'none file')
            continue
        dca_list.append(list(_df['DCS'])[0])

        basic_info = list(_df.iloc[0][1:8])
    dcs_TB = average(dca_list)

    dca_list = []
    for attack_name in UNTARGETED_BACKDOOR:
        _df = total_df.loc[(total_df['Attack_name'] == attack_name) & (total_df['Defense_name'] == defense_name) & (
                    total_df['top_trainable'] == trainable) & (total_df['Q'] == Q)]
        if len(_df) == 0:
            print(defense_name, attack_name, 'none file')
            continue
        dca_list.append(list(_df['DCS'])[0])

        basic_info = list(_df.iloc[0][1:8])
    dcs_NTB = average(dca_list)

    dca_list = []
    for attack_name in LABEL_INFERENCE:
        _df = total_df.loc[(total_df['Attack_name'] == attack_name) & (total_df['Defense_name'] == defense_name) & (
                    total_df['top_trainable'] == trainable) & (total_df['Q'] == Q)]
        if len(_df) == 0:
            print(defense_name, attack_name, 'none file')
            continue
        dca_list.append(list(_df['DCS'])[0])

        basic_info = list(_df.iloc[0][1:8])
    dcs_LI = average(dca_list)

    dca_list = []
    for attack_name in FEATURE_RECONSTRUCTION:
        _df = total_df.loc[(total_df['Attack_name'] == attack_name) & (total_df['Defense_name'] == defense_name) & (
                    total_df['top_trainable'] == trainable) & (total_df['Q'] == Q)]
        if len(_df) == 0:
            print(defense_name, attack_name, 'none file')
            continue
        dca_list.append(list(_df['DCS'])[0])

        basic_info = list(_df.iloc[0][1:8])
    dcs_FR = average(dca_list)

    d_list = defense_name.split('_')
    d_name = d_list[0]
    d_para = d_list[-1]
    cdcs = average([dcs_TB, dcs_NTB, dcs_LI, dcs_FR])
    _info = basic_info + [d_name, d_para] + [dcs_LI, dcs_FR, dcs_TB, dcs_NTB, cdcs]

    final_report.loc[len(final_report.index)] = _info


dcs_report_name0 = 'record/' + dataset_name + '_Q' + str(current_Q) + '_mode0' + '_dcs.xlsx'
print('Load Data From:', dcs_report_name0)
dcs_report0 = pd.read_excel(dcs_report_name0)
dcs_report_name1 = 'record/' + dataset_name + '_Q' + str(current_Q) + '_mode1' + '_dcs.xlsx'
print('Load Data From:', dcs_report_name1)
dcs_report1 = pd.read_excel(dcs_report_name1)

for defense_name in DEFENSE:
    get_final(dcs_report0, 0, current_Q, defense_name)
cdcs_report_name = 'record/' + dataset_name + '_Q' + str(current_Q) + '_mode0' + '_final_dcs.xlsx'
final_report.to_excel(cdcs_report_name)
print(cdcs_report_name)

for defense_name in DEFENSE:
    get_final(dcs_report1, 1, current_Q, defense_name)
cdcs_report_name = 'record/' + dataset_name + '_Q' + str(current_Q) + '_mode1' + '_final_dcs.xlsx'
final_report.to_excel(cdcs_report_name)
print(cdcs_report_name)
