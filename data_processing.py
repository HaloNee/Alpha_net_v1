import pandas as pd
import numpy as np
import os


import warnings
warnings.filterwarnings('ignore')


def tick_time_list():
    """
    This function return a time series of 3-seconds interval
    """
    list1 = pd.date_range("9:30:00", "11:30:00", freq="3S").time
    list2 = pd.date_range('13:00:00', '14:57:00', freq='3S').time
    func = np.vectorize(lambda x: int(x.strftime("%H%M%S")))
    list1 = func(list1)
    list2 = func(list2)
    return np.hstack((list1, list2))


def bins():
    """
    This funtion return a time sequence to form bin edge
    """
    list1 = pd.date_range("9:29:57", "11:30:00", freq="3S").time
    list2 = pd.date_range('13:00:00', '15:00:00', freq='3S').time
    func = np.vectorize(lambda x: int(x.strftime("%H%M%S")))
    list1 = func(list1)
    list2 = func(list2)
    return np.hstack((list1, list2))


def labels():
    """
    This funtion return time label for tick data
    """
    list1 = pd.date_range("9:30:00", "11:30:00", freq="3S").time
    list2 = pd.date_range('13:00:00', '15:00:00', freq='3S').time
    func = np.vectorize(lambda x: int(x.strftime("%H%M%S")))
    list1 = func(list1)
    list2 = func(list2)
    return np.hstack((list1, list2))


def get_valid_code_list(
    tick_data_path,
    circulation_capital_stock_data_path,
    flag_ipo_st_data_path,
    date_str,
):
    """
    This function return valid stock code list for specific date
    """
    circulation_capital_stock = pd.read_csv(
        circulation_capital_stock_data_path, index_col="S_INFO_WINDCODE")
    circulation_capital_stock = circulation_capital_stock["20220301"]
    circulation_capital_stock.dropna(inplace=True)
    circulation_code_list = list(circulation_capital_stock.index)

    flag_ipo_st = pd.read_csv(flag_ipo_st_data_path,
                              index_col="S_INFO_WINDCODE")
    flag_ipo_st = flag_ipo_st[date_str]
    flag_ipo_st.dropna(inplace=True)
    non_ipo_st_list = list(flag_ipo_st.index)

    valid_code_list = list(set(circulation_code_list) & set(non_ipo_st_list))

    return valid_code_list


def get_circulation_capital_stock(circulation_capital_stock_data_path, date: str):
    """
    This function return capital stock in circulation for specific date
    """
    circulation_capital_stock = pd.read_csv(
        circulation_capital_stock_data_path, index_col="S_INFO_WINDCODE")
    circulation_capital_stock = circulation_capital_stock[date]
    circulation_capital_stock.dropna(inplace=True)
    return circulation_capital_stock


def cal_wavgBidVolume(volume: pd.DataFrame):
    """
    This function return the weight average bid volume
    """
    weights = np.array([(1 / 3) ** i for i in range(10)])
    weights = weights / sum(weights)
    result = volume['bidVolume1'] * weights[0] +\
        volume['bidVolume2'] * weights[1] +\
        volume['bidVolume3'] * weights[2] +\
        volume['bidVolume4'] * weights[3] +\
        volume['bidVolume5'] * weights[4] +\
        volume['bidVolume6'] * weights[5] +\
        volume['bidVolume7'] * weights[6] +\
        volume['bidVolume8'] * weights[7] +\
        volume['bidVolume9'] * weights[8] +\
        volume['bidVolume10'] * weights[9]

    return result


def cal_wavgAskVolume(volume: pd.DataFrame):
    """
    This function return the weight average ask volume
    """
    weights = np.array([(1 / 3) ** i for i in range(10)])
    weights = weights / sum(weights)
    result = volume['askVolume1'] * weights[0] +\
        volume['askVolume2'] * weights[1] +\
        volume['askVolume3'] * weights[2] +\
        volume['askVolume4'] * weights[3] +\
        volume['askVolume5'] * weights[4] +\
        volume['askVolume6'] * weights[5] +\
        volume['askVolume7'] * weights[6] +\
        volume['askVolume8'] * weights[7] +\
        volume['askVolume9'] * weights[8] +\
        volume['askVolume10'] * weights[9]

    return result


def split_tick_data(tick_data_path, valid_code_list, tick_save_path):
    """
    This function split the tick data by stock code
    """

    if not os.path.exists(tick_save_path):
        os.makedirs(tick_save_path)

    i = 0
    existedCodeList = []
    chunks = pd.read_csv(tick_data_path, iterator=True, dtype={"status": str})
    while True:
        try:
            tick = chunks.get_chunk(1000000)
        except:
            break
        
        # delete useless columns
        uselessColumnList = ["IOPV", "YTM", "syl1", "syl2", "SD2"]
        for uselessColumn in uselessColumnList:
            if uselessColumn in tick.columns:
                tick.drop([uselessColumn], axis='columns', inplace=True)

        # time_mask = ((tick['time'] >= 93000000) & (tick['time'] <= 113000000)) | (
        #             (tick['time'] >= 130000000) & (tick['time'] <= 150000000))
        # tick = tick[time_mask]
        # tick.loc[:, 'time'] = tick['time'] / 1000
        # tick.loc[:, 'newPrice'] = tick['newPrice'] / 10000
        # tick.loc[:, 'time'] = pd.cut(
        #     tick['time'], bins=tick_bins, labels=tick_labels)
        # tick.drop_duplicates(['time', 'code'], inplace=True)

        tick_groups = tick.groupby(tick.code)
        for code in tick.code.unique():
            if code in valid_code_list:
                tick_temp = tick_groups.get_group(code)
            else:
                continue

            # store the data for the first time
            if code not in existedCodeList:
                existedCodeList.append(code)
                # tick_temp.to_hdf(tick_save_path + f'tick_{code}.h5', key='df',
                #                  format="table", append=False)
                tick_temp.to_csv(tick_save_path + f'tick_{code}.csv', header=True, index=False, mode='w')
            else:
                # tick_temp.to_hdf(tick_save_path + f'tick_{code}.h5', key='df',
                #                  format="table", append=True)
                tick_temp.to_csv(tick_save_path + f'tick_{code}.csv', header=False, index=False, mode='a')
        i += 1
        print(f'The {i}th chunk completed!')


def get_trainset(tick_data_path, code_list, circulation_capital_stock, save_path):
    """
    This function return trainset
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    price_feature_list = ['askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
                          'askPrice6', 'askPrice7', 'askPrice8', 'askPrice9', 'askPrice10',
                          'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
                          'bidPrice6', 'bidPrice7', 'bidPrice8', 'bidPrice9', 'bidPrice10']

    volume_feature_list = ['askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5',
                           'askVolume6', 'askVolume7', 'askVolume8', 'askVolume9', 'askVolume10',
                           'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',
                           'bidVolume6', 'bidVolume7', 'bidVolume8', 'bidVolume9', 'bidVolume10']

    all_feature_list = ["code", "date", "time"] + \
        price_feature_list + volume_feature_list + \
        ['newPrice', 'wavgBidPrice', 'wavgAskPrice', 'volume', 'turover']

    count = 0
    tick_bins = bins()
    tick_labels = labels()
    tick_time = tick_time_list()
    fileList = os.listdir(tick_save_path)
    for fileName in fileList:
        code = fileName[5:14]
        # tick_temp = pd.read_hdf(tick_data_path + f'tick_{code}.h5', key='df')
        tick_temp = pd.read_csv(tick_data_path + fileName)
        tick_temp.loc[:, 'time'] = tick_temp['time'] / 1000
        tick_temp['volume'] = tick_temp['volume'] - tick_temp['volume'].shift(1)
        tick_temp['turover'] = tick_temp['turover'] - tick_temp['turover'].shift(1)

        tick_temp = tick_temp[tick_temp['status'] == '2']

        time_mask = ((tick_temp['time'] >= 93000) & (tick_temp['time'] <= 113000)) | (
            (tick_temp['time'] >= 130000) & (tick_temp['time'] < 150000))
        tick_temp = tick_temp[time_mask]
        
        tick_temp.loc[tick_temp['newPrice'] == 0, 'newPrice'] = tick_temp.loc[tick_temp['newPrice'] == 0, 'preClosePrice']
        tick_temp['newPrice'] = tick_temp['newPrice'] / 10000
        tick_temp['wavgBidPrice'] = tick_temp['wavgBidPrice'] / 10000
        tick_temp['wavgAskPrice'] = tick_temp['wavgAskPrice'] / 10000
        tick_temp.loc[:, 'time'] = pd.cut(tick_temp['time'], bins=tick_bins, labels=tick_labels)
        tick_temp.drop_duplicates(['time', 'code'], inplace=True)
        if tick_temp.time.count() < 3800:
            print(f"Missing data of {code} exceed 20%, eliminating")
            continue
        tick_temp = tick_temp[all_feature_list]
        tick_data = tick_temp.merge(pd.Series(tick_time, name='time_complete'), left_on='time', right_on='time_complete',
                                    how='right')
        tick_data["time"] = tick_data["time_complete"]

        tick_data['volume'].fillna(0, inplace=True)
        tick_data['turover'].fillna(0, inplace=True)

        tick_data = tick_data.ffill().bfill()

        # tick_data.loc[tick_data['newPrice'] == 0, 'newPrice'] = (
        #     tick_data.loc[tick_data['newPrice'] == 0, 'askPrice1'] + tick_data.loc[tick_data['newPrice'] == 0, 'bidPrice1']) / 2

        tick_data['volume'] = tick_data['volume'] / circulation_capital_stock[code] * 10 + 1
        tick_data['volume'] = tick_data['volume'].apply(np.log)

        tick_data["is_limit"] = ((tick_data["askVolume1"] == 0) | (
            tick_data['bidVolume1'] == 0)).astype(np.int32)
        # 对挂单量进行标准化处理，各档挂单量除以流通股本数量（单位：万股）*10，之后取对数
        tick_data.loc[:, volume_feature_list] = tick_data.loc[:,
                                                              volume_feature_list] / circulation_capital_stock[code] * 10 + 1
        tick_data.loc[:, volume_feature_list] = tick_data.loc[:,
                                                              volume_feature_list].apply(np.log)

        tick_data['wavgBidVolume'] = cal_wavgBidVolume(
            tick_data[volume_feature_list])

        tick_data['wavgAskVolume'] = cal_wavgAskVolume(
            tick_data[volume_feature_list])

        tick_data['imbalance'] = tick_data['wavgBidVolume'] - \
            tick_data['wavgAskVolume']
        # 计算未来一分钟收益率
        tick_data["return_1min"] = tick_data['newPrice'].pct_change(
            20).shift(-20)
        # tick_data.to_hdf(save_path + f'{code}.h5', key='df')
        tick_data.to_csv(save_path + f'{code}.csv', header=True, index=False, mode='w')
        count += 1
        print(f'{count}: {code}.csv file has been saved')


if __name__ == '__main__':

    date_list = [
        "20220301", "20220302", "20220303", "20220304", "20220307", "20220308", "20220309", "20220310",
        "20220311", "20220314", "20220315", "20220316", "20220317", "20220318",
        # "20220321", "20220322", "20220323",
        # "20220324", "20220325", "20220328", "20220329", "20220330", "20220331",
        # "20220401", "20220406", "20220407", "20220408",
        # "20220411", "20220412", "20220413", "20220414", "20220415", "20220418", "20220419", "20220420",
        "20220421", "20220422", "20220425", "20220426", "20220427", "20220428", "20220429",
        "20220505", "20220506", "20220509", "20220510",
        "20220511", "20220512", "20220513", "20220516", "20220517", "20220518", "20220519", "20220520",
        "20220523", "20220524", "20220525", "20220526", "20220527", "20220530", "20220531",
        "20220601", "20220602", "20220606", "20220607", "20220608", "20220609", "20220610",
        "20220613", "20220614", "20220615", "20220616", "20220617", "20220620",
        "20220621", "20220622", "20220623", "20220624", "20220627", "20220628", "20220629", "20220630",
    ]

    date_list = ["20220325"]
    date_list = ["20220427"]
    # 读取流通股本数据，ipo、st标识数据
    circulation_capital_stock_data_path = "E:/wuyuzc/AlphaNet/data_input/circulation_capitalStock.csv"
    flag_ipo_st_data_path = 'E:/wuyuzc/AlphaNet/data_input/flag_ipo_st.csv'

    # 此处暂按20220301日流通股本数据作为基准进行标准化处理
    circulation_capital_stock = get_circulation_capital_stock(
        circulation_capital_stock_data_path, "20220301")

    for date_str in date_list:
        tick_data_path = f'E:/wuyuzc/AlphaNet/data_input/{date_str}.csv'
        tick_save_path = f'E:/wuyuzc/AlphaNet/data_output_temp/{date_str}_temp/'
        # 过滤无效股票，去除无流通股本数据的股票、去除ipo、st股票
        valid_code_list = get_valid_code_list(
            tick_data_path, circulation_capital_stock_data_path, flag_ipo_st_data_path, date_str)

        # 按股票代码拆分每日tick数据
        split_tick_data(tick_data_path, valid_code_list, tick_save_path)

        # 计算训练数据集
        trainset_save_path = f'E:/wuyuzc/AlphaNet/processed data/{date_str}_outer/'
        get_trainset(tick_save_path, valid_code_list,
                     circulation_capital_stock, trainset_save_path)
