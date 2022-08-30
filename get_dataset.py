from pathlib import Path

import numpy as np
import pandas as pd


def prepare_dataset(sample_path, date_list, valid_code_list, datatype, save_path):
    """Get dataset for model.
    
    Parameters
    ----------
    sample_path : Sample path.
    datelist :  Date list.
    
    Returns
    -------
    X_set : Training data.
    y_set : Target values.

    """  
    feature_list = ['newPrice', 'wavgBidPrice', 'wavgAskPrice', 'volume', 'wavgBidVolume',
                    'wavgAskVolume', 'imbalance', '3minReturn',
    ]
    filePathList = []
    for date in date_list:
        fold_path =  Path(sample_path) / f'{date}_outer'
        filesList = list(fold_path.glob('*.csv'))
        for filename in filesList:
            if filename.stem in valid_code_list:
                filePathList.append(fold_path/f'{filename.name}')  

    np.random.shuffle(filePathList)
    
    X_set = []
    y_set = []
    count, part_num = 0, 0
    for info in filePathList:
        count += 1
        code = info.name
        print(count, code)
        data = pd.read_csv(info)
        data['3minReturn'] = data['newPrice'].pct_change(60) * 100
        is_limit = data.loc[:, 'is_limit'].values

        training_set = data.loc[:, feature_list].values
        y_value = data['return_1min'].values * 100

        for i in range(0, 4500, 60):
            if (
                # filtering NAN
                ~np.isnan(training_set[i:i + 60, :]).any()
                and ~np.isnan(y_value[i + 60 - 1]).any()
                #  filtering inf value
                and ~np.isinf(training_set[i:i + 60, :]).any()
                and ~np.isinf(y_value[i + 60 - 1]).any()
                # filtering limit sample 
                and (is_limit[i:i + 60] == 0).all()
                # filtering zero return samples
                # and y_value[i + 60 - 1] != 0
            ):
                temp = training_set[i: i + 60, :].copy()
                first_price = training_set[0, 0].copy()
                if first_price == 0:
                    raise ZeroDivisionError("division by zero")
                else:
                    temp[:, :3] = temp[:, :3] / first_price
                X_set.append(temp.T)
                y_set.append(y_value[i + 60 - 1])
        
        if (count % 2000 == 0) | (info == filePathList[-1]):
            part_num += 1    
            X_set = np.array(X_set)
            y_set = np.array(y_set)
            upp_bound = y_set.mean() + 5 * y_set.std()
            low_bound = y_set.mean() - 5 * y_set.std()
            y_set = np.where(y_set > upp_bound, upp_bound, y_set)
            y_set = np.where(y_set < low_bound, low_bound, y_set)
            np.save(Path(save_path) / f"X_{datatype}_part_{part_num}.npy", X_set)
            np.save(Path(save_path) / f"y_{datatype}_part_{part_num}.npy", y_set)
            X_set = []
            y_set = []
    
    return 

if __name__ == "__main__":
    zero_ratio = pd.read_csv("E:/wuyuzc/AlphaNet/data_input/zero_ratio.csv", index_col="code")


    valid_code_list = zero_ratio[zero_ratio['ratio'] < 0.4].index.tolist()

    print(len(valid_code_list))
	
    sample_path = f"E:/wuyuzc/AlphaNet/processed data"
    date_list = ['20220425','20220426']
    datatype = 'train'
    save_path = "E:/wuyuzc/AlphaNet/data_input/train_data_set"    

    # prepare_dataset(sample_path, date_list, valid_code_list, datatype, save_path)
    
    date_list = ['20220427']
    datatype = 'val'
    prepare_dataset(sample_path, date_list, valid_code_list, datatype, save_path)
    






