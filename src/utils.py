import pandas as pd
import numpy as np
import config


def make_train_test(pivot_df, train_y, test_size=0.2):
    dataset = pd.merge(pivot_df, train_y[['dt', 'energy_kwh']],
                   how='left',
                   on='dt')
    size = dataset.shape[0]
    threshold = int(size * (1 - test_size))
    train = dataset.iloc[:threshold, :]
    test = dataset.iloc[threshold:]

    x_train, y_train = train.drop(['dt', 'energy_kwh'], axis=1), train['energy_kwh']
    x_test, y_test = test.drop(['dt', 'energy_kwh'], axis=1), test['energy_kwh']

    gc.collect();
    return dataset, x_train, x_test, y_train, y_test


def reduce_mem_usage(df: pd.DataFrame,
                    use_float16: bool = False) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.       

    Keyword arguments:
    df -- raw pandas dataframe
    use_float16 -- option for tight memory usage instead of float32 
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue

    end_mem = df.memory_usage().sum() / 1024**2
    
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def uv_to_wsd(u_wind_speed, v_wind_speed):
    """ 
        Convert u, v vector to wind speed and direction.
    """
    u_ws = u_wind_speed.to_numpy()
    v_ws = v_wind_speed.to_numpy()

    # NOTE: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    wind_speed = np.nansum([u_ws**2, v_ws**2], axis=0)**(1/2.)

    # math degree
    wind_direction = np.rad2deg(np.arctan2(v_ws, u_ws+1e-6))
    wind_direction[wind_direction < 0] += 360

    # meteorological degree
    wind_direction = 270 - wind_direction
    wind_direction[wind_direction < 0] += 360

    return wind_speed, wind_direction