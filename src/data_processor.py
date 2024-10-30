import warnings
import sys
import gc
sys.path.append('../src')

import pandas as pd
import numpy as np
import dask.dataframe as dd

from astral import LocationInfo
from astral.sun import sun
import pytz
#import swifter

from windpowerlib.wind_speed import logarithmic_profile
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.base import BaseEstimator, TransformerMixin

gj_coef = {
    '1': 0.267877,
    '2': 0.314450,
    '3': 0.360203,
    '4': 0.325319,
    '5': 0.245556,
    '6': 0.321135,
    '7': 0.332802,
    '8': 0.311006,
    '9': 0.292432
}

yg_coef = {
    '1': 0.352686,
    '2': 0.353894,
    '3': 0.305051,
    '4': 0.307169,
    '5': 0.318096,
    '6': 0.301351,
    '7': 0.268466,
    '8': 0.300878,
    '9': 0.269645,
    '10': 0.274170,
    '11': 0.251832,
    '12': 0.257768,
    '13': 0.249391,
    '14': 0.254078,
    '15': 0.238868,
    '16': 0.240151
}

class DatetimeLocalizer:
    """Convert Datetime to Local Time"""
    def __init__(self, 
                 local:str='Asia/Seoul'):
        
        self.local = local

    def fit(self, 
            X:pd.Series, 
            y=None):
        
        return self
    
    def transform(self,
                  X:pd.Series,
                  y=None):
        try:
            X = (pd.to_datetime(X)
                 .dt
                 .tz_convert("Asia/Seoul"))
            
        except TypeError:
            X = (pd.to_datetime(X)
                 .dt
                 .tz_localize("Asia/Seoul"))
            
        return X
    
    def fit_transform(self,
                      X:pd.Series,
                      y=None):
        
        return self.fit(X).transform(X)

# make data transformer
class UVTransformer(BaseEstimator, TransformerMixin):
    """Convert U, V wind component to WindSpeed, WindDirection by sklearn style.

        Parameters:
        u_feature_name (str): u component feature name
        v_feature_name (str): v component feature name
    """
    def __init__(self, u_feature_name:str, v_feature_name:str):
        self.u = u_feature_name
        self.v = v_feature_name
        
    def fit(self, X, y=None):
        """Take u, v components data

        Parameters:
        X (pd.DataFrame): DataFrame that contains u, v component features
        """
        if not all(feature in X.columns for feature in [self.u, self.v]):
            raise ValueError(f"'{self.u}' or '{self.v}' is not in the features of X")

        self.u_ws = X[self.u].to_numpy()
        self.v_ws = X[self.v].to_numpy()

        return self

    def transform(self, X, y=None):
        """Transform u,v components to wind speed and meteorological degree.
        NOTE: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv

        Parameters:
        X (pd.DataFrame): DataFrame that contains u, v component features

        Returns:
        X (pd.DataFrame): DataFrame with converted wind speed and direction
        """
        warnings.filterwarnings("ignore")

        # NOTE: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
        wind_speed = np.nansum([self.u_ws**2, self.v_ws**2], axis=0)**(1/2.)

        # math degree
        wind_direction = np.rad2deg(np.arctan2(self.v_ws, self.u_ws+1e-8))
        wind_direction[wind_direction < 0] += 360

        # meteorological degree
        wind_direction = 270 - wind_direction
        wind_direction[wind_direction < 0] += 360

        X['wind_speed'] = wind_speed
        X['wind_direction'] = wind_direction
        del wind_speed, wind_direction

        gc.collect();
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
class WindTransformer(BaseEstimator, TransformerMixin):
    """Convert WindSpeed to hub height Windspeed by sklearn style.

        Parameters:
        wind_speed_feature_name (str): windspeed feature name
        wind_speed_height (int): height of the wind speed
        hub_height (int): height of the target wind speed
        roughness_length (int,float,str) : roughness_length of the surface, can be constant or feature name
    """

    def Wind2Vec(self, ws, wd):
    
        u = ws * np.cos(np.deg2rad(wd))
        v = ws * np.sin(np.deg2rad(wd))

        return u, v

    def __init__(self, 
                 windspeed_feature_name:str,
                 wind_speed_height:int,
                 hub_height,
                 roughness_length):
        
        self.windspeed_str = windspeed_feature_name
        self.ref_height = wind_speed_height
        self.hub_height = hub_height
        self.rough = roughness_length
        
    def fit(self,
            X: pd.DataFrame,
            y=None):
        """Take u, v components data

        Parameters:
        X (pd.DataFrame): DataFrame that contains windspeed features
        """
        if not self.windspeed_str in X.columns:
            raise ValueError(f"'{self.windspeed_str}' is not in the features of X")

        self.windspeed = X[self.windspeed_str]

        return self

    def transform(self, X, y=None):
        """Transform windspeed to hub height by logarithmic wind profile.

        Parameters:
        X (pd.DataFrame): DataFrame that contains windspeed
        Returns:
        X (pd.DataFrame): DataFrame with converted wind speed
        """
        warnings.filterwarnings("ignore")
        X['wind_speed_100m'] = logarithmic_profile(self.windspeed, 
                                                   self.ref_height, 
                                                   self.hub_height, # 소재지표고에 따라 변하게 해야할것같음, 다만 이 함수에 series로 넣으면 에러가 남.
                                                   self.rough)
        
        X['wind_u_100m'], X['wind_v_100m'] = self.Wind2Vec(X['wind_speed_100m'], X['wind_direction'])
        X['wind_u_100m'], X['wind_v_100m'] = self.Wind2Vec(X['wind_speed_100m'], X['wind_direction'])

        gc.collect();
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
class TimeLagTransformer(BaseEstimator, TransformerMixin):
    """Make Feature for Time Lag by sklearn style.

        Parameters:
        time_lag (list) : time lag info
        rolling (bool): if rolling is True, then get time lag as rolling average window 
    """
    def __init__(self, 
                 time_lag:list,
                 rolling:bool=False):
        
        self.time_lag = time_lag
        self.rolling = rolling
        self.transform_feature = ['wind_speed_100m', 'wind_direction', 'density', 'temp_air']

    def fit(self,
            X:pd.DataFrame,
            y=None):

        return self
        
    def transform(self, 
                  X:pd.DataFrame,
                  y=None):
        
        if self.rolling:
            for feature in self.transform_feature:
                for lag in self.time_lag:
                    X[f"{feature}_rolling_avg_{lag}"] = X[feature].rolling(lag).mean()

        else:
            for feature in self.transform_feature:
                for lag in self.time_lag:
                    X[f"{feature}_lag_{lag}"] = X[feature].shift(lag)

        gc.collect();
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """Get Hour, Day, Month, Year, Season from dt by sklearn style.

        Parameters:
        location (str): 'gj' or 'yg'
        encoding (bool): if true then use cosine to cyclinic Encoding
    """

    def __init__(self,
                 location:str,
                 encoding:bool):
        
        self.location = location
        self.encoding = encoding

        self.latlon = {
            'gj': [35.73088463, 129.3672852],
            'yg': [35.25257837, 126.3422734]
        }

    def cyclical_encoding(self, data, period):
        pass

    def is_day_or_night_(self, dt):
        lat, lon = self.latlon[self.location][0], self.latlon[self.location][1]

        location = LocationInfo(self.location, "Korea", "Asia/Seoul", lat, lon)

        s = sun(location.observer, date=dt)
        sunrise = s['sunrise']
        sunset = s['sunset']
    #     dt = dt.tz_localize('Asia/Seoul')
        if sunrise < dt < sunset:
                return 0 # Day
        else:
                return 1 # Night.
        
    def fit(self, 
            X:pd.DataFrame,
            y=None):
        if 'dt' not in X.columns:
            raise ValueError('dt is not in X')
        else:
            self.datetime = X['dt']

        return self

    def transform(self, 
                  X:pd.DataFrame, 
                  y=None):
        
        X['hour'] = self.datetime.dt.hour
        X['day'] = self.datetime.dt.day
        X['month'] = self.datetime.dt.month
        X['year'] = self.datetime.dt.year

        # get season feature
        X['season'] = (X['month'] % 12 // 3 + 1) # 1 : winter. 2 : spring, 3: summer, 4: fall

        # get day/night feature
        # 경주 - 35.73088463, 129.3672852
        # 영광 - 35.25257837, 126.3422734
        X = dd.from_pandas(X, npartitions=4) # to speed-up
        X['Night'] = X['dt'].apply(lambda x: self.is_day_or_night_(x))

        gc.collect();

        return X.compute()

    def fit_transform(self, 
                      X:pd.DataFrame, 
                      y=None):
        
        return self.fit(X).transform(X)
    
class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Customize Features by sklearn style.
    """
    def __init__(self, 
                 feature_dev=[],
                 feature_diff=[],
                 feature_global=[],
                 windows = [1, 12, 24]
                 ):
        
        self.is_dev = feature_dev
        self.is_diff = feature_diff
        self.is_global = feature_global
        self.windows = windows
        #self.capacity = capacity

    def create_deviation_within_hours(self, df, num_features):
        result = pd.DataFrame()
        for f in num_features:
            feature = df.columns[df.columns.str.contains(f)]
            grouped_median = df.groupby(df['dt'].dt.hour)[feature].transform('mean')
            deviation_col_name = 'deviation_from_mean_' + feature
            new_columns = df[feature] - grouped_median
            new_columns.columns = deviation_col_name
            result = pd.concat([result, new_columns], axis=1)
        return result

    def create_diff_features(self, df, num_features):
        result = pd.DataFrame()
        for f in num_features:
            feature = df.columns[df.columns.str.contains(f)]
            for dt in self.windows:
                diff_col_name = f'diff_{dt}_' + feature
                new_columns = df[feature].diff(dt).bfill()
                new_columns.columns = diff_col_name
                result = pd.concat([result, new_columns], axis=1)
        return result

    def create_global_features(self, df, num_features):
        result = pd.DataFrame()
        for f in num_features:
            feature = df.columns[df.columns.str.contains(f)]
            result[f'{f} mean'] = df[feature].mean(axis=1)
            result[f'{f} std'] = df[feature].std(axis=1)
            result[f'{f} median'] = df[feature].median(axis=1)
        return result
        
    # def power_curve_fit_(self, speed, density):
    #     temp_speed = np.where(speed < 2.5, 0,
    #                         np.where(speed >= 20, 0, speed))

    #     ideal_energy = (temp_speed ** 3)*density*0.5

    #     curve_fitted = np.where(ideal_energy >= self.capacity, self.capacity, ideal_energy)

    #     return curve_fitted

    def fit(self,
            X:pd.DataFrame,
            y=None):
        return self

    def transform(self, 
                X,
                y=None):
        """Feature Engineering Codes
        """

        # get density feature
        X['density'] = (X['pressure'])/(X['temp_air'] * 287)

        # pressure to hpa
        X['pressure'] = X['pressure'] / 100

        # get shear stress
        X['shear_stress'] = (X['frictional_vmax_50m'] ** 2) * X['density']

        # cosine sine encoding
        X['wind_direction_cos'] = np.cos(2 * np.pi * X['wind_direction']/360)
        X['wind_direction_sin'] = np.sin(2 * np.pi * X['wind_direction']/360)

        # deviation
        if len(self.is_dev) != 0:
            dev = self.create_deviation_within_hours(X, list(self.is_dev))
            X = pd.concat([X, dev], axis=1)
        
        if len(self.is_diff) != 0:
            diff = self.create_diff_features(X, list(self.is_diff))
            X = pd.concat([X, diff], axis=1)
        
        if len(self.is_global) != 0:
            global_ = self.create_global_features(X, list(self.is_global))
            X = pd.concat([X, global_], axis=1)

        gc.collect();

        return X  # 결과를 반환하기 전에 compute()를 호출하여 실행
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)