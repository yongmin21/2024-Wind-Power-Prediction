import warnings
import sys
import gc
sys.path.append('../src')

import pandas as pd
import numpy as np

from astral import LocationInfo
from astral.sun import sun
import pytz

from windpowerlib.wind_speed import logarithmic_profile

from sklearn.base import BaseEstimator, TransformerMixin

#===========================================================
# Base Transformer Class
class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base Transformer for common fit and fit_transform methods."""

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
# Pivot Transformer
class PivotTransformer(BaseTransformer):
    """Pivot Data to Each Turbine's features

    Args:
        index (str): index of pivot
        columns (str): columns of pivot
        values (str): values of pivot
    """
    def __init__(self, index:str, columns:str, values:str, reset_columns:bool=True):
        self.pivot_config = {'index': index, 'columns': columns, 'values': values}
        self.reset_columns = reset_columns

    def transform(self, X, y=None):
        result = X.pivot(**self.pivot_config).reset_index()
        
        if self.reset_columns:
            result.columns = [' '.join(col).strip() for col in result.columns.values]
        return result

# Wind Vector to Wind Speed and Direction
class WindVectorTransformer(BaseTransformer):
    """Convert U, V wind component to WindSpeed, WindDirection by sklearn style.

    Args:
        BaseTransformer (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self):
        self.features = ['wind_u_10m', 'wind_v_10m']
        
    def transform(self, X, y=None):
        """Transform u,v components to wind speed and meteorological degree.
        NOTE: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv

        Parameters:
        X (pd.DataFrame): DataFrame that contains u, v component features

        Returns:
        X (pd.DataFrame): DataFrame with converted wind speed and direction
        """
        u, v = X[self.features[0]] + 1e-8, X[self.features[1]] + 1e-8
        X['wind_speed'] = np.nansum([u**2, v**2], axis=0)**(1/2.)

        # math degree
        wind_direction = np.rad2deg(np.arctan2(v, u))
        wind_direction[wind_direction < 0] += 360

        # meteorological degree
        wind_direction = 270 - wind_direction
        wind_direction[wind_direction < 0] += 360
        X['wind_direction'] = wind_direction

        gc.collect();
        return X

class LogarithmicWindTransformer(BaseTransformer):
    """
    Convert wind speed to hub height wind speed in sklearn style.

    Parameters:
        wind_speed_feature_name (str): Name of the wind speed feature.
        hub_height (int): Target height for the wind speed.
        roughness_length (int, float, str): Roughness length of the surface, can be a constant or a feature name.
    """
    
    def __init__(self, wind_speed, hub_height, roughness_length):
        self.logarithmic_config = {
            'wind_speed': wind_speed,
            'wind_speed_height': 10,
            'hub_height': hub_height,
            'roughness_length': roughness_length
        }
    
    def _wind_to_vector(self, wind_speed, wind_direction):
        """
        Convert wind speed and direction to vector components (u, v).

        Parameters:
            wind_speed (float): Wind speed.
            wind_direction (float): Wind direction in degrees.

        Returns:
            tuple: u and v components of the wind vector.
        """
        u = wind_speed * np.cos(np.deg2rad(wind_direction))
        v = wind_speed * np.sin(np.deg2rad(wind_direction))
        return u, v
    
    def fit(self, X, y=None):
        self.logarithmic_config['wind_speed'] = X['wind_speed']
        return self
    
    
    def transform(self, X: pd.DataFrame, y=None):
        """
        Transform wind speed to hub height wind speed using the logarithmic wind profile.

        Parameters:
            X (pd.DataFrame): DataFrame containing wind speed features.

        Returns:
            pd.DataFrame: Transformed DataFrame with converted wind speed and vector components.
        """

        X["wind_speed_100m"] = logarithmic_profile(**self.logarithmic_config)
        X["wind_u_100m"], X["wind_v_100m"] = self._wind_to_vector(X["wind_speed_100m"], X["wind_direction"])

        gc.collect();
        return X
        
    

class DatetimeEncoder(BaseTransformer):
    """Get Hour, Day, Month, Year, Season from dt by sklearn style.

        Parameters:
        location (str): 'gj' or 'yg'
        encoding (bool): if true then use cosine to cyclinic Encoding
    """

    def __init__(self,
                 location:str,
                 encoding:int|bool,
                 verbose:int|bool=0):
        
        self.location = location
        self.encoding = encoding
        self.verbose = verbose
        self.latlon = {
            'gj': [35.73088463, 129.3672852],
            'yg': [35.25257837, 126.3422734]
        }

    def _is_day_or_night_(self, dt):
        lat, lon = self.latlon[self.location][0], self.latlon[self.location][1]

        location = LocationInfo(self.location, "Korea", "Asia/Seoul", lat, lon)

        s = sun(location.observer, date=dt)
        sunrise = s['sunrise']
        sunset = s['sunset']

        if sunrise < dt < sunset:
                return 0 # Day
        else:
                return 1 # Night.
            
    def _cyclical_encoder(self, X, period):
        cos = np.cos(2 * np.pi * X / period)
        sin = np.sin(2 * np.pi * X / period)
        return cos, sin

    def transform(self, X, y=None):
        # get default time feature
        hour = X['dt'].dt.hour
        month = X['dt'].dt.month
        if self.encoding:
            X['cos_hour'], X['sin_hour'] = self._cyclical_encoder(hour, 24)
            X['cos_month'], X['sin_month'] = self._cyclical_encoder(month, 12)
            
        else:
            X['hour'], X['month'] = hour, month
            
        if self.verbose:
            print("create default time features")

        # get season feature
        X['season'] = (month % 12 // 3 + 1) # 1 : winter. 2 : spring, 3: summer, 4: fall
        if self.verbose:
            print("create season features")

        # get day/night feature
        # 경주 - 35.73088463, 129.3672852
        # 영광 - 35.25257837, 126.3422734
        X['Night'] = X['dt'].apply(lambda x: self._is_day_or_night_(x))
        
        if self.verbose:
            print("create day/night features")

        gc.collect();

        return X
        

class DeviationTransformer(BaseTransformer):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    def __init__(self, verbose:int|bool=0):
        self.verbose = verbose
        self.features = ['wind_u_100m', 'wind_v_100m', 'wind_intensity', 'wind_speed_100m','density', 
                         'temp_air', 'relative_humid', 'specific_humid', 'frictional_vmax_50m', 'frictional_vmin_50m']
        
    def _create_deviation_within_hour(self, df, num_features):
        result = pd.DataFrame()
        for f in num_features:
            feature = df.columns[df.columns.str.contains(f)]
            grouped_mean = df.groupby(df['dt'].dt.hour)[feature].transform('mean')
            deviation_col_name = 'hour_deviation_from_mean_' + feature
            new_columns = df[feature] - grouped_mean
            new_columns.columns = deviation_col_name
            result = pd.concat([result, new_columns], axis=1)
        return result
    
    def _create_deviation_within_night(self, df, num_features):
        result = pd.DataFrame()
        for f in num_features:
            feature = df.columns[df.columns.str.contains(f)]
            grouped_mean = df.groupby('Night')[feature].transform('mean')
            deviation_col_name = 'night_deviation_from_mean_' + feature
            new_columns = df[feature] - grouped_mean
            new_columns.columns = deviation_col_name
            result = pd.concat([result, new_columns], axis=1)
        return result

    
    def transform(self, X, y=None):
        hour_dev = self._create_deviation_within_hour(X, self.features)
        if self.verbose:
            print("create hour ensemble feature")
            
        night_dev = self._create_deviation_within_night(X, self.features)
        if self.verbose:
            print("create day/night ensemble feature")
            
        X = pd.concat([X, hour_dev, night_dev], axis=1)
        return X

    
class GlobalTransformer(BaseTransformer):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.features = ['wind_u_100m', 'wind_v_100m', 'wind_intensity', 'wind_speed_100m', 'density', 
                         'temp_air', 'relative_humid', 'specific_humid', 'frictional_vmax_50m', 'frictional_vmin_50m']
        
    def _create_global_features(self, df, num_features):
        result = pd.DataFrame()
        for f in num_features:
            feature = df.columns[df.columns.str.contains(f)]
            result[f'{f} mean'] = df[feature].mean(axis=1)
            result[f'{f} std'] = df[feature].std(axis=1)
            result[f'{f} median'] = df[feature].median(axis=1)
        return result
    
    def transform(self, X, y=None):
        global_agg = self._create_global_features(X, self.features)
        if self.verbose:
            print("create global aggregation feature")
            
        X = pd.concat([X, global_agg], axis=1)
        return X
    
class WeightsEncoder(BaseTransformer):
        def __init__(self, scada, verbose:int|bool=0):
            self.verbose = verbose
            self.scada = scada
            self.weights = {}
            self.features = ['wind_speed_100m', 'temp_air', 'density']
            
        def _wtg_gauss_mean(self, values):
            pivot_values = self.scada.pivot(index='dt',
                                            columns='turbine_id', 
                                            values=values)

            std, mean = pivot_values.std(), pivot_values.mean()  

            gauss = 1/(std * np.sqrt(2 * np.pi))*np.exp(-0.5 * np.square((pivot_values-mean) / (std + 1e-8)) + 1e-8)  
        
            gauss_mean = np.sum(gauss * pivot_values)/np.sum(gauss)
            dist = gauss_mean / np.sum(gauss_mean)
            return dist

        def transform(self, X, y=None):
            """Feature Engineering Codes
            """
            
            self.weights = {
                'wind_speed_100m': 1 - self._wtg_gauss_mean('Nacelle\nWind Speed\n[m/s]'),
                'temp_air': 1 - self._wtg_gauss_mean('Nacelle\nOutdoor Temp\n[℃]'),
                'density': 1 - self._wtg_gauss_mean('Nacelle\nAir Density\n[kg/㎥]')
            }

            # Dot weights
            for f, v in zip(self.features, self.weights.values()):
                feature = X.columns[X.columns.str.contains(f)]
                X[feature] *= v.values
                
            if self.verbose:
                print("Apply Weights to features")

            gc.collect();

            return X 
            
    
class FeatureTransformer(BaseTransformer):
    """Customize Features by sklearn style.
    """
    def __init__(self, verbose:int|bool=0):
        self.verbose = verbose

    def transform(self, X, y=None):
        """Feature Engineering Codes
        """

        # get density feature
        X['density'] = (X['pressure'])/(X['temp_air'] * 287)

        # pressure to hpa
        X['pressure'] = X['pressure'] / 100

        # cosine sine encoding
        X['wind_direction_cos'] = np.cos(2 * np.pi * X['wind_direction']/360)
        X['wind_direction_sin'] = np.sin(2 * np.pi * X['wind_direction']/360)
        
        if self.verbose:
            print("Creates Meteorological features")

        gc.collect();

        return X 

    
