from abc import ABC, abstractmethod
import pandas as pd
import config

class Data(ABC):
    """MetaClass for Datas
    """ 
    @abstractmethod
    def load(self):
        pass

class YeongGwang(Data):
    def __init__(self):
        self.ldaps_path = config.input_path + "train_ldaps_yeonggwang.parquet"
        self.data = pd.DataFrame()
        
    def _set_timezone(self):
        try:
            self.data['dt'] = (pd.to_datetime(self.data['dt'])
                               .dt
                               .tz_convert("Asia/Seoul"))
        except TypeError:
            self.data['dt'] = (pd.to_datetime(self.data['dt'])
                               .dt
                               .tz_localize("Asia/Seoul"))
            
    def load(self, start_date, end_date, inclusive='left'):
        self.data = pd.read_parquet(self.ldaps_path)
        self._set_timezone()
        output = self.data[self.data['dt'].between(start_date, end_date, inclusive=inclusive)]
        
        return output

class GyeongJu(Data):
    def __init__(self):
        self.ldaps_path = config.input_path + "train_ldaps_gyeongju.parquet"
        self.data = pd.DataFrame()
        
    def _set_timezone(self):
        try:
            self.data['dt'] = (pd.to_datetime(self.data['dt'])
                               .dt
                               .tz_convert("Asia/Seoul"))
        except TypeError:
            self.data['dt'] = (pd.to_datetime(self.data['dt'])
                               .dt
                               .tz_localize("Asia/Seoul"))
            
    def load(self, start_date, end_date, inclusive='left'):
        self.data = pd.read_parquet(self.ldaps_path)
        self._set_timezone()
        output = self.data[self.data['dt'].between(start_date, end_date, inclusive=inclusive)]
        
        return output

class Target(Data):
    def __init__(self, plant_name):
        self.target_path = config.input_path + "train_y.parquet"
        self.plant_name = plant_name
        self.data = pd.DataFrame()
        
    def _set_timezone(self):
        try:
            self.data['dt'] = (pd.to_datetime(self.data['dt'])
                               .dt
                               .tz_convert("Asia/Seoul"))
        except TypeError:
            self.data['dt'] = (pd.to_datetime(self.data['dt'])
                               .dt
                               .tz_localize("Asia/Seoul"))
            
    def load(self, start_date, end_date, inclusive='left'):
        self.data = (pd.read_parquet(self.target_path)
                     .rename({'end_datetime': 'dt'}, axis=1))
        self._set_timezone()
        output = (self.data.loc[(self.data['plant_name'] == self.plant_name)
                               & (self.data['dt']).between(start_date, end_date, inclusive=inclusive)])
        
        return output