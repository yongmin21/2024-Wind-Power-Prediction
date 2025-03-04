from dataclasses import dataclass
from typing import Optional
import pathlib

workspace = pathlib.Path().parent

@dataclass
class DefaultConfig:
       input_path = f"{workspace}/input/"
       model_output = f"{workspace}/models/"
       target = 'energy_kwh'
       test_size = 0.2
       mlflow = False

@dataclass
class LightGBMConfig:
       n_estimators: int = 1000
       learning_rate: float = 0.1
       max_depth: int = -1
       objective: str = 'mae'
       random_state: int = 1
       metric: Optional[str] = "l1"
       lambda_l2: Optional[int] = 3
       feature_fraction: Optional[float] = None
       boosting_type: Optional[str] = None
       rate_drop: Optional[float] = None
       skip_drop: Optional[float] = None
       n_jobs: Optional[int] = -1
       device_type: Optional[str] = "gpu"
       tree_learner: Optional[str] = 'feature'

       @classmethod
       def exp_yeonggwang(cls):
              return cls(
                     n_estimators=5000,
                     feature_fraction=0.7,
                     objective="fair",
                     boosting_type="dart",
                     rate_drop=0.6,
                     skip_drop=0.7
              )
       
       @classmethod
       def exp_gyeongju(cls):
              return cls(
                     n_estimators=3000,
                     learning_rate=0.07,
                     feature_fraction=0.5,
                     objective="fair",
                     boosting_type="dart",
                     rate_drop=0.5,
                     skip_drop=0.7
              )