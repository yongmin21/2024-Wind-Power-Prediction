import numpy as np
import pandas as pd

def NMAE(true, pred, capacity=20700):
    mae = np.mean(np.abs(true - pred))
    score = mae / capacity #np.mean(np.abs(true)) - original nmae
    return score * 100

def get_metrics(true, pred, capacity):
    result = pd.DataFrame()

    result['energy_kwh'] = true
    result['pred_energy_kwh'] = pred
    result['capacity'] = capacity

    result["normalized_abs_error"] = abs(result.pred_energy_kwh-result.energy_kwh)/result.capacity*100
    result['incentive'] = 0.
    result.loc[(result.normalized_abs_error > 6) & (result.normalized_abs_error <= 8), 'incentive'] = 3.
    result.loc[(result.normalized_abs_error <= 6), 'incentive'] = 4.
    result.loc[result.energy_kwh < result.capacity*0.1, 'incentive'] = 0.

    nmae = round(result.normalized_abs_error.mean(), 2)
    total_incentive = np.floor((result.incentive * true).sum())
    available_max_incentive = np.floor((4*true[true>=result.capacity*0.1])).sum()

    print("NMAE =", nmae, "%")
    print("예측정산금획득율 =", round(total_incentive/available_max_incentive*100, 2), "%")
    print("예측제도정산금 =", int(total_incentive), "원")
    return round(total_incentive/available_max_incentive*100, 2)