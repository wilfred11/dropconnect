import pickle
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.power import zt_ind_solve_power

def sm_ztest():
    # number of successes in each city
    count = np.array([300, 400])

    # total number of observations in each city
    nobs = np.array([500, 500])

    with open('result.pkl', 'rb') as f:
        no_drop = pickle.load(f)
    with open('result_drop.pkl', 'rb') as f:
        drop = pickle.load(f)
    print(no_drop)
    print(drop)
    no_drop_test= no_drop[1]
    drop_test= drop[1]
    success_counts = [int(no_drop_test[1]*no_drop_test[0]), int(drop_test[1]*drop_test[0])]
    observ_counts = [no_drop_test[0], drop_test[0]]
    print(success_counts)
    print(observ_counts)
    stat, pval = sm.stats.proportions_ztest(count=success_counts, nobs=observ_counts)
    print(stat, pval)
    return stat, pval


def power_analysis(effect_size, alpha, power, ratio=1):
    analysis = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio)
    return analysis




