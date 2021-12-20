import numpy as np
import pandas as pd
from UI.config import FTOR_DECODE


def convert_params_to_readable(res: dict):
    # Расшифровка типа границ и типа скважины
    res['kind_code'] = FTOR_DECODE['kind_code'][res['kind_code']]
    # Расшифровка названий параметров адаптации
    for key in FTOR_DECODE.keys():
        if key in res.keys():
            res[FTOR_DECODE[key]['label']] = res.pop(key)
    return res


def extract_data_ftor(_calculator_ftor, state):
    dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
    state.statistics['ftor'] = pd.DataFrame(index=dates)
    for well_ftor in _calculator_ftor.wells:
        well_name_ois = well_ftor.well_name
        well_name_normal = state.wellnames_key_ois[well_name_ois]
        res_ftor = well_ftor.results
        adapt_params = res_ftor.adap_and_fixed_params[0]
        state.adapt_params[well_name_normal] = convert_params_to_readable(adapt_params.copy())
        # Жидкость. Полный ряд (train + test)
        rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
        rates_liq_ftor = pd.to_numeric(rates_liq_ftor)
        # Нефть. Только test
        rates_oil_test_ftor = res_ftor.rates_oil_test
        rates_oil_test_ftor = pd.to_numeric(rates_oil_test_ftor)
        # Фактические данные для визуализации
        df = well_ftor.df_chess
        state.statistics['ftor'][f'{well_name_normal}_liq_true'] = df['Дебит жидкости']
        state.statistics['ftor'][f'{well_name_normal}_liq_pred'] = rates_liq_ftor
        state.statistics['ftor'][f'{well_name_normal}_oil_true'] = df['Дебит нефти']
        state.statistics['ftor'][f'{well_name_normal}_oil_pred'] = rates_oil_test_ftor


def extract_data_wolfram(_calculator_wolfram, state):
    dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
    state.statistics['wolfram'] = pd.DataFrame(index=dates)
    for _well_wolfram in _calculator_wolfram.wells:
        _well_name_ois = _well_wolfram.well_name
        res_wolfram = _well_wolfram.results
        # Фактические данные (вторично) извлекаются из wolfram, т.к. он использует
        # для вычислений максимально возможный доступный ряд фактичесих данных.
        df_true = _well_wolfram.df
        rates_liq_true = df_true[_well_wolfram.NAME_RATE_LIQ]
        rates_oil_true = df_true[_well_wolfram.NAME_RATE_OIL]
        rates_liq_wolfram = res_wolfram.rates_liq_test
        rates_oil_wolfram = res_wolfram.rates_oil_test

        well_name_normal = state.wellnames_key_ois[_well_name_ois]
        state.statistics['wolfram'][f'{well_name_normal}_liq_true'] = rates_liq_true
        state.statistics['wolfram'][f'{well_name_normal}_liq_pred'] = rates_liq_wolfram
        state.statistics['wolfram'][f'{well_name_normal}_oil_true'] = rates_oil_true
        state.statistics['wolfram'][f'{well_name_normal}_oil_pred'] = rates_oil_wolfram


def extract_data_CRM(df, state, wells_wolfram, mode='CRM'):
    dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
    for well in wells_wolfram:
        well_name_normal = state.wellnames_key_ois[well.well_name]
        if well_name_normal in df.columns:
            if mode not in state.statistics:
                state.statistics[mode] = pd.DataFrame(index=dates)
            df_fact = well.df_chess
            state.statistics[mode][f'{well_name_normal}_liq_true'] = df_fact['Дебит жидкости']
            state.statistics[mode][f'{well_name_normal}_liq_pred'] = df[well_name_normal]
            state.statistics[mode][f'{well_name_normal}_oil_true'] = np.nan
            state.statistics[mode][f'{well_name_normal}_oil_pred'] = np.nan


def convert_tones_to_m3_for_wolfram(state, wells_ftor):
    for well_ftor in wells_ftor:
        density_oil = well_ftor.density_oil
        well_name_normal = state.wellnames_key_ois[well_ftor.well_name]
        state.statistics['wolfram'][f'{well_name_normal}_oil_true'] /= density_oil
        state.statistics['wolfram'][f'{well_name_normal}_oil_pred'] /= density_oil


def prepare_df_for_ensemble(state, well_name_normal, name_of_y_true):
    models = list(state.statistics.keys())
    if 'ensemble' in models:
        models.remove('ensemble')
    dates_test = pd.date_range(state.was_date_test, state.was_date_end, freq='D').date
    input_df_for_ensemble = pd.DataFrame(index=dates_test)
    for model in models:
        if f'{well_name_normal}_oil_pred' in state.statistics[model]:
            input_df_for_ensemble[name_of_y_true] = state.statistics[model][f'{well_name_normal}_oil_true']
            input_df_for_ensemble[model] = state.statistics[model][f'{well_name_normal}_oil_pred']
    return input_df_for_ensemble


def extract_data_ensemble(ensemble_df, state, well_name_normal):
    dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
    if 'ensemble' not in state.statistics:
        state.statistics['ensemble'] = pd.DataFrame(index=dates)
    state.statistics['ensemble'][f'{well_name_normal}_liq_true'] = np.nan
    state.statistics['ensemble'][f'{well_name_normal}_liq_pred'] = np.nan
    state.statistics['ensemble'][f'{well_name_normal}_oil_true'] = ensemble_df['true']
    state.statistics['ensemble'][f'{well_name_normal}_oil_pred'] = ensemble_df['ensemble']

    if 'ensemble_intervals' not in state:
        state['ensemble_intervals'] = pd.DataFrame(index=dates)
    state.ensemble_interval[f'{well_name_normal}_upper'] = ensemble_df['interval_upper']
    state.ensemble_interval[f'{well_name_normal}_lower'] = ensemble_df['interval_lower']


def make_models_stop_well(statistics, well_names):
    # Зануление значений по моделям, когда фактический дебит равен нулю или NaN
    for model in statistics:
        for well_name in well_names:
            if f'{well_name}_oil_pred' not in statistics[model]:
                continue
            liq_zeros = statistics[model][f'{well_name}_liq_true'] == 0
            liq_nans = statistics[model][f'{well_name}_liq_true'].isna()
            statistics[model][f'{well_name}_liq_pred'][liq_zeros | liq_nans] = np.nan

            oil_zeros = statistics[model][f'{well_name}_oil_true'] == 0
            oil_nans = statistics[model][f'{well_name}_oil_true'].isna()
            statistics[model][f'{well_name}_oil_pred'][oil_zeros | oil_nans] = np.nan


def cut_statistics_test_only(state):
    statistics_test_index = pd.date_range(state.was_date_test, state.was_date_end, freq='D')
    # обрезка данных по датам(индексу) ансамбля
    if state.was_calc_ensemble:
        statistics_test_index = pd.date_range(state.was_date_test_if_ensemble, state.was_date_end, freq='D')

    statistics_test_only = {}
    for key in state.statistics:
        statistics_test_only[key] = state.statistics[key].copy().reindex(statistics_test_index).fillna(0)
    return statistics_test_only, statistics_test_index
