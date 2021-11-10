import datetime
import numpy as np
import pandas as pd
from UI.config import FTOR_DECODE


def convert_params_to_readable(res: dict):
    if 'boundary_code' in res.keys():
        # Расшифровка типа границ и типа скважины
        res['boundary_code'] = FTOR_DECODE['boundary_code'][res['boundary_code']]
        res['kind_code'] = FTOR_DECODE['kind_code'][res['kind_code']]
        # Расшифровка названий параметров адаптации
        for key in FTOR_DECODE.keys():
            if key in res.keys():
                res[FTOR_DECODE[key]['label']] = res.pop(key)
    return res


def extract_data_ftor(_calculator_ftor, session):
    session.statistics['ftor'] = pd.DataFrame(index=session.dates)
    for well_ftor in _calculator_ftor.wells:
        well_name_ois = well_ftor.well_name
        well_name_normal = session.wellnames_key_ois[well_name_ois]
        res_ftor = well_ftor.results
        adapt_params = res_ftor.adap_and_fixed_params[0]
        session.adapt_params[well_name_normal] = convert_params_to_readable(adapt_params.copy())
        # Жидкость. Полный ряд (train + test)
        rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
        rates_liq_ftor = pd.to_numeric(rates_liq_ftor)
        # Нефть. Только test
        rates_oil_test_ftor = res_ftor.rates_oil_test
        rates_oil_test_ftor = pd.to_numeric(rates_oil_test_ftor)
        # Фактические данные для визуализации
        df = well_ftor.df_chess
        session.statistics['ftor'][f'{well_name_normal}_liq_true'] = df['Дебит жидкости']
        session.statistics['ftor'][f'{well_name_normal}_liq_pred'] = rates_liq_ftor
        session.statistics['ftor'][f'{well_name_normal}_oil_true'] = df['Дебит нефти']
        session.statistics['ftor'][f'{well_name_normal}_oil_pred'] = rates_oil_test_ftor


def extract_data_wolfram(_calculator_wolfram, session):
    session.statistics['wolfram'] = pd.DataFrame(index=session.dates)
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

        well_name_normal = session.wellnames_key_ois[_well_name_ois]
        session.statistics['wolfram'][f'{well_name_normal}_liq_true'] = rates_liq_true
        session.statistics['wolfram'][f'{well_name_normal}_liq_pred'] = rates_liq_wolfram
        session.statistics['wolfram'][f'{well_name_normal}_oil_true'] = rates_oil_true
        session.statistics['wolfram'][f'{well_name_normal}_oil_pred'] = rates_oil_wolfram


def extract_data_CRM(df_CRM, session, wells_wolfram):
    for well in wells_wolfram:
        if well.well_name in df_CRM.columns:
            if 'CRM' not in session.statistics:
                session.statistics['CRM'] = pd.DataFrame(index=session.dates)
            df_true = well.df
            rates_oil_true = df_true[well.NAME_RATE_OIL]
            well_name_normal = session.wellnames_key_ois[well.well_name]
            session.statistics['CRM'][f'{well_name_normal}_liq_true'] = np.nan
            session.statistics['CRM'][f'{well_name_normal}_liq_pred'] = np.nan
            session.statistics['CRM'][f'{well_name_normal}_oil_true'] = rates_oil_true
            session.statistics['CRM'][f'{well_name_normal}_oil_pred'] = df_CRM[well.well_name]


def convert_tones_to_m3_for_wolfram(session, wells_ftor):
    for well_ftor in wells_ftor:
        density_oil = well_ftor.density_oil
        well_name_normal = session.wellnames_key_ois[well_ftor.well_name]
        session.statistics['wolfram'][f'{well_name_normal}_oil_true'] /= density_oil
        session.statistics['wolfram'][f'{well_name_normal}_oil_pred'] /= density_oil


def prepare_df_for_ensemble(session, well_name_normal, name_of_y_true):
    models = list(session.statistics.keys())
    if 'ensemble' in models:
        models.remove('ensemble')
    dates_test_period = pd.date_range(session.date_test, session.date_end, freq='D').date
    input_df_for_ensemble = pd.DataFrame(index=dates_test_period)
    for model in models:
        if f'{well_name_normal}_oil_pred' in session.statistics[model]:
            input_df_for_ensemble[name_of_y_true] = session.statistics[model][f'{well_name_normal}_oil_true']
            input_df_for_ensemble[model] = session.statistics[model][f'{well_name_normal}_oil_pred']
    return input_df_for_ensemble


def extract_data_ensemble(ensemble_df, session, well_name_normal):
    # date_range_test = pd.date_range(session.date_test, session.date_end, freq='D').date
    if 'ensemble' not in session.statistics:
        session.statistics['ensemble'] = pd.DataFrame(index=session.dates)
    session.statistics['ensemble'][f'{well_name_normal}_liq_true'] = np.nan
    session.statistics['ensemble'][f'{well_name_normal}_liq_pred'] = np.nan
    session.statistics['ensemble'][f'{well_name_normal}_oil_true'] = ensemble_df['true']
    session.statistics['ensemble'][f'{well_name_normal}_oil_pred'] = ensemble_df['ensemble']

    if 'ensemble_intervals' not in session:
        session['ensemble_intervals'] = pd.DataFrame(index=session.dates)
    session.ensemble_interval[f'{well_name_normal}_upper'] = ensemble_df['interval_upper']
    session.ensemble_interval[f'{well_name_normal}_lower'] = ensemble_df['interval_lower']


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


def create_statistics_df_test(session):
    dates_test_period = pd.date_range(session.date_test, session.date_end, freq='D')
    # обрезка данных по датам(индексу) ансамбля
    if session.was_calc_ensemble:
        date_start_ensemble = session.date_test + datetime.timedelta(days=session.adaptation_days_number)
        dates_test_period = pd.date_range(date_start_ensemble, session.date_end, freq='D')

    statistics_df_test = {}
    for key in session.statistics:
        statistics_df_test[key] = session.statistics[key].copy().reindex(dates_test_period).fillna(0)
    return statistics_df_test, dates_test_period
