import datetime
import numpy as np
import pandas as pd


def extract_data_ftor(_calculator_ftor, session):
    session.statistics['ftor'] = pd.DataFrame(index=session.dates)
    for well_ftor in _calculator_ftor.wells:
        _well_name_ois = well_ftor.well_name
        res_ftor = well_ftor.results
        session.adapt_params[_well_name_ois] = res_ftor.adap_and_fixed_params
        # Жидкость. Полный ряд (train + test)
        rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
        rates_liq_ftor = pd.to_numeric(rates_liq_ftor)
        # Нефть. Только test
        rates_oil_test_ftor = res_ftor.rates_oil_test
        rates_oil_test_ftor = pd.to_numeric(rates_oil_test_ftor)

        # df_liq[_well_name_ois]['ftor'] = rates_liq_ftor
        # df_oil[_well_name_ois]['ftor'] = rates_oil_test_ftor
        # Фактические данные для визуализации
        df = well_ftor.df_chess
        # session.events[_well_name_ois] = df['Мероприятие']
        # session.df_draw_liq[_well_name_ois]['true'] = df['Дебит жидкости']
        # session.df_draw_oil[_well_name_ois]['true'] = df['Дебит нефти']
        # session.pressure[_well_name_ois] = df['Давление забойное']
        well_name_normal = session.wellnames_key_ois[_well_name_ois]
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
        bh_pressure = df_true[_well_wolfram.NAME_PRESSURE]
        rates_liq_wolfram = pd.concat(objs=[res_wolfram.rates_liq_train, res_wolfram.rates_liq_test])
        rates_oil_wolfram = pd.concat(objs=[res_wolfram.rates_oil_train, res_wolfram.rates_oil_test])

        # df_liq[_well_name_ois]['wolfram'] = rates_liq_wolfram
        # df_liq[_well_name_ois]['true'] = rates_liq_true
        # df_oil[_well_name_ois]['wolfram'] = rates_oil_wolfram
        # df_oil[_well_name_ois]['true'] = rates_oil_true
        # pressure[_well_name_ois] = bh_pressure[pressure[_well_name_ois].index]
        well_name_normal = session.wellnames_key_ois[_well_name_ois]
        session.statistics['wolfram'][f'{well_name_normal}_liq_true'] = rates_liq_true
        session.statistics['wolfram'][f'{well_name_normal}_liq_pred'] = rates_liq_wolfram
        session.statistics['wolfram'][f'{well_name_normal}_oil_true'] = rates_oil_true
        session.statistics['wolfram'][f'{well_name_normal}_oil_pred'] = rates_oil_wolfram


def extract_data_CRM(df_CRM, session, preprocessor):
    session.statistics['CRM'] = pd.DataFrame(index=session.dates)
    for well in preprocessor.create_wells_ftor(session.selected_wells_ois):
        if well.well_name in df_CRM.columns:
            well_name_normal = session.wellnames_key_ois[well.well_name]
            session.statistics['CRM'][f'{well_name_normal}_liq_true'] = np.nan
            session.statistics['CRM'][f'{well_name_normal}_liq_pred'] = np.nan
            well = preprocessor.create_wells_ftor([well.well_name])[0]  # достаем данные о фактич. дебите
            session.statistics['CRM'][f'{well_name_normal}_oil_true'] = well.df_chess['Дебит нефти']
            session.statistics['CRM'][f'{well_name_normal}_oil_pred'] = session.df_CRM[well.well_name]


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


def create_statistics_df_test(session):
    test_dates = pd.date_range(session.date_test, session.date_end, freq='D')
    # TODO: обрезка данных по датам(индексу) ансамбля. В будущем можно убрать.
    if session.was_calc_ensemble:
        date_start_ensemble = session.date_test + datetime.timedelta(days=session.adaptation_days_number)
        test_dates = pd.date_range(date_start_ensemble, session.date_end, freq='D')

    statistics_test = {}
    for key in session.statistics:
        statistics_test[key] = session.statistics[key].copy().reindex(test_dates).fillna(0)
    return statistics_test, test_dates
