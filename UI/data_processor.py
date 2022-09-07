from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pathlib

from frameworks_hybrid_crm_ml.class_Fedot.calculator import CalculatorFedot
from UI.app_state import AppState
from UI.config import FTOR_DECODE
# from frameworks_crm.class_Fedot.fedot_model import FedotModel
from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from frameworks_ftor.ftor.well import Well as WellFtor
from frameworks_wolfram.wolfram.calculator import Calculator as CalculatorWolfram
from frameworks_shelf_algo.class_Shelf.data_postprocessor_shelf import DataPostProcessorShelf
from frameworks_shelf_algo.class_Shelf.calculator import CalculatorShelf


def convert_params_to_readable(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """ Расшифровка названий параметров адаптации."""
    parsed_dict = params_dict.copy()
    # Расшифровка типа скважины
    parsed_dict['kind_code'] = FTOR_DECODE['kind_code'][parsed_dict['kind_code']]
    # Расшифровка названий параметров адаптации
    for key in FTOR_DECODE.keys():
        if key in parsed_dict.keys():
            parsed_dict[FTOR_DECODE[key]['label']] = parsed_dict.pop(key)
    return parsed_dict



def extract_data_ftor(_calculator_ftor: CalculatorFtor, state: AppState) -> None:
    if 'ftor' not in state.statistics.keys():
        dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
        state.statistics['ftor'] = pd.DataFrame(index=dates)
        for well_ftor in _calculator_ftor.wells:
            well_name_ois = well_ftor.well_name
            well_name_normal = state.wellnames_key_ois[well_name_ois]
            res_ftor = well_ftor.results
            adapt_params = res_ftor.adap_and_fixed_params[0]
            state.adapt_params[well_name_normal] = convert_params_to_readable(adapt_params)
            # Жидкость. Полный ряд (train + test)
            rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
            rates_liq_ftor = pd.to_numeric(rates_liq_ftor)
            # Нефть. Только test
            rates_oil_test_ftor = res_ftor.rates_oil_test
            rates_oil_test_ftor = pd.to_numeric(rates_oil_test_ftor)
            df = well_ftor.df_chess  # Фактические данные для визуализации
            state.statistics['ftor'][f'{well_name_normal}_liq_true'] = df['Дебит жидкости']
            state.statistics['ftor'][f'{well_name_normal}_liq_pred'] = rates_liq_ftor
            state.statistics['ftor'][f'{well_name_normal}_oil_true'] = df['Дебит нефти']
            state.statistics['ftor'][f'{well_name_normal}_oil_pred'] = rates_oil_test_ftor
    else:
        dates = pd.date_range(state.was_date_test_if_ensemble, state.was_date_end, freq='D').date
        for well_ftor in _calculator_ftor.wells:
            well_name_ois = well_ftor.well_name
            well_name_normal = state.wellnames_key_ois[well_name_ois]
            res_ftor = well_ftor.results
            adapt_params = res_ftor.adap_and_fixed_params[0]
            state.adapt_params[well_name_normal] = convert_params_to_readable(adapt_params)
            # Жидкость. Полный ряд (train + test)
            rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
            rates_liq_ftor = pd.to_numeric(rates_liq_ftor)
            # Нефть. Только test
            rates_oil_test_ftor = res_ftor.rates_oil_test
            rates_oil_test_ftor = pd.to_numeric(rates_oil_test_ftor)
            df = well_ftor.df_chess  # Фактические данные для визуализации
            if f'{well_name_normal}_liq_true' in state.statistics['ftor'].columns:
                state.statistics['ftor'][f'{well_name_normal}_liq_true'][dates] = df['Дебит жидкости'][dates]
                state.statistics['ftor'][f'{well_name_normal}_liq_pred'][dates] = rates_liq_ftor[dates]
                state.statistics['ftor'][f'{well_name_normal}_oil_true'][dates] = df['Дебит нефти'][dates]
                state.statistics['ftor'][f'{well_name_normal}_oil_pred'][dates] = rates_oil_test_ftor[dates]


def extract_data_wolfram(_calculator_wolfram: CalculatorWolfram, state: AppState) -> None:
    if 'wolfram' not in state.statistics.keys():
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
    else:
        dates = pd.date_range(state.was_date_test_if_ensemble, state.was_date_end, freq='D').date
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
            if (f'{well_name_normal}_liq_true' in state.statistics['wolfram'].columns) and (f'{well_name_normal}_oil_true' in state.statistics['wolfram'].columns):
                state.statistics['wolfram'][f'{well_name_normal}_liq_true'][dates] = rates_liq_true[dates]
                state.statistics['wolfram'][f'{well_name_normal}_liq_pred'][dates] = rates_liq_wolfram[dates]
                state.statistics['wolfram'][f'{well_name_normal}_oil_true'][dates] = rates_oil_true[dates]
                state.statistics['wolfram'][f'{well_name_normal}_oil_pred'][dates] = rates_oil_wolfram[dates]



def extract_data_CRM(df: pd.DataFrame,
                     state: AppState,
                     wells_ftor: List[WellFtor],
                     df_for_ensemble=None,
                     mode: str = 'CRM') -> None:
    if df_for_ensemble is None:
        dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
        for well in wells_ftor:
            well_name_normal = state.wellnames_key_ois[well.well_name]
            if well_name_normal in df.columns:
                if mode not in state.statistics:
                    state.statistics[mode] = pd.DataFrame(index=dates)
                df_fact = well.df_chess
                state.statistics[mode][f'{well_name_normal}_liq_true'] = df_fact['Дебит жидкости']
                state.statistics[mode][f'{well_name_normal}_liq_pred'] = df[well_name_normal]
                state.statistics[mode][f'{well_name_normal}_oil_true'] = np.nan
                state.statistics[mode][f'{well_name_normal}_oil_pred'] = np.nan
    else:
        dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
        dates_ensemble = pd.date_range(state.was_date_test_if_ensemble, state.was_date_end, freq='D').date
        for well in wells_ftor:
            well_name_normal = state.wellnames_key_ois[well.well_name]
            if well_name_normal in df.columns:
                if mode not in state.statistics:
                    state.statistics[mode] = pd.DataFrame(index=dates)
                df_fact = well.df_chess
                state.statistics[mode][f'{well_name_normal}_liq_true'] = df_fact['Дебит жидкости']
                state.statistics[mode][f'{well_name_normal}_liq_pred'] = df_for_ensemble[well_name_normal]
                state.statistics[mode][f'{well_name_normal}_oil_true'] = np.nan
                state.statistics[mode][f'{well_name_normal}_oil_pred'] = np.nan

                state.statistics[mode][f'{well_name_normal}_liq_pred'][dates_ensemble] = df[well_name_normal][dates_ensemble]


def extract_data_fedot(fedot_entity: CalculatorFedot, state: AppState) -> None:
    if 'fedot' not in state.statistics.keys():
        state.statistics['fedot'] = fedot_entity.statistic_all
    else:
        dates = pd.date_range(state.was_date_test_if_ensemble, state.was_date_end, freq='D').date
        state.statistics['fedot'] = pd.concat([state.statistics['fedot'].drop([dates[0]]),
                                               pd.DataFrame(columns=state.statistics['fedot'].columns, index=dates)])
        for well_fedot in state.statistics['fedot'].columns:
            state.statistics['fedot'][well_fedot][dates] = fedot_entity.statistic_all[well_fedot][dates]

def extract_data_shelf(_calculator_shelf: CalculatorShelf, state: AppState, _change_gtm_info: int) -> None:
    if 'shelf' not in state.statistics.keys():
        dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
        state.statistics['shelf'] = pd.DataFrame(index=dates)
        for well_shelf in _calculator_shelf.wells_list:
            well_name_ois = well_shelf
            well_name_normal = state.wellnames_key_ois[well_name_ois]
            if well_name_ois in _calculator_shelf.df_result:
                res_oil = _calculator_shelf.df_result[well_name_ois]
                res_liq = _calculator_shelf.df_result_liq[well_name_ois]
                true_oil = _calculator_shelf._df_fact_test_prd[well_name_ois]
                true_liq = _calculator_shelf._df_fact_test_prd_liq[well_name_ois]
                state.statistics['shelf'][f'{well_name_normal}_liq_true'] = true_liq
                state.statistics['shelf'][f'{well_name_normal}_liq_pred'] = res_liq
                state.statistics['shelf'][f'{well_name_normal}_oil_true'] = true_oil
                state.statistics['shelf'][f'{well_name_normal}_oil_pred'] = res_oil
    else:
        dates = pd.date_range(state.was_date_test_if_ensemble, state.was_date_end, freq='D').date
        for well_shelf in _calculator_shelf.wells_list:
            well_name_ois = well_shelf
            well_name_normal = state.wellnames_key_ois[well_name_ois]
            if well_name_ois not in _calculator_shelf.df_result:
                continue
            res_oil = _calculator_shelf.df_result[well_name_ois]
            res_liq = _calculator_shelf.df_result_liq[well_name_ois]
            true_oil = _calculator_shelf._df_fact_test_prd[well_name_ois]
            true_liq = _calculator_shelf._df_fact_test_prd_liq[well_name_ois]
            if f'{well_name_normal}_liq_true' in state.statistics['shelf']:
                state.statistics['shelf'][f'{well_name_normal}_liq_true'][dates] = true_liq[dates]
                state.statistics['shelf'][f'{well_name_normal}_liq_pred'][dates] = res_liq[dates]
                state.statistics['shelf'][f'{well_name_normal}_oil_true'][dates] = true_oil[dates]
                state.statistics['shelf'][f'{well_name_normal}_oil_pred'][dates] = res_oil[dates]


def convert_tones_to_m3_for_wolfram(state: AppState, wells_ftor: List[WellFtor]) -> None:
    for well_ftor in wells_ftor:
        density_oil = well_ftor.density_oil
        well_name_normal = state.wellnames_key_ois[well_ftor.well_name]
        if f'{well_name_normal}_oil_true' in state.statistics['wolfram'].columns:
            state.statistics['wolfram'][f'{well_name_normal}_oil_true'] /= density_oil
            state.statistics['wolfram'][f'{well_name_normal}_oil_pred'] /= density_oil


def prepare_data_for_ensemble(state: AppState,
                              wells_norm: list[str],
                              name_of_y_true: str,
                              mode: str = 'liq') -> dict[str: str, str: pd.DataFrame]:
    input_data = []
    for well_name_normal in wells_norm:
        well_input = prepare_single_df_for_ensemble(state, well_name_normal, name_of_y_true, mode)
        input_data.append({'wellname': well_name_normal,
                           'df': well_input})
    return input_data


def prepare_single_df_for_ensemble(state: AppState,
                                   well_name_normal: str,
                                   name_of_y_true: str,
                                   mode: str = 'liq') -> pd.DataFrame:
    models = list(state.statistics.keys())
    if 'ensemble' in models:
        models.remove('ensemble')
    dates_test = pd.date_range(state.was_date_test, state.was_date_end, freq='D').date
    input_df = pd.DataFrame(index=dates_test)
    for model in models:
        well_calculated_by_model = f'{well_name_normal}_{mode}_pred' in state.statistics[model]
        if well_calculated_by_model:
            all_values_are_nan = state.statistics[model][f'{well_name_normal}_{mode}_pred'].isna().all()
            if not all_values_are_nan:
                input_df[name_of_y_true] = state.statistics[model][f'{well_name_normal}_{mode}_true']
                input_df[model] = state.statistics[model][f'{well_name_normal}_{mode}_pred']
    return input_df


def extract_data_ensemble(ensemble_df: pd.DataFrame,
                          state: AppState,
                          well_name_normal: str,
                          mode: str = 'liq') -> None:
    if ensemble_df.empty:
        return
    dates = pd.date_range(state.was_date_start, state.was_date_end, freq='D').date
    if 'ensemble' not in state.statistics:
        state.statistics['ensemble'] = pd.DataFrame(index=dates)
    if mode == 'oil':
        state.statistics['ensemble'][f'{well_name_normal}_oil_true'] = ensemble_df['true']
        state.statistics['ensemble'][f'{well_name_normal}_oil_pred'] = ensemble_df['ensemble']
        state.ensemble_interval[f'{well_name_normal}_oil_upper'] = ensemble_df['interval_upper']
        state.ensemble_interval[f'{well_name_normal}_oil_lower'] = ensemble_df['interval_lower']
    elif mode == 'liq':
        state.statistics['ensemble'][f'{well_name_normal}_liq_true'] = ensemble_df['true']
        state.statistics['ensemble'][f'{well_name_normal}_liq_pred'] = ensemble_df['ensemble']
        state.ensemble_interval[f'{well_name_normal}_liq_upper'] = ensemble_df['interval_upper']
        state.ensemble_interval[f'{well_name_normal}_liq_lower'] = ensemble_df['interval_lower']


def make_models_stop_well(statistics: Dict[str, pd.DataFrame],
                          well_names: List[str]) -> None:
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


def cut_statistics_test_only(state: AppState) -> Tuple[Dict[str, pd.DataFrame], pd.date_range]:
    statistics_test_index = pd.date_range(state.was_date_test, state.was_date_end, freq='D')
    # обрезка данных по датам(индексу) ансамбля
    if state.was_calc_ensemble:
        statistics_test_index = pd.date_range(state.was_date_test_if_ensemble, state.was_date_end, freq='D')

    statistics_test_only = {}
    for key in state.statistics:
        statistics_test_only[key] = state.statistics[key].copy().reindex(statistics_test_index).fillna(0)
    return statistics_test_only, statistics_test_index


def add_fieldshops(fieldshops: dict) -> None:
    fieldshops_path = pathlib.Path.cwd() / 'tools_preprocessor' / 'data'
    fieldshops_name = fieldshops_path.glob("**")
    for fs_name in fieldshops_name:
        if pathlib.Path(fs_name / 'welllist.feather').exists():
            fieldshops_ceh = pd.read_feather(pathlib.Path(fs_name / 'welllist.feather'))
            fs_name = fs_name.name
        else:
            continue
        fieldshops_ceh = fieldshops_ceh.ceh.unique()
        fieldshops[fs_name] = list(fieldshops_ceh)