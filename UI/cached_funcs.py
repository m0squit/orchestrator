from datetime import date
from typing import List, Tuple, Dict

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from loguru import logger

# from frameworks_crm.class_CRM.calculator import Calculator as CalculatorCRM
# from frameworks_crm.class_CRM.config import ConfigCRM
# from frameworks_crm.class_Fedot.fedot_model import FedotModel
from frameworks_shelf_algo.class_Shelf.config import ConfigShelf
from frameworks_shelf_algo.class_Shelf.data_processor_shelf import DataProcessorShelf
from frameworks_shelf_algo.class_Shelf.calculator import CalculatorShelf
from frameworks_shelf_algo.class_Shelf.data_postprocessor_shelf import DataPostProcessorShelf
from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from frameworks_ftor.ftor.config import Config as ConfigFtor
from frameworks_wolfram.wolfram.calculator import Calculator as CalculatorWolfram
from frameworks_wolfram.wolfram.config import Config as ConfigWolfram
from models_ensemble.calculator import Calculator as CalculatorEnsemble
from models_ensemble.config import Config as ConfigEnsemble
from statistics_explorer.config import ConfigStatistics
from statistics_explorer.main import calculate_statistics
from tools_preprocessor.config import Config as ConfigPreprocessor
from tools_preprocessor.preprocessor import Preprocessor


@st.cache(show_spinner=False)
def run_preprocessor(config: ConfigPreprocessor) -> Preprocessor:
    _preprocessor = Preprocessor(config)
    return _preprocessor


@st.cache
def calculate_ftor(_preprocessor: Preprocessor,
                   well_names: List[int],
                   constraints: dict) -> CalculatorFtor:
    config_ftor = ConfigFtor()
    # Если пользователь задал границы\значение параметра, которым производится адаптация на
    # последние точки, то эти значения применяются и для самой адаптации на последние точки
    param_last_point = config_ftor.param_name_last_points_adaptation
    if param_last_point in constraints.keys():
        if type(constraints[param_last_point]) == dict:
            config_ftor.param_bounds_last_points_adaptation = constraints[param_last_point]['bounds']
        else:
            config_ftor.apply_last_points_adaptation = False

    ftor = CalculatorFtor(
        config_ftor,
        _preprocessor.create_wells_ftor(
            well_names,
            user_constraints_for_adap_period=constraints,
        ),
        logging=True
    )
    return ftor

@st.experimental_singleton
def calculate_shelf(oilfield: str,
                    shops: List[str],
                    wells_ois: List[int],
                    train_start: date,
                    train_end: date,
                    predict_start: date,
                    predict_end: date,
                    n_days_past: int,
                    n_days_calc_avg: int,
                    change_gtm_info: int) -> DataPostProcessorShelf:
    config_shelf = ConfigShelf(oilfield=oilfield,
                               shops=shops,
                               wells_ois=wells_ois,
                               train_start=train_start,
                               train_end=train_end,
                               predict_start=predict_start,
                               predict_end=predict_end,
                               n_days_past=n_days_past,
                               n_days_calc_avg=n_days_calc_avg)
    results_shelf = CalculatorShelf(config_shelf)
    # results_shelf = DataPostProcessorShelf(config_shelf)
    return results_shelf


@st.experimental_singleton
def calculate_wolfram(_preprocessor: Preprocessor,
                      well_names: List[int],
                      forecast_days_number: int,
                      estimator_name_group: str,
                      estimator_name_well: str,
                      is_deep_grid_search: bool,
                      window_sizes: List[int],
                      quantiles: List[float]) -> CalculatorWolfram:
    wolfram = CalculatorWolfram(
        ConfigWolfram(
            forecast_days_number,
            estimator_name_group,
            estimator_name_well,
            is_deep_grid_search,
            window_sizes,
            quantiles,
        ),
        _preprocessor.create_wells_wolfram(well_names),
    )
    return wolfram


@st.experimental_singleton
# def calculate_CRM(date_start_adapt: date,
#                   date_end_adapt: date,
#                   date_end_forecast: date,
#                   oilfield: str,
#                   calc_CRM: bool = True,
#                   calc_CRMIP: bool = False,
#                   grad_format_data: bool = True,
#                   influence_R: int = 1300,
#                   maxiter: int = 100,
#                   p_res: int = 220) -> CalculatorCRM or None:
#     config_CRM = ConfigCRM(date_start_adapt=date_start_adapt,
#                            date_end_adapt=date_end_adapt,
#                            date_end_forecast=date_end_forecast,
#                            calc_CRM=calc_CRM,
#                            calc_CRMIP=calc_CRMIP,
#                            grad_format_data=grad_format_data,
#                            oilfield=oilfield)
#     config_CRM.INFLUENCE_R = influence_R
#     config_CRM.options_SLSQP_CRM['maxiter'] = maxiter
#     config_CRM.p_res = p_res
#     try:
#         logger.info(f'CRM: start calculations')
#         calculator_CRM = CalculatorCRM(config_CRM)
#         logger.success(f'CRM: success')
#         return calculator_CRM
#     except Exception as exc:
#         logger.exception('CRM: FAIL', exc)
#         return None


# @st.experimental_singleton
# def calculate_fedot(oilfield: str,
#                     train_start: date,
#                     train_end: date,
#                     predict_start: date,
#                     predict_end: date,
#                     wells_to_calc: list[str],
#                     coeff: pd.DataFrame) -> FedotModel:
#     calculator_fedot = FedotModel(oilfield=oilfield,
#                                   train_start=train_start,
#                                   train_end=train_end,
#                                   predict_start=predict_start,
#                                   predict_end=predict_end,
#                                   wells_to_calc=wells_to_calc,
#                                   coeff=coeff)
#     return calculator_fedot


@st.experimental_singleton
def calculate_ensemble(input_data: list[dict],
                       adaptation_days_number: int,
                       interval_probability: float,
                       draws: int,
                       tune: int,
                       chains: int,
                       target_accept: float,
                       name_of_y_true: str) -> dict[str, pd.DataFrame]:
    calculator_ensemble = CalculatorEnsemble(
        ConfigEnsemble(adaptation_days_number=adaptation_days_number,
                       interval_probability=interval_probability,
                       draws=draws,
                       tune=tune,
                       chains=chains,
                       target_accept=target_accept,
                       name_of_y_true=name_of_y_true),
        input_data,
        logging=True
    )
    return calculator_ensemble.result_test


@st.experimental_memo
def calculate_statistics_plots(
        statistics: dict,
        field_name: str,
        date_start: date,
        date_end: date,
        well_names: tuple,
        use_abs: bool,
        exclude_wells: list,
        bin_size: int
) -> Tuple[Dict[str, go.Figure], ConfigStatistics]:
    config_stat = ConfigStatistics(
        oilfield=field_name,
        dates=pd.date_range(date_start, date_end, freq='D').date,
        well_names=well_names,
        use_abs=use_abs,
        bin_size=bin_size,
    )
    config_stat.exclude_wells(exclude_wells)
    analytics_plots = calculate_statistics(statistics, config_stat)
    return analytics_plots, config_stat
