import datetime
import pandas as pd
import streamlit as st
from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from frameworks_ftor.ftor.config import Config as ConfigFtor
from frameworks_wolfram.wolfram.calculator import Calculator as CalculatorWolfram
from frameworks_wolfram.wolfram.config import Config as ConfigWolfram
from models_ensemble.bayesian_model import BayesianModel
from tools_preprocessor.preprocessor import Preprocessor
from statistics_explorer.config import ConfigStatistics
from statistics_explorer.main import calculate_statistics


@st.cache(show_spinner=False)
def run_preprocessor(config):
    _preprocessor = Preprocessor(config)
    return _preprocessor


@st.experimental_singleton
def calculate_ftor(
        _preprocessor,
        well_names,
        constraints,
):
    config_ftor = ConfigFtor()
    # Если пользователь задал границы\значение параметра, которым производится адаптация на
    # последние точки, то эти значения применяются и для самой адаптации на последние точки
    param_last_point = config_ftor.param_name_last_points_adaptation
    if param_last_point in constraints.keys():
        if type(constraints[param_last_point]) == dict:
            config_ftor.are_param_bounds_discrete = False
            config_ftor.param_bounds_last_points_adaptation = constraints[param_last_point]['bounds']
        else:
            config_ftor.apply_last_points_adaptation = False

    ftor = CalculatorFtor(
        config_ftor,
        _preprocessor.create_wells_ftor(
            well_names,
            user_constraints_for_adap_period=constraints,
        ),
        df_hypotheses=pd.DataFrame()
    )
    return ftor


@st.experimental_singleton
def calculate_wolfram(
        _preprocessor,
        well_names,
        forecast_days_number,
        estimator_name_group,
        estimator_name_well,
        is_deep_grid_search,
        window_sizes,
        quantiles,
):
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
def calculate_ensemble(
        df,
        adaptation_days_number,
        interval_probability,
        draws,
        tune,
        chains,
        target_accept,
        name_of_y_true,
):
    result = pd.DataFrame()
    try:
        bayesian_model = BayesianModel(
            df,
            adaptation_days_number=adaptation_days_number,
            interval_probability=interval_probability,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            name_of_y_true=name_of_y_true
        )
        result = bayesian_model.result_test
    except:
        print('Ошибка расчета ансамбля')
        pass
    return result


@st.experimental_memo
def calculate_statistics_plots(statistics: dict,
                               field_name: str,
                               date_start: datetime.date,
                               date_end: datetime.date,
                               well_names: tuple,
                               use_abs: bool,
                               exclude_wells: list,
                               bin_size: int):
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
