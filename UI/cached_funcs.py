import pandas as pd
import streamlit as st
from models_ensemble.bayesian_model import BayesianModel
from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from frameworks_ftor.ftor.config import Config as ConfigFtor
from preprocessor import Preprocessor
from frameworks_wolfram.wolfram.calculator import Calculator as CalculatorWolfram
from frameworks_wolfram.wolfram.config import Config as ConfigWolfram


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
    # TODO: убрать в будущем: если пользователем задан P_init - меняем ConfigFtor
    config_ftor = ConfigFtor()
    if 'pressure_initial' in constraints.keys():
        if type(constraints['pressure_initial']) == dict:
            config_ftor.are_param_bounds_discrete = False
            config_ftor.param_bounds_last_points_adaptation = constraints['pressure_initial']['bounds']
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
    return bayesian_model.result_test
