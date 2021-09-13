import streamlit as st
from ensemble.bayesian_model import BayesianModel
from ftor.calculator import Calculator as CalculatorFtor
from ftor.config import Config as ConfigFtor
from preprocessor import Preprocessor
from wolfram.calculator import Calculator as CalculatorWolfram
from wolfram.config import Config as ConfigWolfram


@st.cache(show_spinner=False)
def run_preprocessor(config):
    _preprocessor = Preprocessor(config)
    return _preprocessor


@st.cache
def calculate_ftor(
        preprocessor,
        well_name,
        constraints,
        config_ftor=ConfigFtor()
):
    ftor = CalculatorFtor(
        config_ftor,
        preprocessor.create_wells_ftor(
            [well_name],
            user_constraints_for_adap_period=constraints,
        )
    )
    return ftor


@st.cache
def calculate_wolfram(
        preprocessor,
        well_name,
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
        preprocessor.create_wells_wolfram([well_name]),
    )
    return wolfram


@st.cache
def calculate_ensemble(
        df,
        adaptation_days_number,
        name_of_y_true,
):
    bayesian_model = BayesianModel(
        df,
        adaptation_days_number=adaptation_days_number,
        name_of_y_true=name_of_y_true
    )
    return bayesian_model.result_test
