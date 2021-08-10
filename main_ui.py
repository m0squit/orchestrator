import datetime
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from ftor.calculator import Calculator as CalculatorFtor
from plotly.subplots import make_subplots
from wolfram.calculator import Calculator as CalculatorWolfram
from wolfram.config import Config as ConfigWolfram

from config import Config
from preprocessor import Preprocessor


def compute_deviations(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    devs = abs(y_true - y_pred) / y_true
    return devs


st.set_page_config(layout="wide")   # Для отображения на всю ширину браузера

FIELDS_SHOPS = {
    'Валынтойское': ['ЦДНГ-12'],
    'Вынгаяхинское': ['ЦДНГ-10'],
    'Крайнее': ['ЦДНГ-4', 'ЦДНГ-2'],
    'Отдельное': ['ЦДНГ-1'],
    'Романовское': ['ЦДНГ-3'],
    'Холмогорское': ['ЦДHГ-1'],  # H - латинская
}

# Диапазон дат выгрузки sh таблицы
DATE_MIN = datetime.date(2018, 1, 1)
DATE_MAX = datetime.date(2021, 4, 30)

PERIOD_TRAIN_MIN = relativedelta(months=3)
PERIOD_TEST_MIN = relativedelta(months=1)

ML_FULL_ABBR = {
    'ElasticNet': 'ela',
    'LinearSVR': 'svr',
    'XGBoost': 'xgb',
}
YES_NO = {
    'Да': True,
    'Нет': False,
}

with st.sidebar:
    with st.form("input_form"):
        field_name = st.selectbox(
            label='Месторождение',
            options=FIELDS_SHOPS.keys(),
        )
        shops = [
            st.selectbox(
                label='Цех добычи',
                options=FIELDS_SHOPS[field_name],
            )
        ]
        date_start = st.date_input(
            label='Дата начала адаптации (с 00:00)',
            min_value=DATE_MIN,
            max_value=DATE_MAX - PERIOD_TRAIN_MIN - PERIOD_TEST_MIN,
            value=DATE_MIN,
            help="""
            Данная дата используется только для модели пьезопроводности.
            Адаптация модели ML проводится на всех доступных по скважине данных.
            """,
        )
        date_test = st.date_input(
            label='Дата начала прогноза (с 00:00)',
            min_value=date_start + PERIOD_TRAIN_MIN,
            max_value=DATE_MAX - PERIOD_TEST_MIN,
            value=date_start + PERIOD_TRAIN_MIN,
        )
        date_end = st.date_input(
            label='Дата конца прогноза (по 23:59)',
            min_value=date_test + PERIOD_TEST_MIN,
            max_value=DATE_MAX,
            value=date_test + PERIOD_TEST_MIN,
        )
        forecast_days_number = (date_end - date_test).days
        preprocessor = Preprocessor(
            Config(
                field_name,
                shops,
                date_start,
                date_test,
                date_end,
            )
        )
        well_name = st.selectbox(
            label='Скважина',
            options=preprocessor.well_names,
        )

        with st.expander('Настройки модели ML'):
            estimator_name_group = ML_FULL_ABBR[
                st.selectbox(
                    label='Модель на 1-ом уровне',
                    options=ML_FULL_ABBR.keys(),
                    index=1,  # LinearSVR
                    help="""
                    Данная модель использует для обучения только входные данные.
                    Подробнее о моделях см. [sklearn](https://scikit-learn.org) 
                    и [xgboost](https://xgboost.readthedocs.io).
                    """,
                )
            ]
            estimator_name_well = ML_FULL_ABBR[
                st.selectbox(
                    label='Модель на 2-ом уровне',
                    options=ML_FULL_ABBR.keys(),
                    index=0,  # ElasticNet
                    help="""
                    Данная модель использует для обучения как входные данные, 
                    так и результаты работы модели 1-ого уровня.
                    Подробнее о моделях см. [sklearn](https://scikit-learn.org) 
                    и [xgboost](https://xgboost.readthedocs.io).
                    """,
                )
            ]
            is_deep_grid_search = YES_NO[
                st.selectbox(
                    label='Cross Validation на 2-ом уровне',
                    options=YES_NO.keys(),
                    index=1,  # Нет
                    help="""
                    Данная процедура нацелена на предотвращение переобучения модели 2-ого уровня.
                    Подробнее см. [Cross-validation for time series](https://robjhyndman.com/hyndsight/tscv).
                    """,
                )
            ]
            window_sizes = [
                int(ws) for ws in st.text_input(
                    label='Размеры скользящего окна',
                    value='3 5 7 15 30',
                    max_chars=20,
                ).split()
            ]
            quantiles = [
                float(q) for q in st.text_input(
                    label='Квантили',
                    value='0.1 0.3',
                    max_chars=20,
                ).split()
            ]

        # Every form must have a submit button.
        st.form_submit_button(label='Запустить расчеты')

calculator_ftor = CalculatorFtor(
    preprocessor.create_wells_ftor([well_name])
)
calculator_wolfram = CalculatorWolfram(
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

well_ftor = calculator_ftor.wells[0]  # Т.к. считается только 1 скважина
well_wolfram = calculator_wolfram.wells[0]  # Т.к. считается только 1 скважина
res_ftor = well_ftor.results
res_wolfram = well_wolfram.results

# Фактические данные для визуализации извлекаются из wolfram, т.к. он использует для вычислений максимально возможный
# доступный ряд фактичесих данных.
name_rate_liq = 'q_жид'
name_rate_oil = 'q_неф'
name_pressure = 'p_заб'
name_dev_liq = 're_жид'
name_dev_oil = 're_неф'

df = well_wolfram.df
rates_liq_true = df[well_wolfram.NAME_RATE_LIQ]
rates_oil_true = df[well_wolfram.NAME_RATE_OIL]
pressure = df[well_wolfram.NAME_PRESSURE]

# Жидкость
# Полный ряд (train + test)
rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
rates_liq_wolfram = pd.concat(objs=[res_wolfram.rates_liq_train, res_wolfram.rates_liq_test])
# test
rates_liq_test_ftor = res_ftor.rates_liq_test
rates_liq_test_wolfram = res_wolfram.rates_liq_test
rates_liq_test_true = rates_liq_true.loc[rates_liq_test_wolfram.index]
# devs
devs_liq_ftor = compute_deviations(rates_liq_test_true, rates_liq_test_ftor)
devs_liq_wolfram = compute_deviations(rates_liq_test_true, rates_liq_test_wolfram)

# Нефть
# Полный ряд (train + test)
rates_oil_wolfram = pd.concat(objs=[res_wolfram.rates_oil_train, res_wolfram.rates_oil_test])  # Только нефть
# test
rates_oil_test_ftor = res_ftor.rates_oil_test
rates_oil_test_wolfram = res_wolfram.rates_oil_test
rates_oil_test_true = rates_oil_true.loc[rates_oil_test_wolfram.index]
# devs
devs_oil_ftor = compute_deviations(rates_liq_test_true, rates_oil_test_ftor)
devs_oil_wolfram = compute_deviations(rates_oil_test_true, rates_oil_test_wolfram)

fig = make_subplots(
    rows=3,
    cols=2,
    shared_xaxes=True,
    column_width=[0.7, 0.3],
    row_heights=[0.4, 0.4, 0.2],
    vertical_spacing=0.02,
    horizontal_spacing=0.02,
    column_titles=[
        'Адаптация и прогноз',
        'Прогноз',
    ],
    figure=go.Figure(
        layout=go.Layout(
            font=dict(size=10),
            hovermode='x',
            template='seaborn',
            height=650,
            width=1000,
        ),
    ),
)

m = 'markers'
ml = 'markers+lines'
mark = dict(size=3)
line = dict(width=1)

# 1, 1 Полный ряд (train + test)
fig.add_trace(
    go.Scatter(
        name=name_rate_liq + '_факт',
        x=rates_liq_true.index,
        y=rates_liq_true,
        mode=m, marker=mark,
    ), row=1, col=1)
fig.add_trace(
    go.Scatter(
        name=name_rate_liq + '_пьезо',
        x=rates_liq_ftor.index,
        y=rates_liq_ftor,
        mode=ml, marker=mark,
    ), row=1, col=1)
fig.add_trace(
    go.Scatter(
        name=name_rate_liq + '_ML',
        x=rates_liq_wolfram.index,
        y=rates_liq_wolfram,
        mode=ml, marker=mark,
    ), row=1, col=1)

# 1, 2 test
fig.add_trace(
    go.Scatter(
        name=name_rate_liq + '_факт',
        x=rates_liq_test_true.index,
        y=rates_liq_test_true,
        mode=m, marker=mark,
    ), row=1, col=2)
fig.add_trace(
    go.Scatter(
        name=name_rate_liq + '_пьезо',
        x=rates_liq_test_ftor.index,
        y=rates_liq_test_ftor,
        mode=ml, marker=mark,
    ), row=1, col=2)
fig.add_trace(
    go.Scatter(
        name=name_rate_liq + '_ML',
        x=rates_liq_test_wolfram.index,
        y=rates_liq_test_wolfram,
        mode=ml, marker=mark,
    ), row=1, col=2)

# 2, 1 Полный ряд (train + test)
fig.add_trace(
    go.Scatter(
        name=name_rate_oil + '_факт',
        x=rates_oil_true.index,
        y=rates_oil_true,
        mode=m, marker=mark,
    ), row=2, col=1)
fig.add_trace(
    go.Scatter(
        name=name_rate_oil + '_ML',
        x=rates_oil_wolfram.index,
        y=rates_oil_wolfram,
        mode=ml, marker=mark,
    ), row=2, col=1)

# 2, 2 test
fig.add_trace(
    go.Scatter(
        name=name_rate_oil + '_факт',
        x=rates_oil_test_true.index,
        y=rates_oil_test_true,
        mode=m, marker=mark,
    ), row=2, col=2)
fig.add_trace(
    go.Scatter(
        name=name_rate_oil + '_пьезо',
        x=rates_oil_test_ftor.index,
        y=rates_oil_test_ftor,
        mode=ml, marker=mark,
    ), row=2, col=2)
fig.add_trace(
    go.Scatter(
        name=name_rate_oil + '_ML',
        x=rates_oil_test_wolfram.index,
        y=rates_oil_test_wolfram,
        mode=ml, marker=mark,
    ), row=2, col=2)

# 3, 1 Полный ряд (train + test)
fig.add_trace(
    go.Scatter(
        name=name_pressure + '_факт',
        x=pressure.index,
        y=pressure,
        mode=m, marker=mark,
    ), row=3, col=1)

# 3, 2 test
fig.add_trace(
    go.Scatter(
        name=name_dev_liq + '_пьезо',
        x=devs_liq_ftor.index,
        y=devs_liq_ftor,
        mode=ml, marker=mark,
    ), row=3, col=2)
fig.add_trace(
    go.Scatter(
        name=name_dev_liq + '_ML',
        x=devs_liq_wolfram.index,
        y=devs_liq_wolfram,
        mode=ml, marker=mark,
    ), row=3, col=2)
fig.add_trace(
    go.Scatter(
        name=name_dev_oil + '_пьезо',
        x=devs_oil_ftor.index,
        y=devs_oil_ftor,
        mode=ml, marker=mark,
    ), row=3, col=2)
fig.add_trace(
    go.Scatter(
        name=name_dev_oil + '_ML',
        x=devs_oil_wolfram.index,
        y=devs_oil_wolfram,
        mode=ml, marker=mark,
    ), row=3, col=2)


fig.add_vline(row=1, col=1, x=date_test, line_width=2, line_dash='dash')
fig.add_vline(row=2, col=1, x=date_test, line_width=2, line_dash='dash')
fig.add_vline(row=3, col=1, x=date_test, line_width=2, line_dash='dash')


st.plotly_chart(fig, use_container_width=True)
