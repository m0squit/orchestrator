import datetime
import streamlit as st
from dateutil.relativedelta import relativedelta
from ftor.calculator import Calculator as CalculatorFtor
from wolfram.calculator import Calculator as CalculatorWolfram
from wolfram.config import Config

from config import Config
from preprocessor import Preprocessor


FIELDS_SHOPS = {
    'Валынтойское': ['ЦДНГ-12'],
    'Вынгаяхинское': ['ЦДНГ-10'],
    'Крайнее': ['ЦДНГ-4', 'ЦДНГ-2'],
    'Отдельное': ['ЦДНГ-1'],
    'Романовское': ['ЦДНГ-3'],
    'Холмогороское': ['ЦДНГ-1'],
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

st.sidebar.subheader('Общие настройки')
field_name = st.sidebar.selectbox(
    label='Месторождение',
    options=FIELDS_SHOPS.keys(),
)
shops = [
    st.sidebar.selectbox(
        label='Цех добычи',
        options=FIELDS_SHOPS[field_name],
    )
]
date_start = st.sidebar.date_input(
    label='Дата начала адаптации (с 00:00)',
    min_value=DATE_MIN,
    max_value=DATE_MAX - PERIOD_TRAIN_MIN - PERIOD_TEST_MIN,
    value=DATE_MIN,
    help="""
    Данная дата используется только для модели пьезопроводности.
    Адаптация модели ML проводится на всех доступных по скважине данных.
    """,
)
date_test = st.sidebar.date_input(
    label='Дата начала прогноза (с 00:00)',
    min_value=date_start + PERIOD_TRAIN_MIN,
    max_value=DATE_MAX - PERIOD_TEST_MIN,
    value=date_start + PERIOD_TRAIN_MIN,
)
date_end = st.sidebar.date_input(
    label='Дата конца прогноза (по 23:59)',
    min_value=date_test + PERIOD_TEST_MIN,
    max_value=DATE_MAX,
    value=date_test + PERIOD_TEST_MIN,
)
preprocessor = Preprocessor(
    Config(
        field_name,
        shops,
        date_start,
        date_test,
        date_end,
    )
)
well_name = st.sidebar.selectbox(
    label='Скважина',
    options=preprocessor.well_names,
)

with st.sidebar.beta_expander('Настройки модели ML'):
    estimator_name_group = ML_FULL_ABBR[
        st.selectbox(
            label='Модель на 1-ом уровне',
            options=ML_FULL_ABBR.keys(),
            index=1,  # LinearSVR
            help="""
            Данная модель использует для обучения только входные данные.
            Подробнее о моделях см. [sklearn](https://scikit-learn.org) и [xgboost](https://xgboost.readthedocs.io).
            """,
        )
    ]
    estimator_name_well = ML_FULL_ABBR[
        st.selectbox(
            label='Модель на 2-ом уровне',
            options=ML_FULL_ABBR.keys(),
            index=0,  # ElasticNet
            help="""
            Данная модель использует для обучения как входные данные, так и результаты работы модели 1-ого уровня.
            Подробнее о моделях см. [sklearn](https://scikit-learn.org) и [xgboost](https://xgboost.readthedocs.io).
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
