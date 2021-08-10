import datetime
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from ftor.calculator import Calculator as CalculatorFtor
from wolfram.calculator import Calculator as CalculatorWolfram
from wolfram.config import Config as ConfigWolfram

from config import Config
from plots import create_well_plot
from preprocessor import Preprocessor

st.set_page_config(layout="wide")  # Для отображения на всю ширину браузера
session = st.session_state

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
    with st.form("input_form"):  # Необходимо для запуска расчетов по кнопке
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

        # Каждая обертка form должна иметь submit_button
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

fig = create_well_plot(well_ftor, well_wolfram, date_test)

st.plotly_chart(fig, use_container_width=True)

with st.form('upload_CRM_form'):
    CRM_xlsx = st.file_uploader('Загрузить результаты CRM', type='xlsx')
    if CRM_xlsx is not None:
        df_CRM = pd.read_excel(CRM_xlsx, engine='openpyxl')
        if 'df_CRM' not in session:
            session['df_CRM'] = df_CRM
    st.form_submit_button(label='Отобразить')
