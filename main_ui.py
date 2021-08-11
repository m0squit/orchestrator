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


def keep_well_name():
    # Без этой функции при клике submit_button сбрасывается номер скважины в selectbox'e
    session.well_name = st.session_state.well_name


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

# Initialize session values
if 'date_start' not in session:
    session.date_start = datetime.date(2018, 3, 1)
    session.date_test = datetime.date(2019, 3, 1)
    session.date_end = datetime.date(2019, 5, 31)

with st.sidebar:
    with st.form("input_form"):  # Необходимо для запуска расчетов по кнопке "Запустить расчеты"
        field_name = st.selectbox(
            label='Месторождение',
            options=FIELDS_SHOPS.keys(),
            key='field_name'
        )
        shops = [
            st.selectbox(
                label='Цех добычи',
                options=FIELDS_SHOPS[field_name],
                key='shops'
            )
        ]
        date_start = st.date_input(
            label='Дата начала адаптации (с 00:00)',
            min_value=DATE_MIN,
            max_value=DATE_MAX - PERIOD_TRAIN_MIN - PERIOD_TEST_MIN,
            key='date_start',
            help="""
            Данная дата используется только для модели пьезопроводности.
            Адаптация модели ML проводится на всех доступных по скважине данных.
            """,
        )
        date_test = st.date_input(
            label='Дата начала прогноза (с 00:00)',
            min_value=date_start + PERIOD_TRAIN_MIN,
            max_value=DATE_MAX - PERIOD_TEST_MIN,
            key='date_test',
        )
        date_end = st.date_input(
            label='Дата конца прогноза (по 23:59)',
            min_value=date_test + PERIOD_TEST_MIN,
            max_value=DATE_MAX,
            key='date_end',
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
            key='well_name'
        )

        CRM_xlsx = st.file_uploader('Загрузить результаты CRM', type='xlsx')
        if CRM_xlsx is not None:
            df_CRM = pd.read_excel(CRM_xlsx, index_col=0, engine='openpyxl')
            if 'df_CRM' not in session:
                session['df_CRM'] = df_CRM

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
        submit = st.form_submit_button(label='Запустить расчеты', on_click=keep_well_name)

if not submit:
    st.info('Выберите настройки и нажмите кнопку "Запустить расчеты"')
else:
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

    # Фактические данные для визуализации извлекаются из wolfram, т.к. он использует для вычислений максимально
    # возможный доступный ряд фактичесих данных.
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

    # Нефть
    # Полный ряд (train + test)
    rates_oil_wolfram = pd.concat(objs=[res_wolfram.rates_oil_train, res_wolfram.rates_oil_test])  # Только нефть
    # test
    rates_oil_test_ftor = res_ftor.rates_oil_test
    rates_oil_test_wolfram = res_wolfram.rates_oil_test
    rates_oil_test_true = rates_oil_true.loc[rates_oil_test_wolfram.index]

    df_draw_liq = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))
    df_draw_liq['ftor'] = rates_liq_ftor
    df_draw_liq['wolfram'] = rates_liq_wolfram
    df_draw_liq['true'] = rates_liq_true

    df_draw_oil = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))
    df_draw_oil['ftor'] = rates_oil_test_ftor
    df_draw_oil['wolfram'] = rates_oil_wolfram
    df_draw_oil['true'] = rates_oil_true
    if 'df_CRM' in session:
        if well_name in session['df_CRM'].columns:
            df_draw_oil['CRM'] = session['df_CRM'][well_name]

    fig = create_well_plot(df_draw_liq, df_draw_oil, pressure, date_test, well_name)

    st.plotly_chart(fig, use_container_width=True)
