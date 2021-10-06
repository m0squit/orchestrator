import datetime
import pandas as pd
import streamlit as st
import UI.pages.models_settings
import UI.pages.wells_map
import UI.pages.analytics
import UI.pages.specific_well

from config import Config
from UI.cached_funcs import calculate_ftor, calculate_wolfram, calculate_ensemble, run_preprocessor
from UI.plots import create_well_plot
from UI.config import FIELDS_SHOPS, DATE_MIN, DATE_MAX, PERIOD_TEST_MIN, \
    PERIOD_TRAIN_MIN

st.set_page_config(layout="wide")  # Для отображения на всю ширину браузера
session = st.session_state


def extract_data_ftor(_calculator_ftor, df_liq, df_oil):
    for well_ftor in _calculator_ftor.wells:
        _well_name = well_ftor.well_name
        res_ftor = well_ftor.results
        session.adapt_params[_well_name] = res_ftor.adap_and_fixed_params

        # Жидкость. Полный ряд (train + test)
        rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
        rates_liq_ftor = pd.to_numeric(rates_liq_ftor)

        # test
        rates_oil_test_ftor = res_ftor.rates_oil_test
        rates_oil_test_ftor = pd.to_numeric(rates_oil_test_ftor)

        df_liq[_well_name]['ftor'] = rates_liq_ftor
        df_oil[_well_name]['ftor'] = rates_oil_test_ftor


def extract_data_wolfram(_calculator_wolfram, df_liq, df_oil, pressure):
    for _well_wolfram in _calculator_wolfram.wells:
        _well_name = _well_wolfram.well_name
        res_wolfram = _well_wolfram.results

        # Фактические данные (вторично) извлекаются из wolfram, т.к. он использует для вычислений максимально
        # возможный доступный ряд фактичесих данных.
        df_true = _well_wolfram.df
        rates_liq_true = df_true[_well_wolfram.NAME_RATE_LIQ]
        rates_oil_true = df_true[_well_wolfram.NAME_RATE_OIL]
        bh_pressure = df_true[_well_wolfram.NAME_PRESSURE]
        # Жидкость. Полный ряд (train + test)
        rates_liq_wolfram = pd.concat(objs=[res_wolfram.rates_liq_train, res_wolfram.rates_liq_test])
        # Нефть. Полный ряд (train + test)
        rates_oil_wolfram = pd.concat(objs=[res_wolfram.rates_oil_train, res_wolfram.rates_oil_test])

        df_liq[_well_name]['wolfram'] = rates_liq_wolfram
        df_liq[_well_name]['true'] = rates_liq_true
        df_oil[_well_name]['wolfram'] = rates_oil_wolfram
        df_oil[_well_name]['true'] = rates_oil_true
        pressure[_well_name] = bh_pressure


def parse_well_names(well_names_ois):
    welllist = pd.read_feather(f'data/{field_name}/welllist.feather')
    well_names = {}
    for name_ois in well_names_ois:
        well_name = welllist[welllist.ois == name_ois]
        well_name = well_name[well_name.npath == 0]
        well_name = well_name.at[well_name.index[0], 'num']
        well_names[well_name] = name_ois
    return well_names


# Инициализация значений сессии st.session_state
if 'date_start' not in session:
    # TODO: изменить даты на DATE_MIN
    session.date_start = datetime.date(2018, 12, 1)
    session.date_test = datetime.date(2019, 3, 1)
    session.date_end = datetime.date(2019, 5, 30)
    session.constraints = {}
    session.adapt_params = {}

    session.df_draw_liq = {}
    session.df_draw_oil = {}
    session.df_draw_ensemble = {}
    session.pressure = {}
    session.events = {}
    session.fig = {}

    session.estimator_name_group = 'svr'
    session.estimator_name_well = 'ela'
    session.is_deep_grid_search = False
    session.window_sizes = [3, 5, 7, 15, 30]
    session.quantiles = [0.1, 0.3]

with st.sidebar:
    is_calc_ftor = st.checkbox(
        label='Считать модель пьезопр-ти',
        value=True,
        key='is_calc_ftor',
    )
    is_calc_wolfram = st.checkbox(
        label='Считать модель ML',
        value=True,
        key='is_calc_wolfram',
    )
    is_calc_ensemble = st.checkbox(
        label='Считать ансамбль моделей',
        value=True,
        key='is_calc_ensemble',
        help='Ансамбль возможно рассчитать, если рассчитана хотя бы одна модель.'
    )
    field_name = st.selectbox(
        label='Месторождение',
        options=FIELDS_SHOPS.keys(),
        key='field_name'
    )
    # shops = [
    #     st.selectbox(
    #         label='Цех добычи',
    #         options=FIELDS_SHOPS[field_name],
    #         key='shops'
    #     )
    # ]
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
    config = Config(
        field_name,
        FIELDS_SHOPS[field_name],
        date_start,
        date_test,
        date_end,
    )
    preprocessor = run_preprocessor(config)
    session.well_names_parsed = parse_well_names(preprocessor.well_names)
    wells_to_calc = st.multiselect(
        label='Скважина',
        options=session.well_names_parsed.keys(),
        key='wells_to_calc'
    )

    CRM_xlsx = st.file_uploader('Загрузить прогноз CRM по нефти', type='xlsx')
    if CRM_xlsx is not None:
        df_CRM = pd.read_excel(CRM_xlsx, index_col=0, engine='openpyxl')
        session['df_CRM'] = df_CRM

    submit = st.button(label='Запустить расчеты')


if submit:
    session.selected_wells = wells_to_calc.copy()
    for well in preprocessor.create_wells_ftor(well_names_ois):
        # Инициализация данных для визуализации
        _well_name = well.well_name
        session.df_draw_liq[_well_name] = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))
        session.df_draw_oil[_well_name] = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))

        # Фактические данные для визуализации
        df = well.df_chess
        session.events[_well_name] = df['Мероприятие']
        session.df_draw_liq[_well_name]['true'] = df['Дебит жидкости']
        session.df_draw_oil[_well_name]['true'] = df['Дебит нефти']
        session.pressure[_well_name] = df['Давление забойное']

        # Данные CRM для визуализации
        if 'df_CRM' in session:
            for _well_name in well_names_ois:
                if _well_name in session['df_CRM'].columns:
                    session.df_draw_oil[_well_name]['CRM'] = session.df_CRM[_well_name]

    if is_calc_ftor:
        calculator_ftor = calculate_ftor(preprocessor, well_names_ois, session.constraints)
        extract_data_ftor(calculator_ftor, session.df_draw_liq, session.df_draw_oil)
    if is_calc_wolfram:
        calculator_wolfram = calculate_wolfram(
            preprocessor,
            well_names_ois,
            forecast_days_number,
            session.estimator_name_group,
            session.estimator_name_well,
            session.is_deep_grid_search,
            session.window_sizes,
            session.quantiles,
        )
        extract_data_wolfram(calculator_wolfram, session.df_draw_liq, session.df_draw_oil, session.pressure)

    for _well_name in well_names_ois:
        session.df_draw_ensemble[_well_name] = pd.DataFrame()
        if is_calc_ensemble and (is_calc_ftor or is_calc_wolfram):
            try:
                session.df_draw_ensemble[_well_name] = calculate_ensemble(
                    session.df_draw_oil[_well_name][date_test:],
                    adaptation_days_number=(date_end - date_test).days // 4,
                    name_of_y_true='true'
                )
            except:
                st.error('Ошибка при расчете ансамбля.')

        session.fig[_well_name] = create_well_plot(
            session.df_draw_liq[_well_name],
            session.df_draw_oil[_well_name],
            session.df_draw_ensemble[_well_name],
            session.pressure[_well_name],
            date_test,
            session.events[_well_name],
            _well_name,
        )

PAGES = {
    "Настройки моделей": UI.pages.models_settings,
    "Карта скважин": UI.pages.wells_map,
    "Аналитика": UI.pages.analytics,
    "Скважина": UI.pages.specific_well,
}
selection = st.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.show()

