import datetime
import io
import pandas as pd
import streamlit as st
import UI.pages.models_settings as models_settings

from config import Config
from UI.cached_funcs import calculate_ftor, calculate_wolfram, calculate_ensemble, run_preprocessor
from UI.plots import create_well_plot
from UI.config import FIELDS_SHOPS, DATE_MIN, DATE_MAX, PERIOD_TEST_MIN, \
    PERIOD_TRAIN_MIN, FTOR_DECODE

st.set_page_config(layout="wide")  # Для отображения на всю ширину браузера
session = st.session_state


def extract_data_ftor(_calculator_ftor, _df_liq, _df_oil):
    well_ftor = _calculator_ftor.wells[0]  # Т.к. считается только 1 скважина
    res_ftor = well_ftor.results
    session.adap_and_fixed_params = res_ftor.adap_and_fixed_params

    # Жидкость. Полный ряд (train + test)
    rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
    rates_liq_ftor = pd.to_numeric(rates_liq_ftor)

    # test
    rates_oil_test_ftor = res_ftor.rates_oil_test
    rates_oil_test_ftor = pd.to_numeric(rates_oil_test_ftor)

    _df_liq['ftor'] = rates_liq_ftor
    _df_oil['ftor'] = rates_oil_test_ftor


def extract_data_wolfram(_calculator_wolfram, _df_liq, _df_oil, _pressure):
    well_wolfram = _calculator_wolfram.wells[0]  # Т.к. считается только 1 скважина
    res_wolfram = well_wolfram.results

    # Фактические данные (вторично) извлекаются из wolfram, т.к. он использует для вычислений максимально
    # возможный доступный ряд фактичесих данных.
    df_true = well_wolfram.df
    rates_liq_true = df_true[well_wolfram.NAME_RATE_LIQ]
    rates_oil_true = df_true[well_wolfram.NAME_RATE_OIL]
    bh_pressure = df_true[well_wolfram.NAME_PRESSURE]
    _pressure = bh_pressure

    # Жидкость. Полный ряд (train + test)
    rates_liq_wolfram = pd.concat(objs=[res_wolfram.rates_liq_train, res_wolfram.rates_liq_test])

    # Нефть. Полный ряд (train + test)
    rates_oil_wolfram = pd.concat(objs=[res_wolfram.rates_oil_train, res_wolfram.rates_oil_test])

    _df_liq['wolfram'] = rates_liq_wolfram
    _df_liq['true'] = rates_liq_true

    _df_oil['wolfram'] = rates_oil_wolfram
    _df_oil['true'] = rates_oil_true


def convert_to_readable(res: dict):
    if 'boundary_code' in res.keys():
        # Расшифровка типа границ и типа скважины
        res['boundary_code'] = FTOR_DECODE['boundary_code'][res['boundary_code']]
        res['kind_code'] = FTOR_DECODE['kind_code'][res['kind_code']]
        # Расшифровка названий параметров адаптации
        for key in FTOR_DECODE.keys():
            if key in res.keys():
                res[FTOR_DECODE[key]['label']] = res.pop(key)
    return res


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
    session.fig = None
    session.constraints = {}

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
    well_names_parsed = parse_well_names(preprocessor.well_names)
    well_name_parsed = st.selectbox(
        label='Скважина',
        options=well_names_parsed.keys(),
        key='well_name_parsed'
    )

    CRM_xlsx = st.file_uploader('Загрузить прогноз CRM по нефти', type='xlsx')
    if CRM_xlsx is not None:
        df_CRM = pd.read_excel(CRM_xlsx, index_col=0, engine='openpyxl')
        session['df_CRM'] = df_CRM

    submit = st.button(label='Запустить расчеты')

cols = st.columns(4)
cols[0].button(
    label='Настройки моделей',
    key='models_settings',
)
cols[1].button(
    label='page1',
    key='page1'
)
cols[2].button(
    label='page2',
    key='page2'
)
cols[3].button(
    label='page3',
    key='page3'
)

if session.models_settings:
    models_settings.show()

if submit:
    # Инициализация данных для визуализации
    df_draw_liq = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))
    df_draw_oil = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))

    # Фактические данные для визуализации
    well = preprocessor.create_wells_ftor([well_name])
    df = well[0].df_chess
    events = df['Мероприятие']
    df_draw_liq['true'] = df['Дебит жидкости']
    df_draw_oil['true'] = df['Дебит нефти']
    pressure = df['Давление забойное']

    if is_calc_ftor:
        calculator_ftor = calculate_ftor(preprocessor, well_name_ois, session.constraints)
        extract_data_ftor(calculator_ftor, df_draw_liq, df_draw_oil)
    if is_calc_wolfram:
        calculator_wolfram = calculate_wolfram(
            preprocessor,
            well_name,
            forecast_days_number,
            session.estimator_name_group,
            session.estimator_name_well,
            session.is_deep_grid_search,
            session.window_sizes,
            session.quantiles,
        )
        extract_data_wolfram(calculator_wolfram, df_draw_liq, df_draw_oil, pressure)

    if 'df_CRM' in session:
        if well_name in session['df_CRM'].columns:
            df_draw_oil['CRM'] = session['df_CRM'][well_name]

    df_draw_ensemble = pd.DataFrame()
    if is_calc_ensemble and (is_calc_ftor or is_calc_wolfram):
        try:
            df_draw_ensemble = calculate_ensemble(
                df_draw_oil[date_test:],
                adaptation_days_number=(date_end - date_test).days // 4,
                name_of_y_true='true'
            )
        except:
            st.error('Ошибка при расчете ансамбля.')

    session.fig = create_well_plot(
        df_draw_liq,
        df_draw_oil,
        df_draw_ensemble,
        pressure,
        date_test,
        events,
        well_name,
    )

    # Подготовка данных к выгрузке
    session.buffer = io.BytesIO()
    with pd.ExcelWriter(session.buffer) as writer:
        df_draw_liq.to_excel(writer, sheet_name='Дебит жидкости')
        df_draw_oil.to_excel(writer, sheet_name='Дебит нефти')
        df_draw_ensemble.to_excel(writer, sheet_name='Дебит нефти ансамбль')
        pressure.to_excel(writer, sheet_name='Забойное давление')
        events.to_excel(writer, sheet_name='Мероприятие')

if session.fig is not None:
    # Построение графика
    st.plotly_chart(session.fig, use_container_width=True)

    # Вывод параметров адаптации модели пьезопроводности
    # TODO: (возможно) могут выводиться значения параметров от предыдущих расчетов, если нынешние упали с ошибкой
    if is_calc_ftor and 'adap_and_fixed_params' in session:
        result = session.adap_and_fixed_params[0].copy()
        result = convert_to_readable(result)
        st.write('Результаты адаптации модели пьезопроводности:', result)

    # Кнопка экспорта результатов
    st.download_button(
        label="Экспорт результатов",
        data=session.buffer,
        file_name=f'{well_name}_data.xlsx',
        mime='text/csv',
    )
else:
    st.info('Выберите настройки и нажмите кнопку "Запустить расчеты"')
