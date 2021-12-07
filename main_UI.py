import datetime
import streamlit as st

import UI.pages.analytics
import UI.pages.models_settings
import UI.pages.resume_app
import UI.pages.specific_well
import UI.pages.wells_map
from config import Config as ConfigPreprocessor
from UI.app_state import AppState
from UI.cached_funcs import calculate_ftor, calculate_wolfram, calculate_ensemble, run_preprocessor
from UI.config import FIELDS_SHOPS, DATE_MIN, DATE_MAX, DEFAULT_FTOR_BOUNDS
from UI.data_processor import *


def initialize_session(session):
    session.state = AppState()
    # Ftor model
    session.constraints = {}
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        session[f'{param_name}_is_adapt'] = True
        session[f'{param_name}_lower'] = param_dict['lower_val']
        session[f'{param_name}_default'] = param_dict['default_val']
        session[f'{param_name}_upper'] = param_dict['upper_val']
    # ML model
    session.estimator_name_group = 'xgb'
    session.estimator_name_well = 'svr'
    session.is_deep_grid_search = False
    session.quantiles = [0.1, 0.3]
    session.window_sizes = [3, 5, 7, 15, 30]
    # Ensemble model
    session.ensemble_adapt_period = 28
    session.interval_probability = 0.9
    session.draws = 300
    session.tune = 200
    session.chains = 1
    session.target_accept = 0.95


def parse_well_names(well_names_ois):
    # Функция сопоставляет имена скважин OIS и (ГРАД?)
    welllist = pd.read_feather(f'data/{field_name}/welllist.feather')
    wellnames_key_normal = {}
    wellnames_key_ois = {}
    for name_ois in well_names_ois:
        well_name_norm = welllist[welllist.ois == name_ois]
        well_name_norm = well_name_norm[well_name_norm.npath == 0]
        well_name_norm = well_name_norm.at[well_name_norm.index[0], 'num']
        wellnames_key_normal[well_name_norm] = name_ois
        wellnames_key_ois[name_ois] = well_name_norm
    return wellnames_key_normal, wellnames_key_ois


def get_current_state(state: AppState, session: st.session_state) -> None:
    # Функция сохраняет состояние программы с выбранным набором параметров
    state['adapt_params'] = {}
    state['buffer'] = None
    state['ensemble_interval'] = pd.DataFrame()
    state['exclude_wells'] = []
    state['statistics'] = {}
    state['statistics_test_only'] = {}
    state['selected_wells_norm'] = wells_to_calc.copy()
    state['selected_wells_ois'] = selected_wells_ois.copy()
    state['was_config'] = config
    state['was_calc_ftor'] = is_calc_ftor
    state['was_calc_wolfram'] = is_calc_wolfram
    state['was_calc_ensemble'] = is_calc_ensemble
    state['was_date_start'] = date_start
    state['was_date_test'] = date_test
    state['was_date_test_if_ensemble'] = date_test + datetime.timedelta(days=session.ensemble_adapt_period)
    state['was_date_end'] = date_end
    state['wellnames_key_normal'] = wellnames_key_normal.copy()
    state['wellnames_key_ois'] = wellnames_key_ois.copy()


st.set_page_config(
    page_title='КСП',
    layout="wide"  # Для отображения на всю ширину браузера
)
# Инициализация значений сессии st.session_state
session = st.session_state
if 'date_start' not in session:
    initialize_session(session)

PAGES = {
    "Настройки моделей": UI.pages.models_settings,
    "Карта скважин": UI.pages.wells_map,
    "Аналитика": UI.pages.analytics,
    "Скважина": UI.pages.specific_well,
    "Импорт/экспорт расчетов": UI.pages.resume_app,
}

# Реализация интерфейса UI
with st.sidebar:
    selection = st.radio("", list(PAGES.keys()))
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
        key='field_name',
    )
    date_start = st.date_input(
        label='Дата начала адаптации (с 00:00)',
        min_value=DATE_MIN,
        value=datetime.date(2018, 12, 1),
        max_value=DATE_MAX,
        key='date_start',
        help="""
        Данная дата используется только для модели пьезопроводности.
        Адаптация модели ML проводится на всех доступных по скважине данных.
        """,
    )
    date_test = st.date_input(
        label='Дата начала прогноза (с 00:00)',
        min_value=DATE_MIN,
        value=datetime.date(2019, 3, 1),
        max_value=DATE_MAX,
        key='date_test',
    )
    date_end = st.date_input(
        label='Дата конца прогноза (по 23:59)',
        min_value=DATE_MIN,
        value=datetime.date(2019, 5, 30),
        max_value=DATE_MAX,
        key='date_end',
    )
    adaptation_days_number = (date_test - date_start).days
    forecast_days_number = (date_end - date_test).days
    config = ConfigPreprocessor(field_name, FIELDS_SHOPS[field_name],
                                date_start, date_test, date_end,)
    preprocessor = run_preprocessor(config)
    wellnames_key_normal, wellnames_key_ois = parse_well_names(preprocessor.well_names)
    wells_to_calc = st.multiselect(label='Скважина',
                                   options=['Все скважины'] + list(wellnames_key_normal.keys()),
                                   key='wells_to_calc')
    if 'Все скважины' in wells_to_calc:
        wells_to_calc = list(wellnames_key_normal.keys())
    selected_wells_ois = [wellnames_key_normal[well_name_] for well_name_ in wells_to_calc]

    CRM_xlsx = st.file_uploader('Загрузить прогноз CRM по нефти', type='xlsx')
    submit = st.button(label='Запустить расчеты')

if submit and wells_to_calc:
    session.state = AppState()
    get_current_state(session.state, session)
    at_least_one_model = is_calc_ftor or is_calc_wolfram or CRM_xlsx is not None
    if is_calc_ftor:
        calculator_ftor = calculate_ftor(preprocessor, selected_wells_ois, session.constraints)
        extract_data_ftor(calculator_ftor, session.state)
    if is_calc_wolfram:
        calculator_wolfram = calculate_wolfram(preprocessor,
                                               selected_wells_ois,
                                               forecast_days_number,
                                               session.estimator_name_group,
                                               session.estimator_name_well,
                                               session.is_deep_grid_search,
                                               session.window_sizes,
                                               session.quantiles)
        extract_data_wolfram(calculator_wolfram, session.state)
        convert_tones_to_m3_for_wolfram(session.state, preprocessor.create_wells_ftor(selected_wells_ois))
    if CRM_xlsx is None:
        session.pop('df_CRM', None)
    else:
        session['df_CRM'] = pd.read_excel(CRM_xlsx, index_col=0, engine='openpyxl')
        extract_data_CRM(session['df_CRM'], session.state, preprocessor.create_wells_wolfram(selected_wells_ois))
    if at_least_one_model:
        make_models_stop_well(session.state['statistics'], session.state['selected_wells_norm'])
    if at_least_one_model and is_calc_ensemble:
        name_of_y_true = 'true'
        for ind, well_name_normal in enumerate(wells_to_calc):
            print(f'\nWell {ind + 1} out of {len(wells_to_calc)}\n')
            input_df = prepare_df_for_ensemble(session.state, well_name_normal, name_of_y_true)
            ensemble_result = calculate_ensemble(
                input_df,
                adaptation_days_number=session.ensemble_adapt_period,
                interval_probability=session.interval_probability,
                draws=session.draws,
                tune=session.tune,
                chains=session.chains,
                target_accept=session.target_accept,
                name_of_y_true=name_of_y_true
            )
            if not ensemble_result.empty:
                extract_data_ensemble(ensemble_result, session.state, well_name_normal)
    if at_least_one_model:
        dfs, dates = cut_statistics_test_only(session.state)
        session.state.statistics_test_only, session.state.statistics_test_index = dfs, dates

if submit and not wells_to_calc:
    st.info('Не выбрано ни одной скважины для расчета.')
if adaptation_days_number < 90 or forecast_days_number < 28:
    st.error('**Период адаптации** должен быть не менее 90 суток. **Период прогноза** - не менее 28 суток.')
page = PAGES[selection]
page.show(session)
