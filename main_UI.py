import streamlit as st
import UI.pages.models_settings
import UI.pages.wells_map
import UI.pages.analytics
import UI.pages.specific_well

from config import Config as ConfigPreprocessor
from preprocessor import Preprocessor
from UI.cached_funcs import calculate_ftor, calculate_wolfram, calculate_ensemble, run_preprocessor
from UI.config import FIELDS_SHOPS, DATE_MIN, DATE_MAX, DEFAULT_FTOR_BOUNDS
from UI.data_processor import *


def initialize_session(_session):
    _session.selected_wells_ois = []
    _session.selected_wells_norm = []
    _session.analytics_plots = {}
    _session.statistics = {}
    _session.statistics_df_test = {}
    _session.ensemble_interval = pd.DataFrame()
    # Ftor model
    _session.adapt_params = {}
    _session.constraints = {}
    session.was_calc_ftor = False
    session.was_calc_wolfram = False
    session.was_calc_ensemble = False
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        _session[f'{param_name}_is_adapt'] = True
        _session[f'{param_name}_lower'] = param_dict['lower_val']
        _session[f'{param_name}_default'] = param_dict['default_val']
        _session[f'{param_name}_upper'] = param_dict['upper_val']
    # ML model
    _session.estimator_name_group = 'xgb'
    _session.estimator_name_well = 'svr'
    _session.is_deep_grid_search = False
    _session.quantiles = [0.1, 0.3]
    _session.window_sizes = [3, 5, 7, 15, 30]
    # Ensemble model
    _session.adaptation_days_number = 28
    _session.interval_probability = 0.9
    _session.draws = 300
    _session.tune = 200
    _session.chains = 1
    _session.target_accept = 0.95


def clear_session():
    st.session_state.wells_to_calc = []
    st.session_state.selected_wells_ois = []
    st.session_state.selected_wells_norm = []
    session.was_calc_ftor = False
    session.was_calc_wolfram = False
    session.was_calc_ensemble = False


def parse_well_names(well_names_ois):
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


st.set_page_config(
    page_title='КСП',
    layout="wide"  # Для отображения на всю ширину браузера
)

PAGES = {
    "Настройки моделей": UI.pages.models_settings,
    "Карта скважин": UI.pages.wells_map,
    "Аналитика": UI.pages.analytics,
    "Скважина": UI.pages.specific_well,
}

# Инициализация значений сессии st.session_state
session = st.session_state
if 'date_start' not in session:
    initialize_session(session)

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
        on_change=clear_session
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
    config = ConfigPreprocessor(
        field_name,
        FIELDS_SHOPS[field_name],
        date_start,
        date_test,
        date_end,
    )
    preprocessor = run_preprocessor(config)
    session.wellnames_key_normal, session.wellnames_key_ois = parse_well_names(preprocessor.well_names)
    wells_to_calc = st.multiselect(label='Скважина',
                                   options=['Все скважины'] + list(session.wellnames_key_normal.keys()),
                                   key='wells_to_calc')
    if 'Все скважины' in wells_to_calc:
        wells_to_calc = list(session.wellnames_key_normal.keys())
    selected_wells_ois = [session.wellnames_key_normal[well_name_] for well_name_ in wells_to_calc]

    CRM_xlsx = st.file_uploader('Загрузить прогноз CRM по нефти', type='xlsx')
    submit = st.button(label='Запустить расчеты')

if submit and wells_to_calc:
    session.adapt_params = {}
    session.statistics = {}
    session.statistics_df_test = {}
    session.ensemble_interval = pd.DataFrame()
    session.selected_wells_norm = wells_to_calc.copy()
    session.selected_wells_ois = selected_wells_ois.copy()
    session.was_preprocessor = Preprocessor(config)
    session.was_calc_ftor = is_calc_ftor
    session.was_calc_wolfram = is_calc_wolfram
    session.was_calc_ensemble = is_calc_ensemble
    session.dates = pd.date_range(date_start, date_end, freq='D')
    if is_calc_ftor:
        calculator_ftor = calculate_ftor(preprocessor, selected_wells_ois, session.constraints)
        extract_data_ftor(calculator_ftor, session)
    if is_calc_wolfram:
        calculator_wolfram = calculate_wolfram(preprocessor,
                                               selected_wells_ois,
                                               forecast_days_number,
                                               session.estimator_name_group,
                                               session.estimator_name_well,
                                               session.is_deep_grid_search,
                                               session.window_sizes,
                                               session.quantiles)
        extract_data_wolfram(calculator_wolfram, session)
    if CRM_xlsx is None:
        session.pop('df_CRM', None)
    else:
        df_CRM = pd.read_excel(CRM_xlsx, index_col=0, engine='openpyxl')
        session['df_CRM'] = df_CRM
        extract_data_CRM(session['df_CRM'], session)
    if is_calc_ensemble and (is_calc_ftor or is_calc_wolfram):
        name_of_y_true = 'true'
        for ind, well_name_ois in enumerate(selected_wells_ois):
            well_name_normal = session.wellnames_key_ois[well_name_ois]
            # rewrite_fact_data_from_wolfram(session, preprocessor)
            print(f'\nWell {ind} out of {len(selected_wells_ois)}\n')
            input_df = prepare_df_for_ensemble(session, well_name_normal, name_of_y_true)
            ensemble_result = calculate_ensemble(
                input_df,
                adaptation_days_number=session.adaptation_days_number,
                interval_probability=session.interval_probability,
                draws=session.draws,
                tune=session.tune,
                chains=session.chains,
                target_accept=session.target_accept,
                name_of_y_true=name_of_y_true
            )
            if not ensemble_result.empty:
                extract_data_ensemble(ensemble_result, session, well_name_normal)

    if is_calc_ftor or is_calc_wolfram:
        session.statistics_df_test, session.dates_test_period = create_statistics_df_test(session)

if submit and not wells_to_calc:
    st.info('Не выбрано ни одной скважины для расчета.')
if adaptation_days_number < 90 or forecast_days_number < 28:
    st.error('**Период адаптации** должен быть не менее 90 суток. **Период прогноза** - не менее 28 суток.')
page = PAGES[selection]
page.show(session)
