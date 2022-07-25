from datetime import date, timedelta
from typing import Optional, Union

import streamlit as st
from loguru import logger

import UI.pages
from UI.cached_funcs import calculate_ftor, calculate_wolfram, calculate_ensemble, run_preprocessor,\
    calculate_fedot, calculate_CRM,  calculate_shelf
from UI.config import FIELDS_SHOPS, DATE_MIN, DATE_MAX, DEFAULT_FTOR_BOUNDS
from UI.data_processor import *
from frameworks_crm.class_CRM.calculator import Calculator as CalculatorCRM
from frameworks_ftor.ftor.well import Well
from tools_preprocessor.config import Config as ConfigPreprocessor
from tools_preprocessor.preprocessor import Preprocessor


def start_streamlit() -> st.session_state:
    """Возвращает инициализированную сессию streamlit."""
    # Мета-настройки для Streamlit
    st.set_page_config(
        page_title='КСП',
        layout="wide"  # Для отображения на всю ширину браузера
    )
    # Инициализация значений сессии st.session_state
    _session = st.session_state
    if 'date_start' not in _session:
        initialize_session(_session)
    return _session


def start_logger() -> None:
    """Инициализация логгера."""
    logger.remove()
    logger.add('logs/log.log', format="{time:YYYY-MM-DD at HH:mm:ss} {level} {message}",
               level="DEBUG", rotation="1 MB", compression="zip", enqueue=True)
    logger.info('Start UI')


def initialize_session(_session: st.session_state) -> None:
    """Инициализация сессии streamlit.session_state и лога.

    - Инициализируется связь с лог-файлом.
    - Инициализируется пустой словарь состояния программы session.state.
    - Инициализируются значения параметров моделей для страницы models_settings.py

    Notes:
        Функция используется только при первом рендеринге приложения.
    """
    start_logger()
    _session.state = AppState()
    # Ftor model
    _session.constraints = {}
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
    # CRM model
    _session.CRM_influence_R = 1300
    _session.CRM_maxiter = 100
    _session.CRM_p_res = 220
    # Ensemble model
    _session.ensemble_adapt_period = 28
    _session.interval_probability = 0.9
    _session.draws = 300
    _session.tune = 200
    _session.chains = 1
    _session.target_accept = 0.95


def parse_well_names(well_names_ois: List[int], field_name: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Функция сопоставляет имена скважин OIS и (ГРАД?)

    Parameters
    ----------
    well_names_ois: List[int]
        список имен скважин в формате OIS (например 245023100).
    field_name : str
        Имя месторождения.
    Returns
    -------
    wellnames_key_normal : Dict[str, int]
        Ключ = имя скважины в формате ГРАД, значение - имя скважины OIS.
    wellnames_key_ois : Dict[int, str]
        Ключ = имя скважины OIS, значение - имя скважины в формате ГРАД.
    """
    # TODO: заменить костыльный способ
    welllist = pd.read_feather(Preprocessor._path_general / field_name / 'welllist.feather')
    wellnames_key_normal_ = {}
    wellnames_key_ois_ = {}
    for name_ois in well_names_ois:
        well_name_norm = welllist[welllist.ois == name_ois]
        well_name_norm = well_name_norm[well_name_norm.npath == 0]
        well_name_norm = well_name_norm.at[well_name_norm.index[0], 'num']
        wellnames_key_normal_[well_name_norm] = name_ois
        wellnames_key_ois_[name_ois] = well_name_norm
    return wellnames_key_normal_, wellnames_key_ois_


def save_current_state(
        state: AppState,
        _session: st.session_state,
        config: ConfigPreprocessor,
        models_to_run: dict[str, bool],
        date_start: date,
        date_test: date,
        date_end: date,
        selected_wells_norm: list[str],
        selected_wells_ois: list[int],
        wellnames_key_normal: dict[str, int],
        wellnames_key_ois: dict[int, str],
        wells_ftor: list[Well]
) -> AppState:
    """
    Функция сохраняет состояние программы в объект state класса AppState.

    Parameters
    ----------
    state : AppState
        Переменная, в которую будет записано состояние программы.
    _session: streamlit.session_state
        Сессия приложения, из которой будет извлекаться состояние программы.
    config : ConfigPreprocessor
        Конфигурация месторождения, дат адаптации и прогноза, выбранная пользователем.
    models_to_run : dict[str, bool]
        Словарь ключ - имя модели, значение - выбрана ли модель для расчета.
    date_start : date
        Дата начала адаптации.
    date_test : date
        Дата начала прогноза.
    date_end : date
        Дата конца прогноза.
    selected_wells_norm : list[str]
        Список выбранных скважин для расчета в формате ГРАД.
    selected_wells_ois : list[int]
        Список выбранных скважин для расчета в формате OIS.
    wellnames_key_normal : Dict[str, int]
        Ключ = имя скважины в формате ГРАД, значение - имя скважины OIS.
    wellnames_key_ois : Dict[int, str]
        Ключ = имя скважины OIS, значение - имя скважины в формате ГРАД.
    wells_ftor : list[Well]
        Список объектов скважин Well.
    Returns
    -------
    """
    state['adapt_params'] = {}
    state['buffer'] = None
    state['ensemble_interval'] = pd.DataFrame()
    state['exclude_wells'] = []
    state['statistics'] = {}
    state['statistics_test_only'] = {}
    state['selected_wells_norm'] = selected_wells_norm.copy()
    state['selected_wells_ois'] = selected_wells_ois.copy()
    state['was_config'] = config
    state['was_calc_ftor'] = models_to_run['ftor']
    state['was_calc_wolfram'] = models_to_run['wolfram']
    state['was_calc_CRM'] = models_to_run['CRM']
    state['was_calc_shelf'] = models_to_run['shelf']
    state['was_calc_ensemble'] = models_to_run['ensemble']
    state['was_date_start'] = date_start
    state['was_date_test'] = date_test
    state['was_date_test_if_ensemble'] = date_test + timedelta(days=_session.ensemble_adapt_period)
    state['was_date_end'] = date_end
    state['wellnames_key_normal'] = wellnames_key_normal.copy()
    state['wellnames_key_ois'] = wellnames_key_ois.copy()
    state['wells_ftor'] = wells_ftor
    return state


def select_page(pages: Dict[str, Any]) -> str:
    """Виджет выбора страницы (вкладки) в интерфейсе.
    """
    _selected_page = st.radio("", list(pages.keys()))
    return _selected_page


def select_models() -> Dict[str, bool]:
    """Виджет выбора моделей для расчета.
    """
    selected_models = {
        'ftor': st.checkbox(
            label='Считать модель пьезопр-ти',
            value=True,
            key='is_calc_ftor',
        ),
        'wolfram': st.checkbox(
            label='Считать модель ML',
            value=True,
            key='is_calc_wolfram',
        ),
        'CRM': st.checkbox(
            label='Считать модель CRM',
            value=True,
            key='is_calc_CRM',
        ),
        'shelf': st.checkbox(
            label='Считать модель ППТП',
            value=True,
            key='is_calc_shelf',
        ),
        'ensemble': st.checkbox(
            label='Считать ансамбль моделей',
            value=True,
            key='is_calc_ensemble',
            help='Ансамбль возможно рассчитать, если рассчитана хотя бы одна модель.'
        ),
    }
    return selected_models


def select_oilfield(fields_shops: Dict[str, List[str]]) -> str:
    """Виджет выбора месторождения для расчета.

    Parameters
    ----------
    fields_shops : Dict[str, List[str]]
        цеха для каждого месторождения
    """
    oilfield_name = st.selectbox(
        label='Месторождение',
        options=fields_shops.keys(),
        key='field_name',
    )
    return oilfield_name


def select_shops(oilfield_name: str) -> List[str]:
    """Виджет выбора списка цехов для выбранного месторождения.

    Parameters
    ----------
    oilfield_name : str
    """
    selected_shops = st.multiselect(
        label='Цех добычи',
        options=['Все'] + FIELDS_SHOPS[oilfield_name],
        default='Все',
        key='shops'
    )
    if 'Все' in selected_shops:
        selected_shops = FIELDS_SHOPS[oilfield_name]
    return selected_shops


def select_dates(date_min: date,
                 date_max: date) -> Tuple[date, date, date]:
    """Виджет выбора дат адаптации и прогноза.
    """
    date_start_ = st.date_input(
        label='Дата начала адаптации (с 00:00)',
        min_value=date_min,
        value=date(2021, 3, 1),
        max_value=date_max,
        key='date_start',
        help="""
        Данная дата используется только для модели пьезопроводности.
        Адаптация модели ML проводится на всех доступных по скважине данных.
        """,
    )
    date_test_ = st.date_input(
        label='Дата начала прогноза (с 00:00)',
        min_value=date_min,
        value=date(2021, 12, 2),
        max_value=date_max,
        key='date_test',
    )
    date_end_ = st.date_input(
        label='Дата конца прогноза (по 23:59)',
        min_value=date_min,
        value=date(2022, 4, 30),
        max_value=date_max,
        key='date_end',
    )
    return date_start_, date_test_, date_end_


def select_wells_to_calc(wellnames_key_normal_: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """Виджет выбора скважин для расчета.
    """
    wells_norm = st.multiselect(label='Скважина',
                                options=['Все скважины'] + list(wellnames_key_normal_.keys()),
                                key='selected_wells_norm')
    if 'Все скважины' in wells_norm:
        wells_norm = list(wellnames_key_normal_.keys())
    wells_ois = [wellnames_key_normal_[well_name_] for well_name_ in wells_norm]
    return wells_norm, wells_ois


def check_for_correct_params(date_start_: date,
                             date_test_: date,
                             date_end_: date,
                             pressed_submit: bool,
                             selected_wells_norm_: List[str]) -> None:
    """Проверяет корректность параметров, выбранных пользователем.

    - Даты адаптации
    - Даты прогноза
    - Выбрана ли хоть одна скважина для расчета
    """
    adaptation_days_number = (date_test_ - date_start_).days
    forecast_days_number = (date_end_ - date_test_).days
    if adaptation_days_number < 90 or forecast_days_number < 28:
        st.error('**Период адаптации** должен быть не менее 90 суток. **Период прогноза** - не менее 28 суток.')
    if pressed_submit and not selected_wells_norm_:
        st.info('Не выбрано ни одной скважины для расчета.')


def run_models(_session: st.session_state,
               _models_to_run: Dict[str, bool],
               _preprocessor: Preprocessor,
               wells_ois: List[int],
               wells_norm: List[str],
               date_start_adapt: date,
               date_start_forecast: date,
               date_end_forecast: date,
               oilfield: str,
               shops: List[str]) -> None:
    """Запуск расчета моделей, которые выбрал пользователь.

    Parameters
    ----------
    _session : st.session_state
        текущая сессия streamlit. В ней содержатся настройки моделей и
        текущее состояние программы _session.state.
    _models_to_run : Dict[str, bool]
        модели для расчета, которые выбрал пользователь.
    _preprocessor : Preprocessor
        препроцессор с конфигурацией, заданной пользователем.
    wells_ois : List[int]
        список имен скважин в формате OIS.
    wells_norm : List[str]
        список имен скважин в "читаемом" формате (ГРАД?).
    date_start_adapt : date
        дата начала адаптации для модели пьезопроводности.
    date_start_forecast : date
        дата начала прогноза для всех моделей, кроме ансамбля.
    date_end_forecast : date
        дата конца прогноза для всех моделей.
    oilfield : str
        название месторождения, которое выбрал пользователь.

    Notes
    -------
    В конце расчета каждой из моделей вызывается функция извлечения результатов.
    Таким образом все результаты приводятся к единому формату данных.
    """
    at_least_one_model = _models_to_run['ftor'] or _models_to_run['wolfram'] or _models_to_run['CRM'] or _models_to_run['shelf']
    if _models_to_run['ftor']:
        run_ftor(_preprocessor, wells_ois, _session.constraints, _session.state)
    if _models_to_run['wolfram']:
        run_wolfram(date_start_forecast, date_end_forecast, _preprocessor,
                    wells_ois, _session, _session.state)
    if _models_to_run['CRM']:
        calculator_CRM = run_CRM(date_start_adapt, date_start_forecast, date_end_forecast,
                                 oilfield, _session, _session.state)
        if calculator_CRM is not None:
            run_fedot(oilfield, date_start_adapt, date_start_forecast, date_end_forecast, wells_norm,
                      calculator_CRM.f, _session.state)
    if _models_to_run['shelf']:
        run_shelf(oilfield, shops, wells_ois, date_start_adapt, date_start_forecast, date_start_adapt,
                  date_end_forecast, 30, 5, _session.state)
    if at_least_one_model:
        make_models_stop_well(_session.state['statistics'], _session.state['selected_wells_norm'])
    if _models_to_run['ensemble'] and at_least_one_model:
        run_ensemble(_session, wells_norm, mode='liq')
        run_ensemble(_session, wells_norm, mode='oil')


def run_ftor(_preprocessor: Preprocessor,
             wells_ois: List[int],
             constraints: Optional[Dict[
                 str, Union[float, Dict[str, Union[bool, List[float]]]]
             ]],
             state: AppState) -> None:
    """Расчет модели пьезопроводности и последующее извлечение результатов.

    Parameters
    ----------
    _preprocessor : Preprocessor
        препроцессор с конфигурацией, заданной пользователем.
    wells_ois : List[int]
        список имен скважин в формате OIS.
    constraints : Dict
        словарь с границами параметров адаптации.
    state : AppState
        состояние программы, заданное пользователем.
    """
    calculator_ftor = calculate_ftor(_preprocessor, wells_ois, constraints)
    extract_data_ftor(calculator_ftor, state)


def run_wolfram(date_start_forecast: date,
                date_end_forecast: date,
                _preprocessor: Preprocessor,
                wells_ois: List[int],
                _session: st.session_state,
                state: AppState) -> None:
    """Расчет модели ML и последующее извлечение результатов.

    Parameters
    ----------
    date_start_forecast : date
        дата начала прогноза.
    date_end_forecast : date
        дата конца прогноза.
    _preprocessor : Preprocessor
        препроцессор с конфигурацией, заданной пользователем.
    wells_ois : List[int]
        список имен скважин в формате OIS.
    _session : st.session_state
        текущая сессия streamlit. В ней содержатся настройки моделей и
        текущее состояние программы _session.state.
    state: AppState
        состояние программы, заданное пользователем.
    """
    forecast_days_number = (date_end_forecast - date_start_forecast).days + 1
    calculator_wolfram = calculate_wolfram(_preprocessor,
                                           wells_ois,
                                           forecast_days_number,
                                           _session.estimator_name_group,
                                           _session.estimator_name_well,
                                           _session.is_deep_grid_search,
                                           _session.window_sizes,
                                           _session.quantiles)
    extract_data_wolfram(calculator_wolfram, state)
    convert_tones_to_m3_for_wolfram(state, state.wells_ftor)


def run_CRM(date_start_adapt: date,
            date_start_forecast: date,
            date_end_forecast: date,
            oilfield: str,
            _session: st.session_state,
            state: AppState) -> CalculatorCRM:
    """Расчет модели CRM и последующее извлечение результатов.

    Parameters
    ----------
    date_start_adapt : date
        дата начала адаптации.
    date_start_forecast : date
        дата начала прогноза.
    date_end_forecast : date
        дата конца прогноза.
    oilfield : str
        название месторождения, выбранное пользователем.
    _session : st.session_state
        текущая сессия streamlit. В ней содержатся настройки моделей и
        текущее состояние программы _session.state.
    state: AppState
        состояние программы, заданное пользователем.
    """
    calculator_CRM = calculate_CRM(date_start_adapt=date_start_adapt,
                                   date_end_adapt=date_start_forecast - timedelta(days=1),
                                   date_end_forecast=date_end_forecast,
                                   oilfield=oilfield,
                                   influence_R=_session.CRM_influence_R,
                                   maxiter=_session.CRM_maxiter,
                                   p_res=_session.CRM_p_res)
    if calculator_CRM is not None:
        extract_data_CRM(calculator_CRM.pred_CRM, state, state['wells_ftor'], mode='CRM')
    return calculator_CRM


def run_fedot(oilfield: str,
              date_start: date,
              date_test: date,
              date_end: date,
              wells_norm: List,
              coeff: pd.DataFrame,
              state: AppState,
              lags: pd.DataFrame = None) -> None:
    """Расчет модели Fedot (поверх CRM) и последующее извлечение результатов.

    Parameters
    ----------
    oilfield : str
        название месторождения, выбранное пользователем.
    date_start : date
        дата начала адаптации.
    date_test : date
        дата начала прогноза.
    date_end : date
        дата конца прогноза.
    wells_norm : List[str]
        список имен скважин в "читаемом" формате (ГРАД?).
    coeff : pd.DataFrame
        коэффициенты взаимовлияния скважин
    state : AppState
        состояние программы, заданное пользователем.
    """
    calculator_fedot = calculate_fedot(oilfield=oilfield,
                                       train_start=date_start,
                                       train_end=date_test - timedelta(days=1),
                                       predict_start=date_test,
                                       predict_end=date_end,
                                       wells_norm=wells_norm,
                                       coeff=coeff,
                                       lags=lags)
    extract_data_fedot(calculator_fedot, state)

def run_shelf(oilfield: str,
              shops: List[str],
              well_ois: List[int],
              train_start: date,
              train_end: date,
              predict_start: date,
              predict_end: date,
              n_days_past: int,
              n_days_calc_avg: int,
              state: AppState) -> None:
    """Расчет модели прогноза по темпам падений и последующее извлечение результатов.
       Parameters
       ----------
       oilfield : str
           название месторождения, выбранное пользователем.
       date_start : date
           дата начала адаптации.
       date_test : date
           дата начала прогноза.
       date_end : date
           дата конца прогноза.
       state : AppState
           состояние программы, заданное пользователем.
       """
    print('run_shelf inside')
    calculator_shelf = calculate_shelf(oilfield,
                                       shops,
                                       well_ois,
                                       train_start,
                                       train_end,
                                       predict_start,
                                       predict_end,
                                       n_days_past,
                                       n_days_calc_avg)
    # print('run shelf done')
    extract_data_shelf(calculator_shelf,state)

def run_ensemble(_session: st.session_state,
                 wells_norm: list[str],
                 mode: str = 'liq') -> None:
    """Расчет ансамбля моделей, доверительного интервала, и последующее извлечение результатов.

    Parameters
    ----------
    _session : st.session_state
        текущая сессия streamlit. В ней содержатся настройки моделей и
        текущее состояние программы _session.state.
    wells_norm: list[str]
        список имен скважины в формате (ГРАД?).
    mode: str
        режим расчета жидкости/нефти.
    """
    name_of_y_true = 'true'
    input_data = prepare_data_for_ensemble(_session.state, wells_norm, name_of_y_true, mode)
    ensemble_result = calculate_ensemble(
        input_data,
        adaptation_days_number=_session.ensemble_adapt_period,
        interval_probability=_session.interval_probability,
        draws=_session.draws,
        tune=_session.tune,
        chains=_session.chains,
        target_accept=_session.target_accept,
        name_of_y_true=name_of_y_true)
    for well_name_normal in ensemble_result.keys():
        extract_data_ensemble(ensemble_result[well_name_normal], _session.state, well_name_normal, mode)


@logger.catch
def main():
    session = start_streamlit()
    # Реализация UI: сайдбар
    with st.sidebar:
        selected_page = select_page(PAGES)
        models_to_run = select_models()
        field_name = select_oilfield(FIELDS_SHOPS)
        shops = select_shops(field_name)
        date_start, date_test, date_end = select_dates(date_min=DATE_MIN, date_max=DATE_MAX)

        config = ConfigPreprocessor(field_name, shops, date_start, date_test, date_end)
        preprocessor = run_preprocessor(config)
        wellnames_key_normal, wellnames_key_ois = parse_well_names(preprocessor.well_names, field_name)
        selected_wells_norm, selected_wells_ois = select_wells_to_calc(wellnames_key_normal)

        submit = st.button(label='Запустить расчеты')
    check_for_correct_params(date_start, date_test, date_end, submit, selected_wells_norm)

    # Нажата кнопка "Запуск расчетов"
    if submit and selected_wells_norm:
        logger.info('Submit button pressed.')
        session.state = save_current_state(
            AppState(),
            session,
            config,
            models_to_run,
            date_start,
            date_test,
            date_end,
            selected_wells_norm,
            selected_wells_ois,
            wellnames_key_normal,
            wellnames_key_ois,
            preprocessor.create_wells_ftor(selected_wells_ois)
        )
        # Запуск моделей
        run_models(session, models_to_run, preprocessor,
                   selected_wells_ois, selected_wells_norm,
                   date_start, date_test, date_end, field_name, shops)
        logger.success('Finish calculations.')
        # Выделение прогнозов моделей
        dfs, dates = cut_statistics_test_only(session.state)
        session.state.statistics_test_only, session.state.statistics_test_index = dfs, dates

    # Отображение выбранной страницы
    page = PAGES[selected_page]
    page.show(session)


PAGES = {
    "Настройки моделей": UI.pages.models_settings,
    "Карта скважин": UI.pages.wells_map,
    "Аналитика": UI.pages.analytics,
    "Скважина": UI.pages.specific_well,
    "Импорт/экспорт расчетов": UI.pages.resume_app,
}

if __name__ == '__main__':
    main()
