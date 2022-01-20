import io
import pandas as pd
import pickle
import streamlit as st
from pathlib import Path
from typing import IO

from statistics_explorer.config import ConfigStatistics
from UI.app_state import AppState
from UI.config import FIELDS_SHOPS


def show(session: st.session_state) -> None:
    state = session.state
    draw_upload_state(state)
    draw_export_state(state)
    draw_export_excel(state)
    draw_upload_oilfield_data()


def draw_upload_state(state: AppState) -> None:
    st.subheader("Импорт готового состояния программы")
    st.write(
        "При импорте приложение попытается восстановить данные для всех вкладок приложения."
        "\n\nОднако не всегда возможно восстановить данные для вкладки **Настройки моделей**. "
        "В таких случаях вкладки **Аналитика** и **Скважина** все еще будут работать."
    )
    uploaded_state = st.file_uploader(
        'Загрузить готовое состояние программы:',
        type='pickle',
        help="""Входной файл должен иметь расширение **.pickle**"""
    )
    if uploaded_state is not None:
        try:
            saved_state = pickle.load(uploaded_state)
            for key in saved_state:
                state[key] = saved_state[key]
            st.success("Расчеты обработаны успешно! "
                       "Обновлены вкладки **Карта скважин**, **Аналитика** и **Скважина**.")
        except pickle.UnpicklingError as err:
            st.error('Не удалось восстановить расчеты.', err)


def draw_export_state(state: AppState) -> None:
    st.subheader("Экспорт текущего состояния программы")
    try:
        state_to_save = {key: state[key] for key in state}
        state_to_save = pickle.dumps(state_to_save)
        st.download_button(
            label="Экспорт текущего состояния программы",
            data=state_to_save,
            file_name=f'state_to_save.pickle',
            mime='application/octet-stream',
        )
    except pickle.PicklingError as err:
        st.error('Не удалось создать файл для экспорта. Результаты расчетов в формате .xlsx можно '
                 'экспортировать по кнопке ниже.')
        print(err)


def draw_export_excel(state: AppState) -> None:
    st.subheader("Экспорт результатов по всем скважинам в Excel-формате (.xlsx)")
    st.write("""**Внимание!** Результаты, экспортированные в формате .xlsx, будет 
        невозможно импортировать как состояние программы.""")
    # Подготовка данных к выгрузке
    if state.statistics:
        if state.buffer is None:
            state.buffer = io.BytesIO()
            with pd.ExcelWriter(state.buffer) as writer:
                for key in state.statistics:
                    state.statistics[key].to_excel(writer, sheet_name=ConfigStatistics.MODEL_NAMES[key])
                if not state.ensemble_interval.empty:
                    state.ensemble_interval.to_excel(writer, sheet_name='Доверит. интервал ансамбль')
                if state.adapt_params:
                    df_adapt_params = pd.DataFrame(state.adapt_params)
                    df_adapt_params.to_excel(writer, sheet_name='Параметры адаптации пьезо')
        st.download_button(label="Экспорт .xlsx",
                           data=state.buffer,
                           file_name=f'Все результаты {state.was_config.field_name}.xlsx',
                           mime='text/csv', )
    else:
        st.info("Кнопка станет доступна, как только будет рассчитана хотя бы одна скважина.")


def draw_upload_oilfield_data() -> None:
    st.subheader("Загрузка входных данных по месторождению")
    oilfield_name = st.text_input('Введите название месторождения', max_chars=30)
    oilfield_shops = st.text_input('Введите название цехов',
                                   max_chars=30,
                                   help="""Введите название цехов в данном месторождении через запятую. 
                                         Например 'ЦДНГ-4', 'ЦДНГ-2' (H - латинская)""")
    oilfield_files = st.file_uploader('Загрузить данные по месторождению:',
                                      accept_multiple_files=True,
                                      type=['feather', 'xlsm'])
    button_add_oilfield = st.button('OK')
    if button_add_oilfield:
        add_oilfield(oilfield_name, oilfield_shops, oilfield_files)


def add_oilfield(oilfield_name: str,
                 oilfield_shops: str,
                 oilfield_files: IO) -> None:
    oilfield_shops = [shop.strip() for shop in oilfield_shops.strip().split(',')]
    FIELDS_SHOPS[oilfield_name] = oilfield_shops
    path_to_save = Path.cwd() / 'data' / oilfield_name
    if not path_to_save.exists():
        path_to_save.mkdir()
    for file in oilfield_files:
        with open(path_to_save / file.name, 'wb') as f:
            f.write(file.getbuffer())
