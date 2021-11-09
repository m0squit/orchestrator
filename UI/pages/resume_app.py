import pickle
import streamlit as st
from UI.pages.models_settings import update_ftor_constraints, update_ML_params, update_ensemble_params

keys_to_update = [
    'adapt_params',
    'dates',
    'dates_test_period',
    'ensemble_interval',
    'selected_wells_norm',
    'selected_wells_ois',
    'statistics',
    'statistics_df_test',
    'was_config',
    'was_calc_ftor',
    'was_calc_wolfram',
    'was_calc_ensemble',
    'wellnames_key_normal',
    'wellnames_key_ois',
]


def show(session):
    try:
        session_to_save = {key: session[key] for key in session}
        session_to_save = pickle.dumps(session_to_save)
        st.download_button(
            label="Экспорт состояния программы",
            data=session_to_save,
            file_name=f'session_to_save.pickle',
            mime='application/octet-stream',
        )
    except pickle.PicklingError:
        st.error('Не удалось создать файл для экспорта. Результаты расчетов в формате .xlsx можно '
                 'экспортировать с вкладки "Аналитика"')

    uploaded_session = st.file_uploader('Загрузить готовое состояние программы',
                                        type='pickle',
                                        help="""Входной файл должен иметь расширение **.pickle**""")
    if uploaded_session is not None:
        saved_session = None
        try:
            saved_session = pickle.load(uploaded_session)
            for key in keys_to_update:
                session[key] = saved_session[key]
            for key in saved_session:
                if key.endswith('_'):
                    session[key] = saved_session[key]
            st.success("Расчеты обработаны успешно! Обновлены вкладки **Скважина** и **Аналитика**.")
        except pickle.UnpicklingError as err:
            st.error('Не удалось восстановить расчеты.', err)

        # Обновление параметров моделей на странице UI.pages.models_settings.py
        try:
            update_ftor_constraints(write_from=saved_session, write_to=session)
            st.success('Настройки модели пьезопроводности обновлены.')
        except:
            st.info('Не удалось загрузить настройки модели пьезопроводности.'
                    ' Используются границы параметров по умолчанию.')
        try:
            update_ML_params(write_from=saved_session, write_to=session)
            st.success('Настройки модели ML обновлены.')
        except:
            st.info('Не удалось загрузить настройки модели ML.'
                    ' Используются настройки по умолчанию.')
        try:
            update_ensemble_params(write_from=saved_session, write_to=session)
            st.success('Настройки ансамбля моделей обновлены.')
        except:
            st.info('Не удалось загрузить настройки ансамбля.'
                    ' Используются настройки по умолчанию.')
