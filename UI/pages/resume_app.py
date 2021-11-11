import io
import pandas as pd
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
    st.subheader("Импорт готового состояния программы")
    st.write("При импорте приложение попытается восстановить данные для всех вкладок приложения."
             "\n\nОднако не всегда возможно восстановить данные для вкладки **Настройки моделей**. "
             "В таких случаях вкладки **Аналитика** и **Скважина** все еще будут работать.")
    uploaded_session = st.file_uploader('Загрузить готовое состояние программы:',
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

    st.subheader("Экспорт текущего состояния программы")
    try:
        session_to_save = {key: session[key] for key in session}
        session_to_save = pickle.dumps(session_to_save)
        st.download_button(
            label="Экспорт текущего состояния программы",
            data=session_to_save,
            file_name=f'session_to_save.pickle',
            mime='application/octet-stream',
        )
    except pickle.PicklingError as err:
        st.error('Не удалось создать файл для экспорта. Результаты расчетов в формате .xlsx можно '
                 'экспортировать по кнопке ниже.')
        print(err)

    st.subheader("Экспорт результатов по всем скважинам в Excel-формате (.xlsx)")
    st.write("""**Внимание!** Результаты, экспортированные в формате .xlsx, будет 
    невозможно импортировать как состояние программы.""")
    # Подготовка данных к выгрузке
    if session.statistics:
        if session.buffer is None:
            session.buffer = io.BytesIO()
            with pd.ExcelWriter(session.buffer) as writer:
                for key in session.statistics:
                    session.statistics[key].to_excel(writer, sheet_name=key)
                if not session.ensemble_interval.empty:
                    session.ensemble_interval.to_excel(writer, sheet_name='ensemble_interval')
                if session.adapt_params:
                    df_adapt_params = pd.DataFrame(session.adapt_params)
                    df_adapt_params.to_excel(writer, sheet_name='adapt_params')
        st.download_button(
            label="Экспорт .xlsx",
            data=session.buffer,
            file_name=f'Все результаты {session.was_config.field_name}.xlsx',
            mime='text/csv',
        )
    else:
        st.info("Кнопка станет доступна, как только будет рассчитана хотя бы одна скважина.")
