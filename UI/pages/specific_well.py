import io
import pandas as pd
import streamlit as st

from statistics_explorer.plots import create_well_plot_UI
from statistics_explorer.config import ConfigStatistics
from UI.config import FTOR_DECODE


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


def show(session):
    if not session.selected_wells_norm:
        st.info('Здесь будет отображаться прогноз добычи по выбранной скважине.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return

    well_to_draw = st.selectbox(
            label='Скважина',
            options=sorted(session.selected_wells_norm),
            key='well_to_calc'
    )
    well_name_ois = session.wellnames_key_normal[well_to_draw]

    well_wolfram = session.preprocessor.create_wells_wolfram([well_name_ois])[0]
    well_ftor = session.preprocessor.create_wells_ftor([well_name_ois])[0]
    events = well_ftor.df_chess['Мероприятие']
    fig = create_well_plot_UI(
        session.statistics,
        well_wolfram,
        session.dates,
        session.date_test,
        session.dates_test_period[0],
        events,
        well_to_draw,
        ConfigStatistics.MODEL_NAMES,
        ensemble_interval=session.ensemble_interval
    )
    # Построение графика
    st.plotly_chart(fig, use_container_width=True)
    # Вывод параметров адаптации модели пьезопроводности
    if session.was_calc_ftor and well_to_draw in session.adapt_params:
        result = session.adapt_params[well_to_draw][0].copy()
        result = convert_to_readable(result)
        st.write('Результаты адаптации модели пьезопроводности:', result)
    # # Подготовка данных к выгрузке
    # buffer = io.BytesIO()
    # with pd.ExcelWriter(buffer) as writer:
    #     session.df_draw_liq[well_name_ois].to_excel(writer, sheet_name='Дебит жидкости')
    #     session.df_draw_oil[well_name_ois].to_excel(writer, sheet_name='Дебит нефти')
    #     session.df_draw_ensemble[well_name_ois].to_excel(writer, sheet_name='Дебит нефти ансамбль')
    #     session.pressure[well_name_ois].to_excel(writer, sheet_name='Забойное давление')
    #     session.events[well_name_ois].to_excel(writer, sheet_name='Мероприятие')
    # # Кнопка экспорта результатов
    # st.download_button(
    #     label="Экспорт результатов по скважине",
    #     data=buffer,
    #     file_name=f'Скважина {session.wellnames_key_ois[well_name_ois]}.xlsx',
    #     mime='text/csv',
    # )
