import io
import pandas as pd
import streamlit as st

from UI.config import FTOR_DECODE
from UI.plots import create_well_plot


session = st.session_state


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


def show():
    if session.selected_wells:  # Проверка, рассчитана ли хоть одна скважина
        well_to_draw = st.selectbox(
                label='Скважина',
                options=sorted(session.selected_wells),
                key='well_to_calc'
        )
        well_name_ois = session.well_names_parsed[well_to_draw]

        fig = create_well_plot(
            session.df_draw_liq[well_name_ois],
            session.df_draw_oil[well_name_ois],
            session.df_draw_ensemble[well_name_ois],
            session.pressure[well_name_ois],
            session.date_test,
            session.events[well_name_ois],
            well_to_draw,
        )

        # Построение графика
        st.plotly_chart(fig, use_container_width=True)
        # Вывод параметров адаптации модели пьезопроводности
        # TODO: (возможно) могут выводиться значения параметров от предыдущих расчетов,
        #  если нынешние упали с ошибкой
        if session.is_calc_ftor and well_name_ois in session.adapt_params:
            result = session.adapt_params[well_name_ois][0].copy()
            result = convert_to_readable(result)
            st.write('Результаты адаптации модели пьезопроводности:', result)

        # Подготовка данных к выгрузке
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            session.df_draw_liq[well_name_ois].to_excel(writer, sheet_name='Дебит жидкости')
            session.df_draw_oil[well_name_ois].to_excel(writer, sheet_name='Дебит нефти')
            session.df_draw_ensemble[well_name_ois].to_excel(writer, sheet_name='Дебит нефти ансамбль')
            session.pressure[well_name_ois].to_excel(writer, sheet_name='Забойное давление')
            session.events[well_name_ois].to_excel(writer, sheet_name='Мероприятие')
        # Кнопка экспорта результатов
        st.download_button(
            label="Экспорт результатов",
            data=buffer,
            file_name=f'{well_name_ois}_data.xlsx',
            mime='text/csv',
        )
    else:
        st.info('Здесь будет отображаться прогноз добычи по выбранной скважине.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
