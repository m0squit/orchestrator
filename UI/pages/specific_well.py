import io
import pandas as pd
import streamlit as st

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


session = st.session_state


def show():
    well_to_draw = st.selectbox(
            label='Скважина',
            options=sorted(session.selected_wells),
            key='well_to_calc'
    )
    well_name_ois = session.well_names_parsed[well_to_draw]

    if session.fig[well_name_ois] is not None:
        # Построение графика
        st.plotly_chart(session.fig[well_name_ois], use_container_width=True)
        # Вывод параметров адаптации модели пьезопроводности
        # TODO: (возможно) могут выводиться значения параметров от предыдущих расчетов, если нынешние упали с ошибкой
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
        st.info('Выберите настройки и нажмите кнопку "Запустить расчеты"')