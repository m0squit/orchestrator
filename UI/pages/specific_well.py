import streamlit as st

from statistics_explorer.plots import create_well_plot_UI
from statistics_explorer.config import ConfigStatistics
from UI.cached_funcs import run_preprocessor


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
    preprocessor = run_preprocessor(session.was_config)
    well_ftor = preprocessor.create_wells_ftor([well_name_ois])[0]
    df_chess = well_ftor.df_chess
    fig = create_well_plot_UI(
        statistics=session.statistics,
        date_test=session.was_config.date_test,
        date_test_if_ensemble=session.dates_test_period[0],
        df_chess=df_chess,
        wellname=well_to_draw,
        MODEL_NAMES=ConfigStatistics.MODEL_NAMES,
        ensemble_interval=session.ensemble_interval
    )
    # Построение графика
    st.plotly_chart(fig, use_container_width=True)
    # Вывод параметров адаптации модели пьезопроводности
    if session.was_calc_ftor and well_to_draw in session.adapt_params:
        st.write('Результаты адаптации модели пьезопроводности:', session.adapt_params[well_to_draw])
