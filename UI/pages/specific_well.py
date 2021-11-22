import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from statistics_explorer.plots import calc_relative_error
from statistics_explorer.config import ConfigStatistics
from UI.cached_funcs import run_preprocessor


def create_well_plot_UI(statistics: dict,
                        date_test: datetime.date,
                        date_test_if_ensemble: datetime.date,
                        df_chess: pd.DataFrame,
                        wellname: str,
                        MODEL_NAMES: dict,
                        ensemble_interval: pd.DataFrame = pd.DataFrame()):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=['Дебит жидкости, м3',
                                        'Дебит нефти, м3',
                                        'Относительная ошибка по нефти, %',
                                        'Забойное давление, атм'])
    fig.update_layout(font=dict(size=15),
                      template='seaborn',
                      title_text=f'Скважина {wellname}',
                      legend=dict(orientation="v",
                                  font=dict(size=15),
                                  traceorder='normal'),
                      height=630,
                      width=1300)
    mark, m = dict(size=3), 'markers'
    colors = {'ftor': px.colors.qualitative.Pastel[1],
              'wolfram': 'rgba(248, 156, 116, 0.8)',
              'CRM': px.colors.qualitative.Pastel[6],
              'ensemble': 'rgba(115, 175, 72, 0.7)',
              'ensemble_interval': 'rgba(184, 247, 212, 0.7)',
              'true': 'rgba(99, 110, 250, 0.7)',
              'pressure': '#C075A6'}
    df_chess = df_chess.copy().dropna(subset=['Дебит жидкости', 'Дебит нефти'], how='any')
    y_liq_true = df_chess['Дебит жидкости']
    y_oil_true = df_chess['Дебит нефти']
    if not ensemble_interval.empty and f'{wellname}_lower' in ensemble_interval.columns:
        trace = go.Scatter(name=f'OIL: Доверит. интервал',
                           x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_lower'],
                           mode='lines', line=dict(width=1, color=colors['ensemble_interval']))
        fig.add_trace(trace, row=2, col=1)
        trace = go.Scatter(name=f'OIL: Доверит. интервал',
                           x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_upper'],
                           fill='tonexty', mode='lines', line=dict(width=1, color=colors['ensemble_interval']))
        fig.add_trace(trace, row=2, col=1)
        fig.add_vline(x=date_test_if_ensemble, line_width=1, line_dash='dash', exclude_empty_subplots=False)
    # Факт
    trace = go.Scatter(name=f'LIQ: {MODEL_NAMES["true"]}', x=y_liq_true.index, y=y_liq_true,
                       mode=m, marker=dict(size=5, color=colors['true']))
    fig.add_trace(trace, row=1, col=1)
    trace = go.Scatter(name=f'OIL: {MODEL_NAMES["true"]}', x=y_oil_true.index, y=y_oil_true,
                       mode=m, marker=dict(size=5, color=colors['true']))
    fig.add_trace(trace, row=2, col=1)
    for model in statistics:
        if f'{wellname}_oil_pred' in statistics[model]:
            clr = colors[model]
            y_liq = statistics[model][f'{wellname}_liq_pred'].dropna()
            y_oil = statistics[model][f'{wellname}_oil_pred'].dropna()
            trace_liq = go.Scatter(name=f'LIQ: {MODEL_NAMES[model]}', x=y_liq.index, y=y_liq,
                                   mode=m, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace_liq, row=1, col=1)  # Дебит жидкости
            trace_oil = go.Scatter(name=f'OIL: {MODEL_NAMES[model]}', x=y_oil.index, y=y_oil,
                                   mode=m, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace_oil, row=2, col=1)  # Дебит нефти
            deviation = calc_relative_error(y_oil_true, y_oil, use_abs=False)
            trace_err = go.Scatter(name=f'OIL ERR: {MODEL_NAMES[model]}', x=deviation.index, y=deviation,
                                   mode=m, marker=dict(size=4), line=dict(width=1, color=clr))
            fig.add_trace(trace_err, row=3, col=1)  # Ошибка по нефти
    # Забойное давление
    pressure = df_chess['Давление забойное']
    trace_pressure = go.Scatter(name=f'Заб. давление', x=pressure.index, y=pressure,
                                mode=m, marker=dict(size=4, color=colors['pressure']))
    fig.add_trace(trace_pressure, row=4, col=1)
    # Мероприятия
    events = df_chess['Мероприятие']
    _events = events.dropna()
    trace_events = go.Scatter(name='Мероприятие', x=_events.index, y=[0.2] * len(_events),
                              mode='markers+text', marker=dict(size=8), text=_events.array,
                              textposition='top center', textfont=dict(size=12))
    fig.add_trace(trace_events, row=4, col=1)
    fig.add_vline(x=date_test, line_width=1, line_dash='dash')
    return fig


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
