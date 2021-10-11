import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots


session = st.session_state

_MODEL_NAMES = {
    'ftor': 'Пьезо',
    'wolfram': 'ML',
    'CRM': 'CRM',
    'true': 'Факт',
}


def compute_deviation(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    devs = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), np.abs(y_pred)) * 100
    return devs


def calc_relative_error(y_true: pd.Series,
                        y_pred: pd.Series) -> pd.Series:
    err = np.abs(y_pred - y_true) / np.maximum(y_pred, y_true)
    return err * 100


def create_well_plot(df_liq: pd.DataFrame,
                     df_oil: pd.DataFrame,
                     df_ensemble: pd.DataFrame,
                     pressure: pd.Series,
                     date_test: datetime.datetime,
                     events: pd.DataFrame,
                     wellname='',
                     ):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        # TODO: в м3 или нет?
        subplot_titles=[
            'Дебит жидкости, м3',
            'Дебит нефти, м3',
            'Относительная ошибка по нефти, %',
            'Забойное давление, атм',
        ]
    )
    fig.layout.template = 'seaborn'
    fig.update_layout(
        font=dict(size=15),
        title_text=f'Скважина {wellname}',
        legend=dict(orientation="v",
                    font=dict(size=15),
                    traceorder='normal'
                    ),
        height=760,
        width=1300,
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Pastel
    clr_fact = 'rgba(99, 110, 250, 0.7)'
    clr_pressure = '#C075A6'

    # Ансамбль
    if not df_ensemble.empty:
        trace = go.Scatter(name=f'OIL: Ансамбль', x=df_ensemble.index, y=df_ensemble['ensemble'],
                           mode=ml, marker=mark, line=dict(width=1, color='rgba(115, 175, 72, 0.7)'))
        fig.add_trace(trace, row=2, col=1)

        trace = go.Scatter(name=f'OIL: Доверит. интервал', x=df_ensemble.index, y=df_ensemble['interval_lower'],
                           mode='lines', line=dict(width=1, color='rgba(184, 247, 212, 0.7)'))
        fig.add_trace(trace, row=2, col=1)

        trace = go.Scatter(name=f'OIL: Доверит. интервал', x=df_ensemble.index, y=df_ensemble['interval_upper'],
                           fill='tonexty', mode='lines', line=dict(width=1, color='rgba(184, 247, 212, 0.7)'))
        fig.add_trace(trace, row=2, col=1)

        # Ошибка ансамбля
        deviation = compute_deviation(df_oil['true'], df_ensemble['ensemble'])
        trace = go.Scatter(name=f'OIL ERR: Ансамбль', x=deviation.index, y=deviation,
                           mode=ml, marker=mark, line=dict(width=1, color=colors[-3]))
        fig.add_trace(trace, row=3, col=1)

    # Дебит жидкости
    trace = go.Scatter(name=f'LIQ: {_MODEL_NAMES["true"]}', x=df_liq.index, y=df_liq['true'],
                       mode=m, marker=mark, line=dict(width=1, color=clr_fact))
    fig.add_trace(trace, row=1, col=1)
    for ind, col in enumerate(df_liq.columns):
        if col == 'true':
            continue
        trace = go.Scatter(name=f'LIQ: {_MODEL_NAMES[col]}', x=df_liq.index, y=df_liq[col],
                           mode=ml, marker=mark, line=dict(width=1, color=colors[ind]))
        fig.add_trace(trace, row=1, col=1)

    # Дебит нефти
    trace = go.Scatter(name=f'OIL: {_MODEL_NAMES["true"]}', x=df_oil.index, y=df_oil['true'],
                       mode=m, marker=mark, line=dict(width=1, color=clr_fact))
    fig.add_trace(trace, row=2, col=1)
    for ind, col in enumerate(df_oil.columns):
        if col == 'true':
            continue
        trace = go.Scatter(name=f'OIL: {_MODEL_NAMES[col]}', x=df_oil.index, y=df_oil[col],
                           mode=ml, marker=mark, line=dict(width=1, color=colors[ind]))
        fig.add_trace(trace, row=2, col=1)

    # Отклонения по моделям: дебит нефти
    for ind, col in enumerate(df_oil.columns):
        if col == 'true':
            continue
        deviation = compute_deviation(df_oil['true'], df_oil[col])
        trace = go.Scatter(name=f'OIL ERR: {_MODEL_NAMES[col]}', x=deviation.index, y=deviation,
                           mode=ml, marker=mark, line=dict(width=1, color=colors[ind]))
        fig.add_trace(trace, row=3, col=1)

    # Забойное давление
    trace = go.Scatter(name=f'Заб. давление', x=pressure.index, y=pressure,
                       mode=m, marker=dict(size=4, color=clr_pressure))
    fig.add_trace(trace, row=4, col=1)

    # Мероприятия
    _events = events.dropna()
    trace = go.Scatter(
        name='Мероприятие',
        x=_events.index,
        y=[0.2] * len(_events),
        mode='markers+text',
        marker=dict(size=8),
        text=_events.array,
        textposition='top center',
        textfont=dict(size=12),
    )
    fig.add_trace(trace, row=4, col=1)

    fig.add_vline(x=date_test, line_width=1, line_dash='dash')
    if not df_ensemble.empty:
        fig.add_vline(x=df_ensemble.index[0], line_width=1, line_dash='dash')

    return fig


def draw_histogram_model(df_err: pd.DataFrame,
                         bin_size: int,
                         oilfield: str,
                         ):
    length = len(df_err)
    days = [length // 3, 2 * length // 3]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            f'За {days[0]} суток',
            f'За {days[1]} суток',
            f'За весь период прогноза',
        ],
    )
    fig.layout.template = 'seaborn'
    # TODO: по добыче {чего}
    fig.update_layout(
        title_text=f'Распределение средней ошибки за n дней по добыче жидкости'
                   f'; {oilfield}; Скважин: {df_err.shape[1]}',
        bargap=0.005,
        font=dict(size=15),
        showlegend=False,
    )

    for ind, day in enumerate(days):
        x = df_err.iloc[:day].mean()
        fig.add_trace(
            go.Histogram(
                x=x,
                opacity=0.9,
                histnorm='percent',
                xbins=dict(
                    size=bin_size,
                ),
            ),
            row=ind + 1,
            col=1,
        )

        fig.update_xaxes(dtick=bin_size, row=ind + 1, col=1)
        fig.update_yaxes(title_text="Процент скважин", title_font_size=15, row=ind + 1, col=1)

    fig.update_xaxes(title_text="Усредненная относительная ошибка по добыче нефти, %",
                     title_font_size=16,
                     dtick=bin_size,
                     row=3, col=1)

    return fig


def draw_wells_model(df_err_model: pd.DataFrame):
    fig = make_subplots(
        rows=1,
        cols=1,
        vertical_spacing=0.05,
        subplot_titles=['Средняя относит. ошибка по добыче нефти на периоде прогноза, %']
    )
    fig.layout.template = 'seaborn'

    mean_err = df_err_model.mean(axis=0)
    mean_err = mean_err.sort_values()
    trace = go.Bar(x=mean_err.index, y=mean_err)
    fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(title_text="Номер скважины", row=1, col=1)
    fig.update_yaxes(title_text="Относит. ошибка, %", row=1, col=1)

    return fig


def draw_performance(dfs: dict,
                     df_perf: dict,
                     df_err: dict,
                     oilfield: str,
                     mode='oil'):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'Суммарная суточная добыча {mode}, м3',
                        'Относительное отклонение от факта, %'],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение {oilfield}'
    fig.update_layout(
        title=dict(text=text, x=0.05, xanchor='left'),
        font=dict(size=10),
        legend=dict(
            # orientation="h",
            font=dict(size=15)
        )
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # TODO: сейчас факт строится по всем моделям
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]
        x = df_perf[model].index
        trace = go.Scatter(name=f'факт_{model}', x=x, y=df_perf[model]['факт'],
                           mode=m, marker=mark, marker_color=clr)
        fig.add_trace(trace, row=1, col=1)

    # Model errors
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]
        x = df_perf[model].index
        trace1 = go.Scatter(name=model, x=x, y=df_perf[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name=f'', x=x, y=df_err[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr),
                            showlegend=False)

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

    return fig


def draw_statistics(
        models: list,
        model_mean: dict,
        model_std: dict,
        model_mean_daily: dict,
        model_std_daily: dict,
        oilfield: str,
        dates,
):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            'Относит. ошибка по накопленной добыче, %',
            'Стандартное отклонение по накопленной добыче, %',
            'Относит. ошибка суточной добычи, %',
            'Стандартное отклонение по суточной добыче, %',
        ],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение {oilfield} ; Добыча нефти, т'
    fig.update_layout(title=dict(text=text, x=0.05, xanchor='left'), font=dict(size=10))

    mark = dict(size=4)
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # Model errors
    for ind, model in enumerate(models):
        clr = colors[ind]
        trace1 = go.Scatter(name=model, x=dates, y=model_mean[model], mode=ml,
                            marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name='', x=dates, y=model_std[model], mode=ml,
                            marker=mark, line=dict(width=1, color=clr))
        trace3 = go.Scatter(name=f'', x=dates, y=model_mean_daily[model],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace4 = go.Scatter(name=f'', x=dates, y=model_std_daily[model],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)
        fig.add_trace(trace3, row=3, col=1)
        fig.add_trace(trace4, row=4, col=1)

    return fig
