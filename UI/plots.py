import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


_MODEL_NAMES = {
    'ftor': 'Пьезо',
    'wolfram': 'ML',
    'CRM': 'CRM',
    'true': 'Факт',
}


def compute_deviation(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    devs = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), np.abs(y_pred)) * 100
    return devs


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
