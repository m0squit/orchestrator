import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


_MODEL_NAMES = {
    'ftor': 'Модель пьезо',
    'wolfram': 'Модель ML',
    'CRM': 'Модель CRM',
    'true': 'Фактический дебит',
}


def compute_deviations(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    devs = np.abs(y_true - y_pred) / np.maximum(y_true, y_pred) * 100
    return devs


def create_well_plot(df_draw_liq,
                     df_draw_oil,
                     pressure,
                     date_test,
                     wellname=''):
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
                    font=dict(size=15)
                    ),
        height=630,
        width=1100,
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Pastel
    clr_fact = 'rgba(99, 110, 250, 0.7)'
    clr_pressure = '#C075A6'

    # Дебит жидкости
    trace = go.Scatter(name=f'LIQ: {_MODEL_NAMES["true"]}', x=df_draw_liq.index, y=df_draw_liq['true'],
                       mode=m, marker=mark, line=dict(width=1, color=clr_fact))
    fig.add_trace(trace, row=1, col=1)

    for ind, col in enumerate(df_draw_liq.columns):
        if col == 'true':
            continue
        trace = go.Scatter(name=f'LIQ: {_MODEL_NAMES[col]}', x=df_draw_liq.index, y=df_draw_liq[col],
                           mode=ml, marker=mark, line=dict(width=1, color=colors[ind]))
        fig.add_trace(trace, row=1, col=1)

    # Дебит нефти
    trace = go.Scatter(name=f'OIL: {_MODEL_NAMES["true"]}', x=df_draw_oil.index, y=df_draw_oil['true'],
                       mode=m, marker=mark, line=dict(width=1, color=clr_fact))
    fig.add_trace(trace, row=2, col=1)

    for ind, col in enumerate(df_draw_oil.columns):
        if col == 'true':
            continue
        trace = go.Scatter(name=f'OIL: {_MODEL_NAMES[col]}', x=df_draw_oil.index, y=df_draw_oil[col],
                           mode=ml, marker=mark, line=dict(width=1, color=colors[ind]))
        fig.add_trace(trace, row=2, col=1)

    for ind, col in enumerate(df_draw_oil.columns):
        if col == 'true':
            continue
        deviations = compute_deviations(df_draw_oil['true'], df_draw_oil[col])
        trace = go.Scatter(name=f'OIL ERR: {_MODEL_NAMES[col]}', x=deviations.index, y=deviations,
                           mode=ml, marker=mark, line=dict(width=1, color=colors[ind]))
        fig.add_trace(trace, row=3, col=1)

    # Забойное давление
    trace = go.Scatter(name=f'Забойное давление', x=df_draw_oil.index, y=pressure[df_draw_oil.index],
                       mode=m, marker=dict(size=4, color=clr_pressure))
    fig.add_trace(trace, row=4, col=1)

    fig.add_vline(x=date_test, line_width=2, line_dash='dash')

    return fig
