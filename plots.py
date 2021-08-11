import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def compute_deviations(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    devs = abs(y_true - y_pred) / y_true
    return devs


def create_well_plot(df_draw_liq,
                     df_draw_oil,
                     pressure,
                     date_test,
                     wellname=''):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        # TODO: в м3 или нет?
        subplot_titles=[
            'Дебит жидкости, м3',
            'Дебит нефти, м3',
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
    clr_fact = 'rgba(99, 110, 250, 0.8)'
    clr_pressure = '#C075A6'

    # Дебит жидкости
    for ind, col in enumerate(df_draw_liq.columns):
        clr = colors[ind]
        mode = ml
        if col == 'true':
            mode = m
            clr = clr_fact

        trace = go.Scatter(name=f'{col}', x=df_draw_liq.index, y=df_draw_liq[col],
                           mode=mode, marker=mark, line=dict(width=1, color=clr))
        fig.add_trace(trace, row=1, col=1)

    # Дебит нефти
    for ind, col in enumerate(df_draw_oil.columns):
        clr = colors[ind]
        mode = ml
        if col == 'true':
            mode = m
            clr = clr_fact

        trace = go.Scatter(name=f'{col}', x=df_draw_oil.index, y=df_draw_oil[col],
                           mode=mode, marker=mark, line=dict(width=1, color=clr))
        fig.add_trace(trace, row=2, col=1)

    # Забойное давление
    trace = go.Scatter(name=f'pressure', x=df_draw_oil.index, y=pressure[df_draw_oil.index],
                       mode=m, marker=dict(size=4, color=clr_pressure))
    fig.add_trace(trace, row=3, col=1)

    # 3, 2 test
    fig.add_trace(
        go.Scatter(
            name=name_dev_liq + '_пьезо',
            x=devs_liq_ftor.index,
            y=devs_liq_ftor,
            mode=ml, marker=mark,
        ), row=3, col=2)
    fig.add_trace(
        go.Scatter(
            name=name_dev_liq + '_ML',
            x=devs_liq_wolfram.index,
            y=devs_liq_wolfram,
            mode=ml, marker=mark,
        ), row=3, col=2)
    fig.add_trace(
        go.Scatter(
            name=name_dev_oil + '_пьезо',
            x=devs_oil_ftor.index,
            y=devs_oil_ftor,
            mode=ml, marker=mark,
        ), row=3, col=2)
    fig.add_trace(
        go.Scatter(
            name=name_dev_oil + '_ML',
            x=devs_oil_wolfram.index,
            y=devs_oil_wolfram,
            mode=ml, marker=mark,
        ), row=3, col=2)

    fig.add_vline(x=date_test, line_width=2, line_dash='dash')

    return fig
