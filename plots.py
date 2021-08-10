import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def compute_deviations(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    devs = abs(y_true - y_pred) / y_true
    return devs


def create_well_plot(well_ftor, well_wolfram, date_test):
    res_ftor = well_ftor.results
    res_wolfram = well_wolfram.results

    # Фактические данные для визуализации извлекаются из wolfram, т.к. он использует для вычислений максимально
    # возможный доступный ряд фактичесих данных.
    name_rate_liq = 'q_жид'
    name_rate_oil = 'q_неф'
    name_pressure = 'p_заб'
    name_dev_liq = 're_жид'
    name_dev_oil = 're_неф'

    df = well_wolfram.df
    rates_liq_true = df[well_wolfram.NAME_RATE_LIQ]
    rates_oil_true = df[well_wolfram.NAME_RATE_OIL]
    pressure = df[well_wolfram.NAME_PRESSURE]

    # Жидкость
    # Полный ряд (train + test)
    rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
    rates_liq_wolfram = pd.concat(objs=[res_wolfram.rates_liq_train, res_wolfram.rates_liq_test])
    # test
    rates_liq_test_ftor = res_ftor.rates_liq_test
    rates_liq_test_wolfram = res_wolfram.rates_liq_test
    rates_liq_test_true = rates_liq_true.loc[rates_liq_test_wolfram.index]
    # devs
    devs_liq_ftor = compute_deviations(rates_liq_test_true, rates_liq_test_ftor)
    devs_liq_wolfram = compute_deviations(rates_liq_test_true, rates_liq_test_wolfram)

    # Нефть
    # Полный ряд (train + test)
    rates_oil_wolfram = pd.concat(objs=[res_wolfram.rates_oil_train, res_wolfram.rates_oil_test])  # Только нефть
    # test
    rates_oil_test_ftor = res_ftor.rates_oil_test
    rates_oil_test_wolfram = res_wolfram.rates_oil_test
    rates_oil_test_true = rates_oil_true.loc[rates_oil_test_wolfram.index]
    # devs
    devs_oil_ftor = compute_deviations(rates_liq_test_true, rates_oil_test_ftor)
    devs_oil_wolfram = compute_deviations(rates_oil_test_true, rates_oil_test_wolfram)

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        column_width=[0.7, 0.3],
        row_heights=[0.4, 0.4, 0.2],
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        column_titles=[
            'Адаптация и прогноз',
            'Прогноз',
        ],
        figure=go.Figure(
            layout=go.Layout(
                font=dict(size=10),
                hovermode='x',
                template='seaborn',
                height=650,
                width=1000,
                legend=dict(orientation="h",
                            font=dict(size=10)
                            ),
            ),
        ),
    )

    m = 'markers'
    ml = 'markers+lines'
    mark = dict(size=3)

    # 1, 1 Полный ряд (train + test)
    fig.add_trace(
        go.Scatter(
            name=name_rate_liq + '_факт',
            x=rates_liq_true.index,
            y=rates_liq_true,
            mode=m, marker=mark,
        ), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            name=name_rate_liq + '_пьезо',
            x=rates_liq_ftor.index,
            y=rates_liq_ftor,
            mode=ml, marker=mark,
        ), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            name=name_rate_liq + '_ML',
            x=rates_liq_wolfram.index,
            y=rates_liq_wolfram,
            mode=ml, marker=mark,
        ), row=1, col=1)

    # 1, 2 test
    fig.add_trace(
        go.Scatter(
            name=name_rate_liq + '_факт',
            x=rates_liq_test_true.index,
            y=rates_liq_test_true,
            mode=m, marker=mark,
        ), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            name=name_rate_liq + '_пьезо',
            x=rates_liq_test_ftor.index,
            y=rates_liq_test_ftor,
            mode=ml, marker=mark,
        ), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            name=name_rate_liq + '_ML',
            x=rates_liq_test_wolfram.index,
            y=rates_liq_test_wolfram,
            mode=ml, marker=mark,
        ), row=1, col=2)

    # 2, 1 Полный ряд (train + test)
    fig.add_trace(
        go.Scatter(
            name=name_rate_oil + '_факт',
            x=rates_oil_true.index,
            y=rates_oil_true,
            mode=m, marker=mark,
        ), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            name=name_rate_oil + '_ML',
            x=rates_oil_wolfram.index,
            y=rates_oil_wolfram,
            mode=ml, marker=mark,
        ), row=2, col=1)

    # 2, 2 test
    fig.add_trace(
        go.Scatter(
            name=name_rate_oil + '_факт',
            x=rates_oil_test_true.index,
            y=rates_oil_test_true,
            mode=m, marker=mark,
        ), row=2, col=2)
    fig.add_trace(
        go.Scatter(
            name=name_rate_oil + '_пьезо',
            x=rates_oil_test_ftor.index,
            y=rates_oil_test_ftor,
            mode=ml, marker=mark,
        ), row=2, col=2)
    fig.add_trace(
        go.Scatter(
            name=name_rate_oil + '_ML',
            x=rates_oil_test_wolfram.index,
            y=rates_oil_test_wolfram,
            mode=ml, marker=mark,
        ), row=2, col=2)

    # 3, 1 Полный ряд (train + test)
    fig.add_trace(
        go.Scatter(
            name=name_pressure + '_факт',
            x=pressure.index,
            y=pressure,
            mode=m, marker=mark,
        ), row=3, col=1)

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
    # fig.add_vline(row=2, col=1, x=date_test, line_width=2, line_dash='dash')
    # fig.add_vline(row=3, col=1, x=date_test, line_width=2, line_dash='dash')

    return fig
