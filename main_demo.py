import datetime
import pandas as pd
from frameworks_ftor.ftor.config import Config as ConfigFtor
from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from pathlib import Path
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from timeit import default_timer

from config import Config
from preprocessor import Preprocessor


def _create_trans_plot(well_name, df_chess, rates_liq_ftor, date_test, adap_and_fixed_params) -> go.Figure:
    figure = go.Figure(layout=go.Layout(
        font=dict(size=10),
        hovermode='x',
        template='seaborn',
        title=dict(text=f'Скважина {well_name}', x=0.05, xanchor='left'),
    ))
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.05, 0.6, 0.35],
        figure=figure,
    )
    fig.update_xaxes(dtick='M1', tickformat="%b\n%Y")
    fig.update_yaxes(row=1, col=1, range=[0, 1], showticklabels=False)

    m = 'markers'
    ml = 'markers+lines'
    mark = dict(size=3)
    line = dict(width=1)

    s = df_chess['Мероприятие'].dropna()
    trace_a = go.Scatter(
        name='Мероприятие',
        x=s.index,
        y=[0.2] * len(s),
        mode='markers+text',
        marker=dict(size=8),
        text=s.array,
        textposition='top center',
        textfont=dict(size=8),
    )
    fig.add_trace(trace_a, row=1, col=1)

    x = df_chess.index.to_list()

    trace = go.Scatter(name='q_ж_факт', x=x, y=df_chess['Дебит жидкости'], mode=m, marker=mark)
    fig.add_trace(trace, row=2, col=1)

    trace = go.Scatter(name='q_ж_модель', x=x, y=rates_liq_ftor, mode=ml, marker=mark, line=line)
    fig.add_trace(trace, row=2, col=1)

    trace = go.Scatter(name='p_з_факт', x=x, y=df_chess['Давление забойное'], mode=m, marker=mark)
    fig.add_trace(trace, row=3, col=1)

    fig.add_vline(x=date_test, line_width=2, line_dash='dash')

    text = 'params:'
    for i in adap_and_fixed_params:
        for name, value in i.items():
            i[name] = round(value, 1)
        text += f'<br>{i}'
    fig.add_annotation(
        showarrow=False,
        text=text,
        xref='paper',
        yref='paper',
        x=0.5,
        y=1.175,
    )
    return fig


start = default_timer()
field_name = 'Крайнее'
shops = ['ЦДНГ-4']
date_start = datetime.date(2018, 1, 1)
date_test = datetime.date(2018, 11, 1)
date_end = datetime.date(2019, 1, 31)

preprocessor = Preprocessor(
    Config(
        field_name,
        shops,
        date_start,
        date_test,
        date_end,
    )
)

for well_name in preprocessor.well_names:
    if well_name == 2560209600:
        data_ftor = preprocessor.create_wells_ftor(
            [well_name],
            {
                # 'length_hor_well_bore': 100,
                # "skin": 0.1,
                # 'pressure_initial': 700,
                # 'permeability': {'is_discrete': False, 'bounds': [0, 0.4]}
                }
        )[0]
        calculator_ftor = CalculatorFtor(ConfigFtor(), [data_ftor])
        res_ftor = calculator_ftor.wells[0].results

        pressure = data_ftor.df_chess['Давление забойное']
        df_draw_liq = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))
        df_draw_oil = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='D'))
        df_draw_liq['ftor'] = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
        df_draw_liq['true'] = data_ftor.df_chess['Дебит жидкости']
        df_draw_oil['ftor'] = res_ftor.rates_oil_test
        df_draw_oil['true'] = data_ftor.df_chess['Дебит нефти']
        rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
        f = _create_trans_plot(well_name, data_ftor.df_chess, rates_liq_ftor, date_test, res_ftor.adap_and_fixed_params)

        path_str = str(Path.cwd() / 'test_graphs')
        time_str = str(datetime.datetime.now()).replace(':', '-')
        file = f'{path_str}\\{well_name} new.png'
        plotly.io.write_image(f, file=file, width=1450, height=700, scale=2, engine='kaleido')
print('time =', default_timer() - start)
