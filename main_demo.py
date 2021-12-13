import datetime
import pandas as pd
from frameworks_ftor.ftor.config import Config as ConfigFtor
from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from pathlib import Path
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from timeit import default_timer
import os

from config import Config as ConfigPreprocessor
from preprocessor import Preprocessor
from frameworks_ftor.ftor.well import Well as WellFtor


def _create_trans_plot(well_name, df_chess, rates, date_test, adap_and_fixed_params, path, is_liq):
    name = 'liq' if is_liq else 'oil'
    figure = go.Figure(layout=go.Layout(
        font=dict(size=10),
        hovermode='x',
        template='seaborn',
        title=dict(text=f'{well_name} {name}', x=0.05, xanchor='left'),
    ))
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.05, 0.6, 0.35], figure=figure
    )
    fig.update_xaxes(dtick='M1', tickformat="%b\n%Y")
    fig.update_yaxes(row=1, col=1, range=[0, 1], showticklabels=False)

    m = 'markers'
    ml = 'markers+lines'
    mark = dict(size=3)
    line = dict(width=1)

    s = df_chess[WellFtor.NAME_EVENT].dropna()
    trace_a = go.Scatter(
        name=WellFtor.NAME_EVENT,
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
    if is_liq:
        dir_path = path / 'liq'
        x_model = x
        name_fact = WellFtor.NAME_RATE_LIQ
        name_model = 'Дебит жидкости модельный'
        name_graph = 'liq'
    else:
        dir_path = path / 'oil'
        x_model = rates.index.to_list()
        name_fact = WellFtor.NAME_RATE_OIL
        name_model = 'Дебит нефти модельный'
        name_graph = 'oil'

    fig.add_trace(go.Scatter(name=name_fact, x=x, y=df_chess[name_fact], mode=m, marker=mark), row=2, col=1)
    fig.add_trace(go.Scatter(name=name_model, x=x_model, y=rates, mode=ml, marker=mark, line=line), row=2, col=1)
    fig.add_trace(go.Scatter(name='Pз', x=x, y=df_chess[WellFtor.NAME_PRESSURE], mode=m, marker=mark), row=3, col=1)
    fig.add_vline(x=date_test, line_width=2, line_dash='dash')

    text = 'params:'
    for ad_prd_prms in adap_and_fixed_params.copy():
        text += '<br>' + str({name: round(ad_prd_prms[name], 1) for name in ad_prd_prms})

    fig.add_annotation(showarrow=False, text=text, xref='paper', yref='paper', x=0.5, y=1.175)

    if not dir_path.exists():
        os.mkdir(dir_path)
    file = f'{str(dir_path)}\\{well_name} {name_graph}.png'
    plotly.io.write_image(fig, file=file, width=1450, height=700, scale=2, engine='kaleido')


fields = [
    'Крайнее',
    # 'Вынгаяхинское'
]

shops_lst = [
    ['ЦДНГ-4'],
    # ['ЦДНГ-10']
]

dates_start = [
    datetime.date(2018, 1, 1),
    # datetime.date(2018, 1, 1)
]

dates_test = [
    datetime.date(2018, 11, 1),
    # datetime.date(2019, 1, 1)
]

dates_end = [
    datetime.date(2019, 1, 31),
    # datetime.date(2019, 3, 31)
]

use_eq_t = [
    False,
    # True
]
data = zip(fields, shops_lst, dates_start, dates_test, dates_end, use_eq_t)

for field_name, shops, date_start, date_test, date_end, use_eq_t in data:
    start = default_timer()
    name_dir = field_name + " " + str(datetime.datetime.now()).replace(':', '-')
    path = Path.cwd() / 'tests' / name_dir
    if not path.exists():
        os.mkdir(path)
    df_hypotheses = pd.DataFrame()

    preprocessor = Preprocessor(
        ConfigPreprocessor(
            field_name,
            shops,
            date_start,
            date_test,
            date_end,
        )
    )

    for well_name in preprocessor.well_names:
        try:
            if well_name == 2560204400:
                data_ftor = preprocessor.create_wells_ftor(
                    [well_name],
                    # user_constraints_for_adap_period = {
                    #     'permeability': 1.2,
                    #     'skin': 1.7,
                    #     'res_width': 636,
                    #     'res_length': 144,
                    #     'pressure_initial': 1000,
                    #     'boundary_code': 3}
                )[0]
                calculator_ftor = CalculatorFtor(
                    ConfigFtor(
                        use_equal_time_algorithm=use_eq_t,
                        apply_last_points_adaptation=False,
                    ),
                    [data_ftor],
                    df_hypotheses)
                df_chess = data_ftor.df_chess
                res_ftor = calculator_ftor.wells[0].results

                adap_and_fixed_params = res_ftor.adap_and_fixed_params
                rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
                rates_oil_ftor = res_ftor.rates_oil_test

                df = pd.DataFrame()
                df[f'{well_name}_oil_true'] = df_chess['Дебит нефти'].loc[df_chess.index >= date_test]
                df[f'{well_name}_oil_pred'] = rates_oil_ftor
                df[f'{well_name}_liq_true'] = df_chess['Дебит жидкости'].loc[df_chess.index >= date_test]
                df[f'{well_name}_liq_pred'] = rates_liq_ftor.loc[df_chess.index >= date_test]
                df.to_excel(path / f'{well_name}.xlsx')

                _create_trans_plot(well_name, df_chess, rates_liq_ftor, date_test,
                                   adap_and_fixed_params, path, is_liq=True)
                _create_trans_plot(well_name, df_chess, rates_oil_ftor, date_test,
                                   adap_and_fixed_params, path, is_liq=False)
        except Exception as exc:
            file = open(path / f'{well_name} error.txt', 'w')
            file.write(str(exc))
            file.close()

    exec_time = default_timer() - start
    file = open(path / 'test_data.txt', 'w')
    print(f'{exec_time = } с', file=file)
    print(f'{field_name = }', file=file)
    print(f'{shops = }', file=file)
    print(f'{date_start = }', file=file)
    print(f'{date_test = }', file=file)
    print(f'{date_end = }', file=file)
    file.close()
    df_hypotheses.to_excel(path / 'df_hypotheses.xlsx')
