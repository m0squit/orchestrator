import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import n_colors
import streamlit as st
from typing import Tuple, List

from UI.pages.analytics import select_wells_set, draw_form_exclude_wells
from statistics_explorer.plots import calc_relative_error
from statistics_explorer.config import ConfigStatistics
from UI.app_state import AppState


def show(session: st.session_state) -> None:
    state = session.state
    if not state.statistics:
        st.info('Здесь будет отображаться карта скважин, выбранных для расчета.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return
    selected_wells_set = select_wells_set(state)
    fig, selected_plot, fig_table = select_plot(state, selected_wells_set)
    st.plotly_chart(fig, use_container_width=True)
    if selected_plot == 'Карта скважин':
        st.plotly_chart(fig_table, use_container_width=True)
    if selected_plot == 'TreeMap':
        draw_form_exclude_wells(state, selected_wells_set)
        st.info('Справка по TreeMap:  \n'
                '**Цвет сектора** зависит от средней посуточной ошибки на периоде прогноза (модуль отклонения).  \n'
                '**Размер сектора** зависит от накопленной добычи на периоде прогноза, [м3].  \n'
                'Надписи внутри каждого сектора идут в следующем порядке:'
                '  \n- имя скважины,  \n- накопленная добыча,  \n- посуточная ошибка на прогнозе.  \n')


def select_plot(state: AppState, selected_wells_set: Tuple[str, ...]) -> [go.Figure, str]:
    selected_plot = st.selectbox(label='', options=['Карта скважин', 'TreeMap'])
    if selected_plot == 'Карта скважин':
        mode_well = select_well(state.coeff_f.columns)
        coords_df, f_dict = crm_map(state.coeff_f, state.wells_coords_CRM)
        df_f_coeff = plot_influence_table(state.coeff_f, mode_well)
        return crm_plot(coords_df, f_dict, mode_well, state.CRM_influence_R), selected_plot, df_f_coeff
    if selected_plot == 'TreeMap':
        mode_dict = {'Нефть': 'oil', 'Жидкость': 'liq'}
        mode = st.selectbox(label='Жидкость/нефть', options=sorted(mode_dict))
        model_for_error = select_model(state)
        df = prepare_data_for_treemap(state, model_for_error, selected_wells_set)
    return create_tree_plot(df, mode=mode), selected_plot, None


# def prepare_data_for_wells_map(state: AppState) -> pd.DataFrame:
#     columns = ['wellname', 'coord_x', 'coord_y']
#     df = pd.DataFrame(columns=columns)
#     for well in state.wells_ftor:
#         wellname_norm = state.wellnames_key_ois[well.well_name]
#         df.loc[len(df)] = wellname_norm, well.x_coord, well.y_coord
#     return df


def select_model(state: AppState) -> str:
    MODEL_NAMES = ConfigStatistics.MODEL_NAMES
    MODEL_NAMES_REVERSED = {v: k for k, v in MODEL_NAMES.items()}
    models_without_ensemble = [MODEL_NAMES[model] for model in state.statistics.keys() if model != 'ensemble']
    models_without_ensemble.insert(0, 'Ансамбль')
    model = st.selectbox(label="Модель для расчета ошибки:",
                         options=models_without_ensemble)
    return MODEL_NAMES_REVERSED[model]


def prepare_data_for_treemap(state: AppState, model: str, selected_wells_set: Tuple[str, ...]) -> pd.DataFrame:
    columns = ['wellname', 'cum_q_liq', 'cum_q_oil', 'err_liq', 'err_oil']
    df = pd.DataFrame(columns=columns)
    well_names = [
        elem for elem in selected_wells_set if elem not in state.exclude_wells
    ]
    for well in well_names:
        cum_q_liq, cum_q_oil, err_liq, err_oil = None, None, pd.DataFrame(), pd.DataFrame()
        if f'{well}_liq_true' in state.statistics[model]:
            df_test_period = state.statistics[model][state.was_date_test:]
            q_test_period = df_test_period[[f'{well}_liq_true', f'{well}_oil_true']]
            cum_q_liq, cum_q_oil = q_test_period.sum().round(1)
            err_liq = calc_relative_error(df_test_period[f'{well}_liq_true'],
                                          df_test_period[f'{well}_liq_pred'],
                                          use_abs=True)
            err_liq = round(err_liq.mean(), 1)
            err_oil = calc_relative_error(df_test_period[f'{well}_oil_true'],
                                          df_test_period[f'{well}_oil_pred'],
                                          use_abs=True)
            err_oil = round(err_oil.mean(), 1)
        df.loc[len(df)] = well, cum_q_liq, cum_q_oil, err_liq, err_oil
    return df


# def create_wells_map_plot(df: pd.DataFrame) -> go.Figure:
#     fig = go.Figure()
#     fig.update_layout(font=dict(size=15),
#                       title_text=f'Карта скважин',
#                       height=630,
#                       width=1300,
#                       separators='. ')
#     fig.add_trace(go.Scatter(
#         x=df['coord_x'],
#         y=df['coord_y'],
#         mode='markers+text',
#         text=df['wellname'],
#         textposition='top center',
#         # hovertext=df['cum_q_oil'],
#         hoverinfo='all',
#         showlegend=False, ))
#     return fig


def create_tree_plot(df: pd.DataFrame, mode: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(font=dict(size=15),
                      title_text=f'Treemap',
                      height=630,
                      width=1300,
                      separators='. ')
    mode_dict = {'Нефть': 'oil', 'Жидкость': 'liq'}
    mode = mode_dict[mode]
    df['text_error'] = 'Ошибка: ' + df[f'err_{mode}'].apply(str) + '%'
    fig.add_trace(go.Treemap(labels=df['wellname'],
                             parents=["Все скважины" for _ in df['wellname']],
                             values=df[f'cum_q_{mode}'],
                             textinfo="text+label+value",
                             text=df['text_error'],
                             **{'marker_cmin': 0,
                                'marker_cmax': 100,
                                'marker_colors': df[f'err_{mode}'],
                                'marker_colorscale': 'oranges'},
                             ))
    return fig

def select_well(produced_wells: List):
    well_chosen = st.selectbox(label="Скважина:",
                               options=list(produced_wells), #['', 'Все скважины'] +
                               key='selected_wells_for_influence')
    return well_chosen

def crm_map(f_values, coords_df, border=0.):
    '''
    f_values - матрица взаимовлияния
       |
       |
       v
    Dataframe n x m, n - строки, названия нагнетательных скважин:str, m - столбцы, названия добывающих скважин:str

    coords_df - координаты
       |
       |
       v
    Dataframe с 3 колонками: 'Скважина':str, 'Координата X':float, 'Координата Y':float

    prod_df - добывающие скважины
       |
       |
       v
    Dataframe с 3 колонками: 'Скважина':str, 'Дата':pd.Timestamp, 'Дебит нефти':float

    inj_df - нагнетательные скважины
       |
       |
       v
    Dataframe с 3 колонками: 'Скважина':str, 'Дата':pd.Timestamp, 'Приемистость':float

    mult - регулирует толщину линий на карте
    '''

    def wut_trans(x):
        l = [str(i) for i in x.values]
        l.sort()
        return tuple(l)

    coords_df = coords_df[['Координата X', 'Координата Y']].reset_index()
    f_values = f_values[list(set(f_values.columns) - set(f_values.index))]
    inj_wells_list = f_values.index.tolist()
    prod_wells_list = f_values.columns.tolist()
    # !!!!!!
    # f_values = f_values[f_values > 0]

    f_values = f_values[f_values > border]

    f_values = f_values + 1.5
    f_values = f_values.stack().reset_index()
    f_values['wut'] = f_values.apply(lambda x: wut_trans(x), axis=1)
    f_values = f_values.drop_duplicates(subset='wut', keep='first').drop('wut', axis=1)
    f_dict = f_values.set_index(['level_0', 'Well']).to_dict()[0]


    coords_df['Тип'] = coords_df['Скважина'].apply(lambda x: 'Добывающая' if x in prod_wells_list else \
        ('Нагнетательная' if x in inj_wells_list else np.NaN))
    coords_df['value'] = 0.

    return coords_df, f_dict


def crm_plot(coords_df, f_dict, mode:str, influence_R:int):
    _df = pd.DataFrame(columns=['x', 'y', 'value'])
    count = 0
    fig = go.Figure()

    ht_inj, m_inj, ht_prod, m_prod = None, None, None, None

    fig.add_trace(go.Scatter(
        x=coords_df[coords_df['Тип'] == 'Нагнетательная']['Координата X'].tolist(),
        y=coords_df[coords_df['Тип'] == 'Нагнетательная']['Координата Y'].tolist(),
        mode='markers',
        name='Нагнетательная скважина',
        text=coords_df[coords_df['Тип'] == 'Нагнетательная']['Скважина'].tolist(),
        textposition='top center',
        hovertext=ht_inj,
        hoverinfo='text',
        marker=m_inj,
        showlegend=True, ))

    fig.add_trace(go.Scatter(
        x=coords_df[coords_df['Тип'] == 'Добывающая']['Координата X'].tolist(),
        y=coords_df[coords_df['Тип'] == 'Добывающая']['Координата Y'].tolist(),
        mode='markers',
        name='Добывающая скважина',
        text=coords_df[coords_df['Тип'] == 'Добывающая']['Скважина'].tolist(),
        textposition='top center',
        hovertext=ht_prod,
        hoverinfo='text',
        marker=m_prod,
        showlegend=True))
    influence_count = 0
    for key, value in f_dict.items():
        if key[1] == mode:
            influence_count +=1
    influence_count += 1 if influence_count == 1 else 0
    colorscale = n_colors('rgb(0, 0, 0)', 'rgb(0, 255, 0)', influence_count, colortype='rgb')
    for key, value in f_dict.items():
        if key[1] == mode:
            _x = [coords_df[coords_df['Скважина'] == key[0]]['Координата X'].values[0],
                  coords_df[coords_df['Скважина'] == key[1]]['Координата X'].values[0]]
            _y = [coords_df[coords_df['Скважина'] == key[0]]['Координата Y'].values[0],
                  coords_df[coords_df['Скважина'] == key[1]]['Координата Y'].values[0]]

            fig.add_trace(go.Scatter(x=_x, y=_y, mode='lines',
                                     line=go.scatter.Line(color=colorscale[count], width=value),
                                     showlegend=False))
            _df.loc[count] = [np.mean(_x), np.mean(_y), value - 1.5]
            count += 1

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  fillcolor="rgb(255,0,0)",
                  x0=coords_df[coords_df['Скважина'] == mode]['Координата X'].values[0] - influence_R,
                  y0=coords_df[coords_df['Скважина'] == mode]['Координата Y'].values[0] - influence_R,
                  x1=coords_df[coords_df['Скважина'] == mode]['Координата X'].values[0] + influence_R,
                  y1=coords_df[coords_df['Скважина'] == mode]['Координата Y'].values[0] + influence_R,
                  opacity=0.1,
                  line_color="rgb(255,0,0)",
                  )

    fig.add_trace(go.Scatter(
        x=_df['x'].tolist(),
        y=_df['y'].tolist(),
        mode='markers',
        opacity=0.2,
        text=[round(i, 3) for i in _df['value'].tolist()],
        hovertext=_df['value'].tolist(),
        hoverinfo='text',
        textposition='top center',
        marker=dict(color='black', size=5, opacity=.6),
        showlegend=False))

    fig.update_layout(
        # title='Взаимовлияние скважин',
        xaxis_title='X координата',
        yaxis_title='Y координата',
        height=630,
        width=1300,
        legend = dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=1,
        bgcolor='rgba(0,0,0,0)'))


    return fig

def plot_influence_table(df_table, mode):
    # if mode == "":
    #     return None
    # elif mode == "Все скважины":
    #     mode = df.columns.values()
    df_table = df_table[df_table>0].round(3)
    df_table.index = df_table.index.astype(str)
    inj_wells = []
    for w in df_table[mode].dropna().index:
        inj_wells.append(f'<b>{w}<b>')
    fig = go.Figure(data=[go.Table(header=dict(values=['', f'<b>{mode}<b>']),
                                   cells=dict(values=[inj_wells, df_table[mode].dropna()],
                                              align='center', font=dict(color='black')),
                                   )
                          ])
    return fig
