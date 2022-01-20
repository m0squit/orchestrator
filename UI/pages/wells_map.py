import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statistics_explorer.plots import calc_relative_error
from UI.app_state import AppState


def show(session: st.session_state) -> None:
    state = session.state
    if not state.statistics:
        st.info('Здесь будет отображаться карта скважин, выбранных для расчета.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return
    df = prepare_data_for_plots(state)
    fig = select_plot(df)
    st.plotly_chart(fig, use_container_width=True)
    st.info('Справка по TreeMap:  \n'
            '**Цвет сектора** зависит от средней посуточной ошибки прогноза (модуль отклонения). '
            '**Размер сектора** зависит от накопленной добычи на периоде прогноза, [м3].  \n'
            'Надписи внутри каждого сектора идут в следующем порядке:'
            '  \n- имя скважины,  \n- накопленная добыча,  \n- посуточная ошибка на прогнозе.  \n')


def prepare_data_for_plots(state: AppState) -> pd.DataFrame:
    # TODO: возможно, разделить на два датафрейма отдельно для tree_plot и wells_map_plot
    columns = ['wellname', 'coord_x', 'coord_y', 'cum_q_liq', 'cum_q_oil', 'err_liq', 'err_oil']
    df = pd.DataFrame(columns=columns)
    models_without_ensemble = [model for model in state.statistics.keys() if model != 'ensemble']
    any_model_not_ensemble = models_without_ensemble[0]
    for well in state.wells_ftor:
        wellname_norm = state.wellnames_key_ois[well.well_name]
        cum_q_liq, cum_q_oil, err_liq, err_oil = None, None, pd.DataFrame(), pd.DataFrame()
        if f'{wellname_norm}_liq_true' in state.statistics[any_model_not_ensemble]:
            df_test_period = state.statistics[any_model_not_ensemble][state.was_date_test:]
            q_test_period = df_test_period[[f'{wellname_norm}_liq_true', f'{wellname_norm}_oil_true']]
            cum_q_liq, cum_q_oil = q_test_period.sum().round(1)
            err_liq = calc_relative_error(df_test_period[f'{wellname_norm}_liq_true'],
                                          df_test_period[f'{wellname_norm}_liq_pred'],
                                          use_abs=True)
            err_liq = round(err_liq.mean(), 1)
            err_oil = calc_relative_error(df_test_period[f'{wellname_norm}_oil_true'],
                                          df_test_period[f'{wellname_norm}_oil_pred'],
                                          use_abs=True)
            err_oil = round(err_oil.mean(), 1)
        new_row = wellname_norm, well.x_coord, well.y_coord, cum_q_liq, cum_q_oil, err_liq, err_oil
        df.loc[len(df)] = new_row
    return df


def select_plot(df: pd.DataFrame) -> go.Figure:
    selected_plot = st.selectbox(label='', options=['Карта скважин', 'TreeMap'])
    if selected_plot == 'Карта скважин':
        return create_wells_map_plot(df)
    if selected_plot == 'TreeMap':
        mode_dict = {'Нефть': 'oil', 'Жидкость': 'liq'}
        selected_mode = st.selectbox(label='Жидкость/нефть', options=sorted(mode_dict))
        return create_tree_plot(df, mode=selected_mode)


def create_wells_map_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(font=dict(size=15),
                      title_text=f'Карта скважин',
                      height=630,
                      width=1300,
                      separators='. ')
    fig.add_trace(go.Scatter(
        x=df['coord_x'],
        y=df['coord_y'],
        mode='markers+text',
        text=df['wellname'],
        textposition='top center',
        hovertext=df['cum_q_oil'],
        hoverinfo='all',
        # marker=m_inj,
        showlegend=False, ))
    return fig


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
                             parents=["Все скважины" for _ in df.wellname],
                             values=df[f'cum_q_{mode}'],
                             textinfo="text+label+value",
                             text=df['text_error'],
                             **{'marker_cmin': 0,
                                'marker_cmax': 100,
                                'marker_colors': df[f'err_{mode}'],
                                'marker_colorscale': 'oranges'},
                             ))
    return fig
