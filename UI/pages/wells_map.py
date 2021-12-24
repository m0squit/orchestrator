import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


def prepare_data_for_plots(state: AppState) -> pd.DataFrame:
    columns = ['wellname', 'coord_x', 'coord_y', 'cum_q_liq', 'cum_q_oil']
    df = pd.DataFrame(columns=columns)
    models_without_ensemble = [model for model in state.statistics.keys() if model != 'ensemble']
    any_model_not_ensemble = models_without_ensemble[0]
    for well in state.wells_ftor:
        wellname_norm = state.wellnames_key_ois[well.well_name]
        cum_q_liq, cum_q_oil = None, None
        if f'{wellname_norm}_oil_true' in state.statistics[any_model_not_ensemble]:
            adapt_end_date = state.was_date_test - datetime.timedelta(days=1)
            df_adapt_period = state.statistics[any_model_not_ensemble][:adapt_end_date]
            q_adapt_period = df_adapt_period[[f'{wellname_norm}_liq_true', f'{wellname_norm}_oil_true']]
            cum_q_liq, cum_q_oil = q_adapt_period.sum().round(1)
        new_row = wellname_norm, well.x_coord, well.y_coord, cum_q_liq, cum_q_oil
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
                      title_text=f'Накопленная добыча на периоде адаптации: {mode}, м3',
                      height=630,
                      width=1300,
                      separators='. ')
    values = df.cum_q_liq
    if mode == 'Нефть':
        values = df.cum_q_oil
    fig.add_trace(go.Treemap(labels=df.wellname,
                             parents=["Все скважины" for _ in df.wellname],
                             values=values,
                             textinfo="text+label+value",
                             **{'marker_colorscale': 'oranges'},
                             ))
    return fig
