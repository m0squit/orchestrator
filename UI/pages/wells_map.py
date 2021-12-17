import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def show(session):
    state = session.state
    if not state.statistics:
        st.info('Здесь будет отображаться карта скважин, выбранных для расчета.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return
    mode_dict = {'Нефть': 'oil', 'Жидкость': 'liq'}
    selected_mode = st.selectbox(label='Жидкость/нефть', options=sorted(mode_dict))
    df_treemap = create_df_treemap(statistics=state.statistics,
                                   date_test=state.was_date_test,
                                   selected_wells_norm=state.selected_wells_norm,
                                   mode=mode_dict[selected_mode])
    fig = create_tree_plot(df_treemap, mode=selected_mode)
    st.plotly_chart(fig, use_container_width=True)


def create_df_treemap(statistics: dict,
                      date_test: datetime.date,
                      selected_wells_norm: list,
                      mode: str):
    df = pd.DataFrame(columns=['well', 'cumulative_q'])
    df['well'] = selected_wells_norm
    models_without_ensemble = [model for model in statistics.keys() if model != 'ensemble']
    any_model_not_ensemble = models_without_ensemble[0]
    for well_name in selected_wells_norm:
        if f'{well_name}_{mode}_true' in statistics[any_model_not_ensemble]:
            df_adapt_period = statistics[any_model_not_ensemble][:date_test - datetime.timedelta(days=1)]
            q_adapt_period = df_adapt_period[f'{well_name}_{mode}_true']
            df['cumulative_q'][df.well == well_name] = np.sum(q_adapt_period)
    return df


def create_tree_plot(df_treemap: pd.DataFrame, mode: str):
    fig = go.Figure()
    fig.update_layout(font=dict(size=15),
                      title_text=f'Накопленная добыча на периоде адаптации: {mode}, м3',
                      height=630,
                      width=1300,)
    fig.add_trace(go.Treemap(labels=df_treemap.well,
                             parents=["Все скважины" for elem in df_treemap.well],
                             values=df_treemap.cumulative_q,
                             textinfo="text+label+value",
                             **{'marker_colorscale': 'oranges'},
                             ))
    return fig
