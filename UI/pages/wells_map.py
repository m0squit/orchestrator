import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def show(session):
    state = session.state
    if not state.selected_wells_norm:
        st.info('Здесь будет отображаться карта скважин, выбранных для расчета.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return
    df_treemap = create_df_treemap(state.statistics_test_only, state.selected_wells_norm)
    fig = create_tree_plot(df_treemap)
    st.plotly_chart(fig, use_container_width=True)


def create_tree_plot(df_treemap: pd.DataFrame):
    fig = go.Figure()
    fig.update_layout(font=dict(size=15),
                      title_text=f'Накопленная добыча нефти, м3',
                      height=630,
                      width=1300,)
    fig.add_trace(go.Treemap(labels=df_treemap.well,
                             parents=["Все скважины" for elem in df_treemap.well],
                             values=df_treemap.cumulative_q,
                             textinfo="text+label+value",
                             # root_color='lightgrey',
                             **{'marker_colorscale': 'oranges'},
                             ))
    fig.update_traces(root_color="lightgrey")
    return fig


def create_df_treemap(statistics_test_only: dict, selected_wells_norm: list):
    df = pd.DataFrame(columns=['well', 'cumulative_q'])
    df['well'] = selected_wells_norm
    for well_name in selected_wells_norm:
        for model in statistics_test_only:
            if f'{well_name}_oil_true' in statistics_test_only[model]:
                if well_name == '2012':
                    st.write(statistics_test_only[model][f'{well_name}_oil_true'])
                df['cumulative_q'][df.well == well_name] = np.sum(statistics_test_only[model][f'{well_name}_oil_true'])
                break
    st.write(df)
    return df
