import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def create_tree_plot(df_treemap: pd.DataFrame):
    fig = go.Figure()
    fig.update_layout(font=dict(size=15),
                      title_text=f'hhhhh123hh',
                      height=630,
                      width=1300,)
    fig.add_trace(go.Treemap(labels=df_treemap.well,
                             parents=["Все скважины" for elem in df_treemap.well],
                             values=df_treemap.cumulative_q,
                             textinfo="text+label+value",
                             # root_color='lightgrey',
                             marker_colorscale='oranges',
                             ))
    fig.update_traces(root_color="lightgrey")
    return fig


def create_df_treemap(statistics_df_test: dict, selected_wells_norm: list):
    df = pd.DataFrame(columns=['well', 'cumulative_q'])
    df['well'] = selected_wells_norm
    for well_name in selected_wells_norm:
        for model in statistics_df_test:
            if f'{well_name}_oil_true' in statistics_df_test[model]:
                if well_name == '2012':
                    st.write(statistics_df_test[model][f'{well_name}_oil_true'])
                df['cumulative_q'][df.well == well_name] = np.sum(statistics_df_test[model][f'{well_name}_oil_true'])
                break
    st.write(df)
    return df


def show(session):
    if not session.selected_wells_norm:
        st.info('Здесь будет отображаться карта скважин, выбранных для расчета.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return
    df_treemap = create_df_treemap(session.statistics_df_test, session.selected_wells_norm)
    fig = create_tree_plot(df_treemap)
    st.plotly_chart(fig, use_container_width=True)

