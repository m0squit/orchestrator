import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

from statistics_explorer.config import ConfigStatistics
from statistics_explorer.main import calculate_statistics

session = st.session_state


def prepare_data_for_statistics(
        df_draw_liq,
        df_draw_oil,
        df_draw_ensemble,
        statistics,
        selected_wells_ois,
        was_calc_ftor,
        was_calc_wolfram,
        was_calc_ensemble
):
    session.dates = pd.date_range(session.date_test, session.date_end, freq='D')
    if was_calc_ftor:
        statistics['ftor'] = pd.DataFrame(index=session.dates)
        for _well_name in selected_wells_ois:
            if 'ftor' in df_draw_liq[_well_name] and 'ftor' in df_draw_oil[_well_name]:
                statistics['ftor'][f'{_well_name}_liq_true'] = df_draw_liq[_well_name]['true']
                statistics['ftor'][f'{_well_name}_liq_pred'] = df_draw_liq[_well_name]['ftor']
                statistics['ftor'][f'{_well_name}_oil_true'] = df_draw_oil[_well_name]['true']
                statistics['ftor'][f'{_well_name}_oil_pred'] = df_draw_oil[_well_name]['ftor']

    if was_calc_wolfram:
        statistics['wolfram'] = pd.DataFrame(index=session.dates)
        for _well_name in selected_wells_ois:
            if 'wolfram' in df_draw_liq[_well_name] and 'wolfram' in df_draw_oil[_well_name]:
                statistics['wolfram'][f'{_well_name}_liq_true'] = df_draw_liq[_well_name]['true']
                statistics['wolfram'][f'{_well_name}_liq_pred'] = df_draw_liq[_well_name]['wolfram']
                statistics['wolfram'][f'{_well_name}_oil_true'] = df_draw_oil[_well_name]['true']
                statistics['wolfram'][f'{_well_name}_oil_pred'] = df_draw_oil[_well_name]['wolfram']

    if 'df_CRM' in session:
        for _well_name in selected_wells_ois:
            if _well_name in session['df_CRM'].columns:
                if 'CRM' not in statistics:
                    statistics['CRM'] = pd.DataFrame(index=session.dates)
                statistics['CRM'][f'{_well_name}_liq_true'] = np.nan
                statistics['CRM'][f'{_well_name}_liq_pred'] = np.nan
                statistics['CRM'][f'{_well_name}_oil_true'] = df_draw_oil[_well_name]['true']
                statistics['CRM'][f'{_well_name}_oil_pred'] = df_draw_oil[_well_name]['CRM']

    if was_calc_ensemble and (was_calc_ftor or was_calc_wolfram):
        statistics['ensemble'] = pd.DataFrame(index=session.dates)
        for _well_name in selected_wells_ois:
            if 'ensemble' in df_draw_ensemble[_well_name]:
                statistics['ensemble'][f'{_well_name}_liq_true'] = np.nan
                statistics['ensemble'][f'{_well_name}_liq_pred'] = np.nan
                statistics['ensemble'][f'{_well_name}_oil_true'] = df_draw_oil[_well_name]['true']
                statistics['ensemble'][f'{_well_name}_oil_pred'] = df_draw_ensemble[_well_name]['ensemble']
        # TODO: обрезка данных по индексу ансамбля. В будущем можно убрать.
        date_start_ensemble = session.date_test + timedelta(days=session.adaptation_days_number)
        session.ensemble_index = pd.date_range(date_start_ensemble, session.date_end, freq='D')
        for key in statistics:
            statistics[key] = statistics[key].reindex(session.ensemble_index)
            statistics[key] = statistics[key].fillna(0)
        session.dates = session.ensemble_index


def show():
    if session.was_calc_ftor or session.was_calc_wolfram:
        session.statistics = {}
        prepare_data_for_statistics(session.df_draw_liq,
                                    session.df_draw_oil,
                                    session.df_draw_ensemble,
                                    session.statistics,
                                    session.selected_wells_ois,
                                    session.was_calc_ftor,
                                    session.was_calc_wolfram,
                                    session.was_calc_ensemble,
                                    )

        wells_in_model = []
        for df in session.statistics.values():
            wells_in_model.append(set([col.split('_')[0] for col in df.columns]))
        session.well_names_common = tuple(set.intersection(*wells_in_model))
        session.well_names_all = tuple(set.union(*wells_in_model))
        session.config_stat = ConfigStatistics(
            oilfield=session.field_name,
            dates=session.dates,
            well_names=session.well_names_common,
            use_abs=False,
            ignore_wells=(),
            bin_size=10,
        )
        session.analytics_plots = calculate_statistics(session.statistics, session.config_stat)

        stat_to_draw = st.selectbox(
            label='Статистика',
            options=sorted(session.analytics_plots),
            key='stat_to_draw'
        )
        st.plotly_chart(session.analytics_plots[stat_to_draw], use_container_width=True)

        # Подготовка данных к выгрузке
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            for key in session.statistics:
                session.statistics[key].to_excel(writer, sheet_name=key)
        # Кнопка экспорта результатов
        st.download_button(
            label="Экспорт результатов по всем скважинам",
            data=buffer,
            file_name=f'Все результаты {session.field_name}.xlsx',
            mime='text/csv',
        )
    else:
        st.info('Здесь будет отображаться статистика по выбранному набору скважин.')
