import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

from UI.config import MODEL_NAMES
from UI.plots import \
    calc_relative_error, \
    draw_statistics, \
    draw_performance, \
    draw_wells_model, \
    draw_histogram_model

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


def calculate_statistics(dfs,
                         well_names_ois,
                         dates,
                         bin_size,
                         analytics_plots):
    # Initialize data
    models = list(dfs.keys())
    df_perf = {key: pd.DataFrame(data=0, index=dates, columns=['факт', 'модель']) for key in models}
    df_perf_liq = {key: pd.DataFrame(data=0, index=dates, columns=['факт', 'модель']) for key in models}
    df_err_liq = {key: pd.DataFrame(data=0, index=dates, columns=['модель']) for key in models}
    df_err = {key: pd.DataFrame(data=0, index=dates, columns=['модель']) for key in models}
    # Daily model error
    df_err_model = {key: pd.DataFrame(index=dates) for key in models}
    df_err_model_liq = {key: pd.DataFrame(index=dates) for key in models}
    # Cumulative model error
    df_cumerr_model = {key: pd.DataFrame(index=dates) for key in models}
    df_cumerr_model_liq = {key: pd.DataFrame(index=dates) for key in models}

    model_mean = dict.fromkeys(models)
    model_std = dict.fromkeys(models)
    model_mean_daily = dict.fromkeys(models)
    model_std_daily = dict.fromkeys(models)

    # Calculations
    print(f'Месторождение: {session.field_name}')
    print(f'Количество различных скважин: {len(well_names_ois)}')

    for model, df in dfs.items():
        print(f'{model} число скважин: {df.shape[1] // 4}')

    for model in models:
        for _well_name in well_names_ois:
            # Check if current model has this well
            if f'{_well_name}_oil_true' not in dfs[model].columns:
                continue

            q_fact = dfs[model][f'{_well_name}_oil_true']
            q_model = dfs[model][f'{_well_name}_oil_pred']
            q_fact_liq = dfs[model][f'{_well_name}_liq_true']
            q_model_liq = dfs[model][f'{_well_name}_liq_pred']
            df_err_model[model][f'{_well_name}'] = (
                    np.abs(q_model - q_fact) / np.maximum(q_model, q_fact) * 100
            )
            df_err_model_liq[model][f'{_well_name}'] = (
                    np.abs(q_model_liq - q_fact_liq) / np.maximum(q_model_liq, q_fact_liq) * 100
            )

            # Cumulative q
            Q_model = q_model.cumsum()
            Q_fact = q_fact.cumsum()
            df_cumerr_model[model][f'{_well_name}'] = (Q_model - Q_fact) / np.maximum(Q_model, Q_fact) * 100

            Q_model_liq = q_model_liq.cumsum()
            Q_fact_liq = q_fact_liq.cumsum()
            df_cumerr_model_liq[model][f'{_well_name}'] = (
                    (Q_model_liq - Q_fact_liq) / np.maximum(Q_model_liq, Q_fact_liq) * 100
            )

            df_perf[model]['факт'] += q_fact.fillna(0)
            df_perf[model]['модель'] += q_model
            df_perf_liq[model]['факт'] += q_fact_liq
            df_perf_liq[model]['модель'] += q_model_liq

    for model in models:
        df_err[model]['модель'] = calc_relative_error(df_perf[model]['факт'], df_perf[model]['модель'])
        df_err_liq[model]['модель'] = calc_relative_error(df_perf_liq[model]['факт'], df_perf_liq[model]['модель'])

        model_mean[model] = df_cumerr_model[model].mean(axis=1)
        model_std[model] = df_cumerr_model[model].std(axis=1)

        # model_mean_liq = df_cumerr_model_liq[model].mean(axis=1)
        # model_std_liq = df_cumerr_model_liq[model].std(axis=1)

        model_mean_daily[model] = df_err_model[model].mean(axis=1)
        model_std_daily[model] = df_err_model[model].std(axis=1)

        # TODO: строится для жидкости/нефти. Если надо для жидкости, то подать "df_err_model_liq"
        temp_name = f'Распределение ошибки: {MODEL_NAMES[model]}'
        analytics_plots[temp_name] = draw_histogram_model(df_err_model[model],
                                                          bin_size,
                                                          session.field_name
                                                          )
        temp_name = f'Ошибка прогноза: {MODEL_NAMES[model]}'
        analytics_plots[temp_name] = draw_wells_model(df_err_model[model])

    # Draw common statistics
    analytics_plots['Суммарная добыча: нефть'] = draw_performance(dfs,
                                                                  df_perf,
                                                                  df_err,
                                                                  session.field_name,
                                                                  mode='oil')
    analytics_plots['Суммарная добыча: жидкость'] = draw_performance(dfs,
                                                                     df_perf_liq,
                                                                     df_err_liq,
                                                                     session.field_name,
                                                                     mode='liq')

    analytics_plots['Статистика'] = draw_statistics(models,
                                                    model_mean,
                                                    model_std,
                                                    model_mean_daily,
                                                    model_std_daily,
                                                    session.field_name,
                                                    dates)


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

        bin_size = 10
        calculate_statistics(session.statistics,
                             session.selected_wells_ois,
                             session.dates,
                             bin_size,
                             session.analytics_plots)

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
