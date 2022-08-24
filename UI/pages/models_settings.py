import streamlit as st

from UI.config import ML_FULL_ABBR, YES_NO, DEFAULT_FTOR_BOUNDS


def show(session: st.session_state) -> None:
    draw_ftor_settings(session)
    draw_wolfram_settings(session)
    draw_CRM_settings(session)
    draw_shelf_settings(session)
    draw_ensemble_settings(session)


def draw_ftor_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели пьезопроводности'):
        with st.form(key='ftor_bounds'):
            for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
                cols = st.columns([0.4, 0.2, 0.2, 0.2])
                cols[0].checkbox(
                    label=param_dict['label'],
                    value=session[f'{param_name}_is_adapt'],
                    key=f'{param_name}_is_adapt_',
                    help=param_dict['help']
                )
                cols[1].number_input(
                    label='От',
                    min_value=param_dict['min'],
                    value=session[f'{param_name}_lower'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_lower_'
                )
                cols[2].number_input(
                    label='Фиксированное',
                    min_value=param_dict['min'],
                    value=session[f'{param_name}_default'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_default_'
                )
                cols[3].number_input(
                    label='До',
                    min_value=param_dict['min'],
                    value=session[f'{param_name}_upper'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_upper_',
                    help='включительно'
                )
            submit_bounds = st.form_submit_button('Применить',
                                                  on_click=update_ftor_constraints,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})
        button_use_GDIS = st.button('Использовать границы из ГДИС')
        if button_use_GDIS:
            session.constraints = {}


def draw_wolfram_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели ML'):
        with st.form(key='ML_params'):
            st.selectbox(
                label='Модель на 1-ом уровне',
                options=ML_FULL_ABBR.keys(),
                index=2,  # XGBoost
                help="""
                    Данная модель использует для обучения только входные данные.
                    Подробнее о моделях см. [sklearn](https://scikit-learn.org) 
                    и [xgboost](https://xgboost.readthedocs.io).
                    """,
                key='estimator_name_group_'
            )
            st.selectbox(
                label='Модель на 2-ом уровне',
                options=ML_FULL_ABBR.keys(),
                index=1,  # LinearSVR
                help="""
                    Данная модель использует для обучения как входные данные, 
                    так и результаты работы модели 1-ого уровня.
                    Подробнее о моделях см. [sklearn](https://scikit-learn.org) 
                    и [xgboost](https://xgboost.readthedocs.io).
                    """,
                key='estimator_name_well_'
            )
            st.selectbox(
                label='Cross Validation на 2-ом уровне',
                options=YES_NO.keys(),
                index=1,  # Нет
                help="""
                    Данная процедура нацелена на предотвращение переобучения модели 2-ого уровня.
                    Подробнее см. [Cross-validation for time series](https://robjhyndman.com/hyndsight/tscv).
                    """,
                key='is_deep_grid_search_'
            )
            st.text_input(
                label='Размеры скользящего окна',
                value='3 5 7 15 30',
                max_chars=20,
                help="""Укажите через пробел""",
                key='window_sizes_'
            )
            st.text_input(
                label='Квантили',
                value='0.1 0.3',
                max_chars=20,
                help="""Укажите через пробел""",
                key='quantiles_'
            )

            submit_params = st.form_submit_button('Применить',
                                                  on_click=update_ML_params,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})


def draw_CRM_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели CRM'):
        with st.form(key='CRM_params'):
            st.number_input(
                label='Радиус влияния, м',
                min_value=300,
                value=session.CRM_influence_R,
                max_value=10000,
                step=100,
                help="""Вне радиуса скважины считаются невзаимодействующими""",
                key='CRM_influence_R_'
            )
            st.number_input(
                label='Оптимизатор: максимальное число итераций',
                min_value=1,
                value=session.CRM_maxiter,
                max_value=1000,
                step=50,
                key='CRM_maxiter_'
            )
            st.number_input(
                label='Пластовое давление',
                min_value=100,
                value=session.CRM_p_res,
                max_value=1000,
                step=50,
                help="""Параметр для функции восстановления давления""",
                key='CRM_p_res_'
            )

            submit_params = st.form_submit_button('Применить',
                                                  on_click=update_CRM_params,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})


def draw_shelf_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели ППТП'):
        with st.form(key='shelf_params'):
            max_adapt_period = (session.date_end - session.date_test).days - 1
            # if max_adapt_period <= 25:
            #     max_adapt_period = 30
            st.number_input(
                label='Количество дней для расчета темпа падения',
                min_value=2,
                value=session.n_days_past,
                max_value=max_adapt_period,
                step=1,
                key='n_days_past_'
            )
            max_avg = int(session.n_days_past / 2)
            st.number_input(
                label='Количество дней для осреднения',
                min_value=1,
                value=session.n_days_calc_avg,
                max_value=max_avg,
                step=1,
                key='n_days_calc_avg_'
            )
            submit_params = st.form_submit_button('Применить',
                                                  on_click=update_shelf_params,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})
        # _well1 = st.selectbox(
        #     label='Скважина',
        #     options=st.session_state.state.selected_wells_norm,
        #     key='well',
        # )
        # print(_well1)
        #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


def draw_ensemble_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели ансамбля'):
        with st.form(key='ensemble_params'):
            max_adapt_period = (session.date_end - session.date_test).days - 1
            if max_adapt_period <= 25:
                max_adapt_period = 30
            st.number_input(
                label='Количество дней обучения ансамбля',
                min_value=25,
                value=session.ensemble_adapt_period,
                max_value=max_adapt_period,
                step=1,
                key='ensemble_adapt_period_'
            )
            st.number_input(
                label='Значимость доверительного интервала (от 0 до 1)',
                min_value=0.01,
                value=session.interval_probability,
                max_value=1.,
                step=0.01,
                key='interval_probability_'
            )
            st.number_input(
                label='Draws',
                min_value=100,
                value=session.draws,
                max_value=10000,
                step=10,
                help="""The number of samples to draw. The number of tuned samples are discarded by default.""",
                key='draws_'
            )
            st.number_input(
                label='Tune',
                min_value=100,
                value=session.tune,
                max_value=1000,
                step=10,
                help="""Number of iterations to tune, defaults to 1000. Samplers adjust the step sizes, scalings or
                        similar during tuning. Tuning samples will be drawn in addition to the number specified in
                        the ``draws`` argument.""",
                key='tune_'
            )
            st.number_input(
                label='Chains',
                min_value=1,
                value=session.chains,
                max_value=5,
                step=1,
                help="""The number of chains to sample. Running independent chains is important for some
                        convergence statistics and can also reveal multiple modes in the posterior.""",
                key='chains_'
            )
            st.number_input(
                label='Target_accept',
                min_value=0.01,
                value=session.target_accept,
                max_value=1.,
                step=0.01,
                help="""The step size is tuned such that we approximate this acceptance rate. 
                        Higher values like 0.9 or 0.95 often work better for problematic posteriors""",
                key='target_accept_'
            )
            submit_params = st.form_submit_button('Применить',
                                                  on_click=update_ensemble_params,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})


def update_ftor_constraints(write_from: st.session_state,
                            write_to: st.session_state) -> None:
    # TODO: костыль для многостраничности: приходится записывать параметры модели в session и подтягивать
    #  их для каждой следующей отрисовки. Изменить, когда выйдет версия Streamlit multipage. (~1 квартал 2022)
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        write_to[f'{param_name}_is_adapt'] = write_from[f'{param_name}_is_adapt_']
        write_to[f'{param_name}_lower'] = write_from[f'{param_name}_lower_']
        write_to[f'{param_name}_default'] = write_from[f'{param_name}_default_']
        write_to[f'{param_name}_upper'] = write_from[f'{param_name}_upper_']

    constraints = {}
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        # Если параметр нужно адаптировать
        if write_to[f'{param_name}_is_adapt']:
            constraints[param_name] = [write_to[f'{param_name}_lower'], write_to[f'{param_name}_upper']]
        else:
            # Если значение параметра нужно зафиксировать
            constraints[param_name] = write_to[f'{param_name}_default']
    write_to.constraints = constraints


def update_ML_params(write_from: st.session_state,
                     write_to: st.session_state) -> None:
    write_to['estimator_name_group'] = ML_FULL_ABBR[write_from['estimator_name_group_']]
    write_to['estimator_name_well'] = ML_FULL_ABBR[write_from['estimator_name_well_']]
    write_to['is_deep_grid_search'] = YES_NO[write_from['is_deep_grid_search_']]
    write_to['window_sizes'] = [int(ws) for ws in write_from['window_sizes_'].split()]
    write_to['quantiles'] = [float(q) for q in write_from['quantiles_'].split()]


def update_CRM_params(write_from: st.session_state,
                      write_to: st.session_state) -> None:
    write_to['CRM_influence_R'] = int(write_from['CRM_influence_R_'])
    write_to['CRM_maxiter'] = int(write_from['CRM_maxiter_'])
    write_to['CRM_p_res'] = int(write_from['CRM_p_res_'])

def update_shelf_params(write_from: st.session_state,
                        write_to: st.session_state) -> None:
    write_to['n_days_past'] = int(write_from['n_days_past_'])
    write_to['n_days_calc_avg'] = int(write_from['n_days_calc_avg_'])


def update_shelf_params(write_from: st.session_state,
                        write_to: st.session_state) -> None:
    write_to['n_days_past'] = int(write_from['n_days_past_'])
    write_to['n_days_calc_avg'] = int(write_from['n_days_calc_avg_'])


def update_ensemble_params(write_from: st.session_state,
                           write_to: st.session_state) -> None:
    write_to['interval_probability'] = write_from['interval_probability_']
    write_to['draws'] = int(write_from['draws_'])
    write_to['tune'] = int(write_from['tune_'])
    write_to['chains'] = int(write_from['chains_'])
    write_to['target_accept'] = float(write_from['target_accept_'])
    write_to['ensemble_adapt_period'] = int(write_from['ensemble_adapt_period_'])
