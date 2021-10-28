import streamlit as st
from UI.config import ML_FULL_ABBR, YES_NO, DEFAULT_FTOR_BOUNDS


def update_ftor_constraints():
    # TODO: костыль для многостраничности: приходится записывать параметры модели в st.session_state и подтягивать
    #  их для каждой следующей отрисовки. Изменить, когда выйдет версия Streamlit multipage. (4 квартал 2021)
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        st.session_state[f'{param_name}_is_adapt'] = st.session_state[f'{param_name}_is_adapt_']
        st.session_state[f'{param_name}_lower'] = st.session_state[f'{param_name}_lower_']
        st.session_state[f'{param_name}_default'] = st.session_state[f'{param_name}_default_']
        st.session_state[f'{param_name}_upper'] = st.session_state[f'{param_name}_upper_']

    discrete_params = ['boundary_code', 'number_fractures']
    constraints = {}
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        # Если параметр нужно адаптировать
        if st.session_state[f'{param_name}_is_adapt']:
            if param_name in discrete_params:
                constraints[param_name] = {
                    'is_discrete': True,
                    'bounds': [i for i in range(st.session_state[f'{param_name}_lower'],
                                                st.session_state[f'{param_name}_upper'] + 1)]
                }
            else:
                constraints[param_name] = {
                    'is_discrete': False,
                    'bounds': [st.session_state[f'{param_name}_lower'], st.session_state[f'{param_name}_upper']]
                }
        else:
            # Если значение параметра нужно зафиксировать
            constraints[param_name] = st.session_state[f'{param_name}_default']
    st.session_state.constraints = constraints


def update_ML_params():
    st.session_state.estimator_name_group = ML_FULL_ABBR[st.session_state.estimator_name_group_]
    st.session_state.estimator_name_well = ML_FULL_ABBR[st.session_state.estimator_name_well_]
    st.session_state.is_deep_grid_search = YES_NO[st.session_state.is_deep_grid_search_]
    st.session_state.window_sizes = [int(ws) for ws in st.session_state.window_sizes_.split()]
    st.session_state.quantiles = [float(q) for q in st.session_state.quantiles_.split()]


def update_ensemble_params():
    st.session_state['interval_probability'] = st.session_state['interval_probability_']
    st.session_state['draws'] = st.session_state['draws_']
    st.session_state['tune'] = st.session_state['tune_']
    st.session_state['chains'] = st.session_state['chains_']
    st.session_state['target_accept'] = st.session_state['target_accept_']
    st.session_state['adaptation_days_number'] = st.session_state['adaptation_days_number_']


def show():
    with st.expander('Настройки модели пьезопроводности'):
        with st.form(key='ftor_bounds'):
            for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
                cols = st.columns([0.4, 0.2, 0.2, 0.2])
                cols[0].checkbox(
                    label=param_dict['label'],
                    value=st.session_state[f'{param_name}_is_adapt'],
                    key=f'{param_name}_is_adapt_',
                    help=param_dict['help']
                )
                cols[1].number_input(
                    label='От',
                    min_value=param_dict['min'],
                    value=st.session_state[f'{param_name}_lower'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_lower_'
                )
                cols[2].number_input(
                    label='Фиксированное',
                    min_value=param_dict['min'],
                    value=st.session_state[f'{param_name}_default'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_default_'
                )
                cols[3].number_input(
                    label='До',
                    min_value=param_dict['min'],
                    value=st.session_state[f'{param_name}_upper'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_upper_',
                    help='включительно'
                )
            submit_bounds = st.form_submit_button('Применить', on_click=update_ftor_constraints)

    with st.expander('Настройки модели ML'):
        with st.form(key='ML_params'):
            st.selectbox(
                label='Модель на 1-ом уровне',
                options=ML_FULL_ABBR.keys(),
                index=1,  # LinearSVR
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
                index=0,  # ElasticNet
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

            submit_params = st.form_submit_button('Применить', on_click=update_ML_params)

    with st.expander('Настройки модели ансамбля'):
        with st.form(key='ensemble_params'):
            st.number_input(
                label='Количество дней обучения ансамбля',
                min_value=25,
                value=st.session_state.adaptation_days_number,
                max_value=(st.session_state.date_end - st.session_state.date_test).days - 1,
                step=1,
                key='adaptation_days_number_'
            )
            st.number_input(
                label='Значимость доверительного интервала (от 0 до 1)',
                min_value=0.01,
                value=st.session_state.interval_probability,
                max_value=1.,
                step=0.01,
                key='interval_probability_'
            )
            st.number_input(
                label='Draws',
                min_value=100,
                value=st.session_state.draws,
                max_value=10000,
                step=10,
                help="""The number of samples to draw. The number of tuned samples are discarded by default.""",
                key='draws_'
            )
            st.number_input(
                label='Tune',
                min_value=100,
                value=st.session_state.tune,
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
                value=st.session_state.chains,
                max_value=5,
                step=1,
                help="""The number of chains to sample. Running independent chains is important for some
                        convergence statistics and can also reveal multiple modes in the posterior.""",
                key='chains_'
            )
            st.number_input(
                label='Target_accept',
                min_value=0.01,
                value=st.session_state.target_accept,
                max_value=1.,
                step=0.01,
                help="""The step size is tuned such that we approximate this acceptance rate. 
                        Higher values like 0.9 or 0.95 often work better for problematic posteriors""",
                key='target_accept_'
            )
            submit_params = st.form_submit_button('Применить', on_click=update_ensemble_params)

