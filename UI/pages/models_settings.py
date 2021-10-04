import streamlit as st
from UI.config import ML_FULL_ABBR, YES_NO, DEFAULT_FTOR_BOUNDS


def update_ftor_constraints():
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


def show():
    with st.expander('Настройки модели пьезопроводности'):
        with st.form(key='ftor_bounds'):
            for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
                cols = st.columns([0.4, 0.2, 0.2, 0.2])
                cols[0].checkbox(
                    label=param_dict['label'],
                    value=True,
                    key=f'{param_name}_is_adapt',
                    help=param_dict['help']
                )
                cols[1].number_input(
                    label='От',
                    min_value=param_dict['min'],
                    value=param_dict['lower_val'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_lower'
                )
                cols[2].number_input(
                    label='Фиксированное',
                    min_value=param_dict['min'],
                    value=param_dict['default_val'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_default'
                )
                cols[3].number_input(
                    label='До',
                    min_value=param_dict['min'],
                    value=param_dict['upper_val'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_upper',
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
                key='window_sizes_'
            )
            st.text_input(
                label='Квантили',
                value='0.1 0.3',
                max_chars=20,
                key='quantiles_'
            )

            submit_params = st.form_submit_button('Применить', on_click=update_ML_params)

