import pandas as pd
import streamlit as st
from frameworks_shelf_algo.class_Shelf.constants import GTMS, GTM_DATA_FORMAT, NAME, LAST_MEASUREMENT, DATE, \
    VALUE, VALUE_LIQ, DEC_RATES, DEC_RATES_LIQ
import datetime
from copy import deepcopy
from frameworks_shelf_algo.class_Shelf.support_functions import transform_str_dates_to_datetime_or_vice_versa, \
    get_s_decline_rates, get_s_decline_rates_liq, get_date_range


# def draw_last_measurement_settings(_well: str, _date_start: datetime.date):
#     def edit_last_measurement():
#         st.session_state.shelf_json[_well][LAST_MEASUREMENT][DATE] = st.session_state['changed_date']
#         st.session_state.shelf_json[_well][LAST_MEASUREMENT][VALUE] = st.session_state['changed_val']
#         st.session_state.shelf_json[_well][LAST_MEASUREMENT][VALUE_LIQ] = st.session_state['changed_val_liq']
#         st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1
#
#     def del_last_measurement():
#         st.session_state.shelf_json[_well][LAST_MEASUREMENT] = dict()
#         st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1
#
#     st.write('**Последний замер**')
#     last_measurement_data = st.session_state.shelf_json[_well][LAST_MEASUREMENT]
#     there_is_data = len(last_measurement_data) != 0
#     if there_is_data:
#         date = last_measurement_data[DATE]
#         val = last_measurement_data[VALUE]
#         val_liq = last_measurement_data[VALUE_LIQ]
#         st.write(f"Дата: {date}")
#         st.write(f"Значение нефти: {val}")
#         st.write(f"Значение жидкости: {val_liq}")
#     else:
#         st.write('Данные отсутствуют')
#         date, val, val_liq = _date_start, 0, 0
#     with st.empty():
#         if 'b_edit_last_measurement' in st.session_state and st.session_state['b_edit_last_measurement'] is True:
#             with st.form('form_edit_last_measurement'):
#                 st.date_input('Дата', value=date, key='changed_date')
#                 st.number_input('Значение нефти', value=val, key='changed_val')
#                 st.number_input('Значение жидкости', value=val_liq, key='changed_val_liq')
#                 st.form_submit_button('Применить', on_click=edit_last_measurement)
#         else:
#             st.button('Добавить/изменить', key='b_edit_last_measurement')
#     if there_is_data:
#         st.button('Удалить', on_click=del_last_measurement)
#
#
# def draw_decline_rates_settings(_well: str, _date_start: datetime.date, _date_end: datetime.date):
#     def edit_dec_rate():
#         st.session_state.shelf_json[_well][DEC_RATES][st.session_state['new_date']] = st.session_state['new_val']
#         st.session_state.shelf_json[_well][DEC_RATES_LIQ][st.session_state['new_date']] = st.session_state['new_val_liq']
#         st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1
#
#     def del_dec_rate():
#         del st.session_state.shelf_json[_well][DEC_RATES][st.session_state['date_to_delete']]
#         del st.session_state.shelf_json[_well][DEC_RATES_LIQ][st.session_state['date_to_delete']]
#         st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1
#
#     st.write('**Темпы падения**')
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write('Введенные пользователем:')
#         dec_rates_to_show = deepcopy(st.session_state.shelf_json[_well][DEC_RATES])
#         dec_rates_to_show_liq = deepcopy(st.session_state.shelf_json[_well][DEC_RATES_LIQ])
#         transform_str_dates_to_datetime_or_vice_versa(dec_rates_to_show, dates_to_datetime=False)
#         transform_str_dates_to_datetime_or_vice_versa(dec_rates_to_show_liq, dates_to_datetime=False)
#         st.write('Темпы падения для нефти')
#         st.write(dec_rates_to_show)
#         st.write('Темпы падения для жидкости')
#         st.write(dec_rates_to_show_liq)
#         with st.empty():
#             if 'b_edit_dec_rate' in st.session_state and st.session_state['b_edit_dec_rate'] is True:
#                 with st.form('form_edit_dec_rate'):
#                     st.date_input('Дата', value=_date_start, key='new_date')
#                     st.number_input('Значение ТП нефти', key='new_val')
#                     st.number_input('Значение ТП жидкости', key='new_val_liq')
#                     st.form_submit_button('Добавить/изменить', on_click=edit_dec_rate)
#             else:
#                 st.button('Добавить/изменить', key='b_edit_dec_rate')
#         with st.empty():
#             if 'b_del_dec_rate' in st.session_state and st.session_state['b_del_dec_rate'] is True:
#                 with st.form('form_del_dec_rate'):
#                     dates_when_dec_rate_changes = [*st.session_state.shelf_json[_well][DEC_RATES].keys()]
#                     st.selectbox('Дата', dates_when_dec_rate_changes, key='date_to_delete')
#                     st.form_submit_button('Удалить', on_click=del_dec_rate)
#             else:
#                 if len(st.session_state.shelf_json[_well][DEC_RATES]) != 0:
#                     st.button('Удалить', key='b_del_dec_rate')
#     with col2:
#         st.write('Автозаполненные:')
#         s_decline_rates = get_s_decline_rates(st.session_state.shelf_json, _well, _date_start, _date_end)
#         s_decline_rates_liq = get_s_decline_rates_liq(st.session_state.shelf_json, _well, _date_start, _date_end)
#         pd_decline_show = pd.concat([s_decline_rates, s_decline_rates_liq], axis=1, ignore_index=True)
#         pd_decline_show.columns = ['нефть', 'жидкость']
#         st.write(pd_decline_show)

def show(session: st.session_state):
    # print("GTM show")
    if 'change_gtm_info' not in st.session_state:
        st.session_state['change_gtm_info'] = 0
    _well1 = st.selectbox(
        label='Скважина',
        options=st.session_state.state.selected_wells_norm,
        key='well',
    )
    _well = st.session_state.state.wellnames_key_normal[_well1]
    # _date_start = st.session_state['date_test']
    # draw_last_measurement_settings(_well, _date_start)
    # print('measurements done')
    # st.write('-' * 100)

    def change_gtm_info(command: str):
        # print("change_gtm_info")
        _date = st.session_state['DATE' + command]
        if command == 'add':
            st.session_state.shelf_json[_well][GTMS][_date] = dict()
            st.session_state.shelf_json[_well][GTMS][_date][NAME] = st.session_state['NAME' + command]
        for _param in GTM_DATA_FORMAT[st.session_state['NAME' + command]]:
            st.session_state.shelf_json[_well][GTMS][_date][_param] = st.session_state[_param + command]
        st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1

    def del_gtm():
        # print("del_gtm")
        del st.session_state.shelf_json[_well][GTMS][st.session_state['DATE' + 'edit']]
        st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1

    with st.expander('Планируемые мероприятия'):
        date_lst, name_lst, other_data_lst, other_data_liq_lst = [], [], [], []
        for date, all_data in sorted(st.session_state.shelf_json[_well][GTMS].items()):
            date_lst.append(date)
            name_lst.append(all_data[NAME])
            other_data = all_data.copy()
            del other_data[NAME]
            other_data_lst.append(other_data)
        date_and_name_lst = [f"{date}: {name}" for date, name in zip(date_lst, name_lst)]
        date_and_name = st.selectbox('Название', date_and_name_lst)
        if date_and_name is not None:
            idx = date_and_name_lst.index(date_and_name)
            other_data = other_data_lst[idx]
            st.session_state['NAME' + 'edit'] = name_lst[idx]
            st.session_state['DATE' + 'edit'] = date_lst[idx]
            if 'b_edit_gtm' not in st.session_state or st.session_state['b_edit_gtm'] is False:
                for param, val in other_data.items():
                    st.write(f"{param}: {val}")
                col1, col2 = st.columns(2)
                col1.button('Править', key='b_edit_gtm')
                col2.button('Удалить', on_click=del_gtm)
            else:
                with st.form('form_edit_gtm'):
                    for param, val in other_data.items():
                        st.number_input(param, value=val, key=param + 'edit')
                    st.form_submit_button('Применить', on_click=change_gtm_info, kwargs={'command': 'edit'})

    with st.expander('Добавить новое мероприятие'):
        st.selectbox('Название', GTM_DATA_FORMAT.keys(), key='NAME' + 'add')
        with st.form('form_add_gtm'):
            st.date_input('Дата', key='DATE' + 'add')
            for param, type_ in GTM_DATA_FORMAT[st.session_state['NAME' + 'add']].items():
                val = 0 if type_ == 'int' else 0.0
                st.number_input(param, value=val, key=param + 'add')
            st.form_submit_button('Применить', on_click=change_gtm_info, kwargs={'command': 'add'})

    # st.write('-' * 100)
    # _date_start = st.session_state['date_test']
    # draw_last_measurement_settings(_well, _date_start)
    # print('measurements done')
    # st.write('-' * 100)
    # _date_end = st.session_state['date_end']
    # print(_well, _date_start, _date_end)
    # draw_decline_rates_settings(_well, _date_start, _date_end)

    # _date_start = st.session_state['date_test']
    # draw_last_measurement_settings(_well, _date_start)
    # print('measurements done')

    # dates = get_date_range(_date_start, _date_end)
    # all_gtm = pd.Dataframe(index=dates, dtype=float)
    # all_gtm.columns = st.session_state.state.selected_wells_norm
    # print(all_gtm)

    # for date, val in sorted(_data[_well][DEC_RATES].items()):
    #     s_dec_rates[s_dec_rates.index >= date] = val
    # for date, all_data in sorted(st.session_state.shelf_json[_well][GTMS].items()):
    #     st.session_state.state.selected_wells_norm