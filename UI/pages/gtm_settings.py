import streamlit as st
from frameworks_shelf_algo.class_Shelf.constants import GTMS, GTM_DATA_FORMAT, NAME
# from frameworks_shelf_algo.class_Shelf.config import ConfigShelf
# from frameworks_shelf_algo.class_Shelf.data_processor_shelf import DataProcessorShelf
# from frameworks_shelf_algo.class_Shelf.support_functions import transform_str_dates_to_datetime_or_vice_versa, \
#     datetime_to_str
# from UI.app_state import AppState


def show(session: st.session_state):
    # state = session.state
    # selected_wells_set = select_wells_set(state)
    # draw_form_exclude_wells(state, selected_wells_set)
    # _config = ConfigShelf
    # _data_processor = DataProcessorShelf(_config)
    # _data_shelf = _get_input_data(_data_processor)
    print("GTM show")
    if 'change_gtm_info' not in st.session_state:
        st.session_state['change_gtm_info'] = False
    # print(st.session_state.shelf_json)
    # print(st.session_state.state.wellnames_key_normal)
    # _data_shelf = DataProcessorShelf.data_shelf
    # print("data_shelf")
    # _well = "9Г"
    # _well = 3350000900
    _well1 = st.selectbox(
        label='Скважина',
        options=st.session_state.state.selected_wells_norm, #.keys(),
        key='well',
    )
    # print(_well1)
    _well = st.session_state.state.wellnames_key_normal[_well1]
    # print(_well)

    def change_gtm_info(command: str):
        print("change_gtm_info")
        _date = st.session_state['DATE' + command]
        # print(_date)
        # print(datetime_to_str(_date))
        # _date1 = datetime_to_str(_date)
        # _date1 = transform_str_dates_to_datetime_or_vice_versa(_date, dates_to_datetime=True)
        if command == 'add':
            # print(_date1)
            # print(_well)
            # print(GTMS)
            st.session_state.shelf_json[_well][GTMS][_date] = dict()
            st.session_state.shelf_json[_well][GTMS][_date][NAME] = st.session_state['NAME' + command]
        for _param in GTM_DATA_FORMAT[st.session_state['NAME' + command]]:
            st.session_state.shelf_json[_well][GTMS][_date][_param] = st.session_state[_param + command]
        st.session_state['change_gtm_info'] = True

    def del_gtm():
        print("del_gtm")
        del st.session_state.shelf_json[_well][GTMS][st.session_state['DATE' + 'edit']]
        st.session_state['change_gtm_info'] = True

    # if _well is None:
    #     st.write('**Выберите скважину**')
    # else:
    with st.expander('Планируемые мероприятия'):
        # print("plan")
        date_lst, name_lst, other_data_lst = [], [], []
        # print(st.session_state.shelf_json[_well][GTMS].items())
        for date, all_data in sorted(st.session_state.shelf_json[_well][GTMS].items()):
            # print(date, all_data)
            date_lst.append(date)
            name_lst.append(all_data[NAME])
            other_data = all_data.copy()
            del other_data[NAME]
            other_data_lst.append(other_data)
            # print("for plan 1")
        date_and_name_lst = [f"{date}: {name}" for date, name in zip(date_lst, name_lst)]
        # print(date_and_name_lst)
        date_and_name = st.selectbox('Название', date_and_name_lst)
        # print("date_and_name")
        if date_and_name is not None:
            # print("if 1")
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
        # if st.session_state['NAME' + 'add'] == 'Ввод новых скважин':
        #     print('Ввод новых скважин')
        #     with st.form('form_add_gtm'):
        #         st.date_input('Дата', key='DATE' + 'add')
        with st.form('form_add_gtm'):
            st.date_input('Дата', key='DATE' + 'add')
            for param, type_ in GTM_DATA_FORMAT[st.session_state['NAME' + 'add']].items():
                val = 0 if type_ == 'int' else 0.0
                st.number_input(param, value=val, key=param + 'add')
            st.form_submit_button('Применить', on_click=change_gtm_info, kwargs={'command': 'add'})

    print(st.session_state['change_gtm_info'])


# def _get_input_data(data_processor):
#     _data_shelf = data_processor.data_shelf
#     _data_shelf_liq = data_processor.data_shelf_liq
#     return _data_shelf
