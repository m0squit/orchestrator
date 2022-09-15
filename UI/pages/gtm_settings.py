import pandas as pd
import streamlit as st
from io import BytesIO
from frameworks_shelf_algo.class_Shelf.constants import GTMS, GTM_DATA_FORMAT, NAME
import datetime as dt
from frameworks_shelf_algo.class_Shelf.support_functions import get_date_range, _get_path


def show(session: st.session_state):
    # print("GTM show")
    if 'change_gtm_info' not in st.session_state:
        st.session_state['change_gtm_info'] = 0
    _path = _get_path(session.field_name)
    welllist = pd.read_feather(_path / 'welllist.feather')
    #
    wells_work = pd.read_feather(_path / 'sh_sost_fond.feather')
    wells_work.set_index('dt', inplace=True)
    wells_work = wells_work[wells_work.index > session.date_test]
    wells_work = wells_work[wells_work["sost"] == 'В работе']
    wells_work = wells_work[wells_work["charwork.name"] == 'Нефтяные']
    all_wells_ois_ = wells_work["well.ois"]
    #
    wellnames_key_normal_ = {}
    wellnames_key_ois_ = {}
    for name_well in all_wells_ois_.unique():
        well_name_norm = welllist[welllist["ois"] == name_well]
        well_name_norm = well_name_norm[well_name_norm.npath == 0]
        well_name_norm = well_name_norm.at[well_name_norm.index[0], 'num']
        wellnames_key_normal_[well_name_norm] = name_well
        wellnames_key_ois_[name_well] = well_name_norm
    if 'Все скважины' in session.selected_wells_norm:
        wells_ois = list(session.shelf_json.keys())
        del wells_ois[0]
    else:
        wells_ois = [wellnames_key_normal_[well_name_] for well_name_ in session.selected_wells_norm]
    wells_sorted_ois = sorted(wells_ois)
    wells_sorted_norm = [wellnames_key_ois_[w] for w in wells_sorted_ois]

    _well1 = st.selectbox(
        label='Скважина',
        options=wells_sorted_norm,
        key='well',
    )
    _well = wellnames_key_normal_[_well1]

    def change_gtm_info(command: str):
        _date = st.session_state['DATE' + command]
        if command == 'add':
            st.session_state.shelf_json[_well][GTMS][_date] = dict()
            st.session_state.shelf_json[_well][GTMS][_date][NAME] = st.session_state['NAME' + command]
        for _param in GTM_DATA_FORMAT[st.session_state['NAME' + command]]:
            st.session_state.shelf_json[_well][GTMS][_date][_param] = st.session_state[_param + command]
        st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1

    def del_gtm():
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


    def draw_final_table():
        st.write('-' * 100)
        st.write('**Сводная таблица по ГТМ**')
        # Таблица начинается либо с начала периода адаптации (date_start), либо с начала данных вообще (first_date)
        _date_start = st.session_state['date_start']
        # _date_start = st.session_state.first_date
        _date_end = st.session_state['date_end']
        dates = get_date_range(_date_start, _date_end)
        all_gtm_columns = wells_sorted_norm
        all_gtms = pd.DataFrame(index=dates, columns=all_gtm_columns)
        for _well1 in all_gtm_columns:
            _well = wellnames_key_normal_[_well1]
            for _date1, all_data in sorted(st.session_state.shelf_json[_well][GTMS].items()):
                if _date1 >= _date_start:
                    name = st.session_state.shelf_json[_well][GTMS][_date1][NAME]
                    name_vnr = 'Выход на режим'
                    all_gtms.loc[_date1,_well1] = name
                    if name == 'Текущий ремонт скважин':
                        n_days_trs = st.session_state.shelf_json[_well][GTMS][_date1]['длительность ТРС']
                        dates = get_date_range(_date1, _date1 + dt.timedelta(days=n_days_trs - 1))
                        for _date2 in dates:
                            all_gtms.loc[_date2, _well1] = name
                        n_days_vnr = st.session_state.shelf_json[_well][GTMS][_date1]['длительность выхода на режим']
                        dates = get_date_range(_date1 + dt.timedelta(n_days_trs),
                                               _date1 + dt.timedelta(n_days_trs) + dt.timedelta(days=n_days_vnr - 1))
                        for _date2 in dates:
                            all_gtms.loc[_date2, _well1] = name_vnr
                    elif name == 'Капитальный ремонт скважин':
                        n_days_krs = st.session_state.shelf_json[_well][GTMS][_date1]['длительность КРС']
                        dates = get_date_range(_date1, _date1 + dt.timedelta(days=n_days_krs - 1))
                        for _date2 in dates:
                            all_gtms.loc[_date2, _well1] = name
                        n_days_vnr = st.session_state.shelf_json[_well][GTMS][_date1]['длительность выхода на режим']
                        dates = get_date_range(_date1 + dt.timedelta(n_days_krs),
                                               _date1 + dt.timedelta(n_days_krs) + dt.timedelta(days=n_days_vnr - 1))
                        for _date2 in dates:
                            all_gtms.loc[_date2, _well1] = name_vnr
                    elif name == 'Соляно-кислотная обработка':
                        n_days_sko = st.session_state.shelf_json[_well][GTMS][_date1]['длительность СКО']
                        dates = get_date_range(_date1, _date1 + dt.timedelta(days=n_days_sko - 1))
                        for _date2 in dates:
                            all_gtms.loc[_date2, _well1] = name
                        n_days_vnr = 2
                        dates = get_date_range(_date1 + dt.timedelta(n_days_sko),
                                               _date1 + dt.timedelta(n_days_sko) + dt.timedelta(days=n_days_vnr - 1))
                        for _date2 in dates:
                            all_gtms.loc[_date2, _well1] = name_vnr
                    elif name == 'Промыслово-геофизические исследования':
                        n_days_stop = st.session_state.shelf_json[_well][GTMS][_date1]['длительность остановки']
                        dates = get_date_range(_date1, _date1 + dt.timedelta(days=n_days_stop - 1))
                        for _date2 in dates:
                            all_gtms.loc[_date2, _well1] = name
                    elif name == 'Перевод в нагнетательный фонд':
                        all_gtms.loc[_date1, _well1] = name

        all_gtms = all_gtms.fillna('В работе')

        def color_gtm(val):
            if val == 'Текущий ремонт скважин':
                color = 'blueviolet'
            elif val == 'Капитальный ремонт скважин':
                color = 'antiquewhite'
            elif val == 'Соляно-кислотная обработка':
                color = 'magenta'
            elif val == 'Промыслово-геофизические исследования':
                color = 'moccasin'
            elif val == 'Выход на режим':
                color = 'yellow'
            elif val == 'Перевод в нагнетательный фонд':
                color = 'tomato1'
            else:
                color = 'white'
            return 'background-color: %s' % color

        # st.dataframe(all_gtms.style.applymap(color_gtm))

        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=True, sheet_name='ГТМ')
            worksheet = writer.sheets['ГТМ']
            worksheet.set_column('A:A', None)
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        df_xlsx = to_excel(all_gtms)
        st.download_button(label='Сохранить таблицу', data=df_xlsx, file_name='Сводный_ГТМ.xlsx')

    draw_final_table()


