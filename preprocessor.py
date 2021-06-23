import datetime
import pathlib
import pandas as pd
from typing import Dict, List, Tuple

from config import Config
from models.gtm_test.api.bounds import Bounds
from models.gtm_test.api.well import Well as WellByFtor
from models.wolfram.api.well import Well as WellByWolfram


class Preprocessor(object):

    _path_general = pathlib.Path.cwd() / 'data'
    _tables = [
        'coord',
        'coordplast',
        'fond',
        'frac',
        'merop',
        'mersum',
        'projectcoord',
        'projectlist',
        'sh_sost_fond',
        'sost',
        'sppl',
        'troil',
        'welllist',
        'wellplast',
    ]
    _fonds = [
        'Нефтяные',
    ]
    _sosts = [
        'В работе',
    ]

    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._run()

    def _run(self) -> None:
        self._check_dir_existence()
        self._read_data()
        self._select_well_names()
        self._create_wells()

    def _check_dir_existence(self) -> None:
        self.path_current = self._path_general / self._config.field_name / self._config.folder_name
        if not self.path_current.exists():
            raise FileNotFoundError(f'Директория "{self.path_current}" не существует.')

    def _read_data(self) -> None:
        self._data = {}
        for table in self._tables:
            self._data[table] = pd.read_feather(self.path_current / f'{table}.feather')
        self._read_gdis_from_xlsm()

    def _select_well_names(self) -> None:
        df_train = self._data['sh_sost_fond'].loc[
            (self._data['sh_sost_fond']['dt'] >= self._config.date_start) &
            (self._data['sh_sost_fond']['dt'] < self._config.date_test)
        ]
        df_test = self._data['sh_sost_fond'].loc[
            (self._data['sh_sost_fond']['dt'] >= self._config.date_test) &
            (self._data['sh_sost_fond']['dt'] <= self._config.date_end)
        ]
        names_by_train = self._select_well_names_unique(df_train)
        names_by_test = self._select_well_names_unique(df_test)
        if self._config.shops is None:
            df = self._data['welllist']
            names_by_shops = df.loc[df['ceh'].isin(self._config.shops)]['well.ois'].unique().tolist()
            self._well_names = sorted(set(names_by_train) & set(names_by_test) & set(names_by_shops))
        else:
            self._well_names = sorted(set(names_by_train) & set(names_by_test))

    def _select_well_names_unique(self, df: pd.DataFrame) -> List[int]:
        df = df.loc[
            (df['charwork.name'].isin(self._fonds)) &
            (df['sost'].isin(self._sosts))
        ]
        well_names = df['well.ois'].unique().tolist()
        return well_names

    def _create_wells(self) -> None:
        self.wells = []
        for well_name in self._well_names:
            well = Well(
                self._config,
                self._data,
                well_name,
            )
            self.wells.append(well.well)

    def _read_gdis_from_xlsm(self) -> None:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Кпр, мД',
            'Безразмерная проводимость трещины (Fc)',
            'Кгидр, Д*см/сПз',
            'Xf',
            'S мех',
            'Пласт ОИС',
            'Lэфф,м',
            'Рпл текущее на ВНК, кгс/см2',
            'Тип скважины',
            'Кол-во тр-н',
            'k цвет',
            'Pпласт цвет',
            'l цвет',
            'xf цвет',
            'fcd цвет',
        ]
        df = pd.read_excel(
            io=self.path_current / 'гдис.xlsm',
            usecols=cols,
            dtype={
                'Дата окончания исследования': str,
                'Рпл текущее на ВНК, кгс/см2': str,
            },
            engine='openpyxl',
        )
        df.dropna(subset=['Скважина', 'Дата окончания исследования', 'Пласт ОИС'], inplace=True)
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(self._convert_day_date)
        df['Пласт ОИС'] = df['Пласт ОИС'].apply(lambda string: set(string.split()))
        production_names = [
            'добыв.',
            'доб',
            'Добывающая',
        ]
        df = df.loc[df['Тип скважины'].isin(production_names)]
        self._data['gdis'] = df

    @staticmethod
    def _convert_day_date(x: str) -> datetime.date:
        return datetime.datetime.strptime(x, '%d.%m.%Y').date()


class Well(object):

    _ql_name = 'Дебит жидкости среднесуточный'
    _qo_name = 'Дебит нефти расчетный'
    _bhp_name = 'Давление забойное'
    _kind_codes_frac_no = {
        'Вертикально': 0,
        'Наклонно-направленно': 0,
        'Горизонтально': 1,
    }
    _kind_codes_frac = {
        'Вертикально': 2,
        'Наклонно-направленно': 2,
        'Горизонтально': 3
    }

    def __init__(
            self,
            config: Config,
            data: Dict[str, pd.DataFrame],
            well_name_ois: int,
    ):
        self._config = config
        self._data = data
        self._well_name_ois = well_name_ois
        self._run()

    def _run(self) -> None:
        self._set_dates()
        self._set_well_name_geo()
        self._set_formation_names()
        self._set_properties_by_sppl()
        self._set_chess()
        if self._config.model_name == 'ftor':
            self._set_flood()
            self._set_properties_by_troil()
            self._add_bounds_to_chess()
            self._create_well_for_ftor()
        else:
            self._create_well_for_wolfram()

    def _set_dates(self) -> None:
        self._date_start = self._config.date_start
        self._date_test = self._config.date_test
        self._date_end = self._config.date_end

    def _set_well_name_geo(self) -> None:
        df = self._data['welllist'].loc[
            (self._data['welllist']['ois'] == self._well_name_ois) &
            (self._data['welllist']['dtstart'] < self._date_test) &
            (self._data['welllist']['dtend'] > self._date_start)
        ]
        date_start_max = df['dtstart'].max()
        self._well_name_geo = df[df['dtstart'] == date_start_max]['well'].iloc[-1]
        self._correct_date_start(date_start_max)

    def _set_formation_names(self) -> None:
        df = self._data['wellplast'].loc[
            (self._data['wellplast']['well.ois'] == self._well_name_ois) &
            (self._data['wellplast']['dtstart'] < self._date_test) &
            (self._data['wellplast']['dtend'] > self._date_start)
        ]
        date_form_max = df['dtstart'].max()
        self._formation_names = df['plast'].to_list()
        self._correct_date_start(date_form_max)

    def _correct_date_start(self, date: datetime.date) -> None:
        if self._date_start < date:
            self._date_start = date

    def _set_properties_by_sppl(self) -> None:
        df = self._data['sppl'].loc[
            (self._data['sppl']['plastmer'].isin(self._formation_names)) &
            (self._data['sppl']['tk'] < self._date_test)
        ]
        self._porosity = df['pm'].mean()
        self._cr = df['sp'].mean()
        self._cw = df['sw'].mean()
        self._co = df['sn'].mean()
        self._so = df['nb'].mean()
        self._compressibility_total = (self._co * self._so + self._cw * (1 - self._so) + self._cr) * 10
        self._density_oil = df['hs'].mean()

    def _set_chess(self) -> None:
        self._df_chess = self._data['sh_sost_fond'].loc[self._data['sh_sost_fond']['well.ois'] == self._well_name_ois]
        self._df_chess.drop(columns='well.ois', inplace=True)
        self._df_chess.set_index(keys='dt', inplace=True, verify_integrity=True)
        self._df_chess.sort_index(inplace=True)
        self._df_chess = self._df_chess.loc[self._date_start:self._date_end]
        self._df_chess[self._qo_name] = self._df_chess[self._qo_name].apply(lambda x: x / self._density_oil)
        ql = self._df_chess[self._ql_name]
        qo = self._df_chess[self._qo_name]
        self._df_chess['watercut'] = (ql - qo) / ql

    def _set_flood(self) -> None:
        df_mersum = self._data['mersum'].loc[
            (self._data['mersum']['well.ois'] == self._well_name_ois) &
            (self._data['mersum']['plastmer'].isin(self._formation_names)) &
            (self._data['mersum']['dt'] < self._date_start)
        ].copy()
        df_mersum.drop(columns=['well.ois', 'plastmer'], inplace=True)
        df_mersum.dropna(axis=0, how='any', inplace=True)
        df_mersum.set_index(keys=['dt', df_mersum.index], inplace=True, verify_integrity=True)
        df_mersum = df_mersum.sum(axis=0, level='dt')
        df_mersum.index = df_mersum.index.map(lambda x: x.date())
        df_mersum['wc'] = (df_mersum['liq'] - df_mersum['oilm3']) / df_mersum['liq']
        df_mersum.columns = [
            'cuml',
            'cumo',
            'wc_fact',
        ]
        day = datetime.timedelta(days=1)
        df_chess = self._df_chess.loc[:self._date_test - day][[
            self._ql_name,
            self._qo_name,
            'watercut',
        ]]
        df_chess.columns = df_mersum.columns
        self._df_flood = pd.concat(objs=[df_mersum, df_chess])
        self._df_flood.fillna(value=0, inplace=True)
        self._df_flood[['cuml', 'cumo']] = self._df_flood[['cuml', 'cumo']].cumsum()
        self._cum_liq_start = df_mersum['cuml'].sum()
        self._cum_liq_test = self._df_flood['cuml'].iloc[-1]
        self._cum_oil_test = self._df_flood['cumo'].iloc[-1]

    def _set_properties_by_troil(self) -> None:
        df = self._data['troil'].loc[
            (self._data['troil']['well.ois'] == self._well_name_ois) &
            (self._data['troil']['plastmer'].isin(self._formation_names)) &
            (self._data['troil']['dt'] >= self._date_start) &
            (self._data['troil']['dt'] < self._date_test)
        ]
        df.dropna(inplace=True)
        if df.empty or df['skvtype'].iloc[-1] == '':
            print(f'Скважина {self._well_name_ois} не имеет данных в таблице troil на заданные даты расчета.')
            self._kind_code = 1
            self._thickness_oil_saturated = 5
            self._viscosity_liq = 2
            self._volume_factor_liq = 1.2
        else:
            kind_name = df['skvtype'].iloc[-1]
            date = self._date_start + datetime.timedelta(days=31)
            frac_dates = df[df['dt'] < date]['grpdate'].dropna()
            if frac_dates.empty:
                self._kind_code = self._kind_codes_frac_no[kind_name]
            else:
                self._kind_code = self._kind_codes_frac[kind_name]
            self._thickness_oil_saturated = 0
            for formation in self._formation_names:
                self._thickness_oil_saturated += df[df['plastmer'] == formation]['oilsaturatedthickness'].mean()
            if self._thickness_oil_saturated > 50:
                self._thickness_oil_saturated = 10
            self._viscosity_liq = df['mu_liq'].mean()
            self._volume_factor_liq = df['ob_kt'].mean()

    def _add_bounds_to_chess(self) -> None:
        event_dates = self._df_chess['merid.name'].dropna().index
        event_dates = event_dates.insert(0, self._date_start)
        for date in event_dates:
            bound_selector = _BoundParamSelector(
                self._data,
                date,
                self._date_test,
                self._config.field_name,
                self._well_name_ois,
                self._well_name_geo,
                self._kind_code,
                self._formation_names,
                self._thickness_oil_saturated,
                self._viscosity_liq,
            )
            self._df_chess.loc[date, 'bounds'] = Bounds(
                k=bound_selector.k,
                l_hor=bound_selector.l_hor,
                xf=bound_selector.xf,
            )

    def _create_well_for_ftor(self) -> None:
        df_chess = self._df_chess[[
            self._bhp_name,
            self._ql_name,
            self._qo_name,
            'watercut',
            'sost',
            'merid.name',
            'bounds',
        ]]
        df_chess.columns = [
            'p',
            'ql_m3_fact',
            'qo_m3_fact',
            'wc_fact',
            'status',
            'event',
            'bounds',
        ]
        self.well = WellByFtor(
            self._well_name_ois,
            self._date_start,
            self._date_test,
            self._date_end,
            df_chess,
            self._df_flood,
            self._kind_code,
            self._porosity,
            self._compressibility_total,
            self._thickness_oil_saturated,
            self._viscosity_liq,
            self._volume_factor_liq,
            self._density_oil,
            self._cum_liq_start,
            self._cum_liq_test,
            self._cum_oil_test,
        )

    def _create_well_for_wolfram(self) -> None:
        df_chess = self._df_chess[[
            self._bhp_name,
            self._ql_name,
            self._qo_name,
        ]]
        self.well = WellByWolfram(
            self._well_name_ois,
            df_chess,
            self._density_oil,
        )


class _BoundParamSelector:

    _mark_code = {
        'not count': 16711680,
        'bad': 0,
        'good': 32768,
        'excellent': 255,
    }

    def __init__(
            self,
            data: Dict[str, pd.DataFrame],
            date: datetime.date,
            date_test: datetime.date,
            field_name: str,
            well_name: int,
            well_name_geo: str,
            kind_code: int,
            formation_names: List[str],
            thickness_oil_saturated: float,
            viscosity_liq: float,
    ):
        self._data = data
        self._date = date
        self._date_test = date_test
        self._field_name = field_name
        self._well_name = well_name
        self._well_name_geo = well_name_geo
        self._kind_code = kind_code
        self._formation_names = formation_names
        self._thickness_oil_saturated = thickness_oil_saturated
        self._viscosity_liq = viscosity_liq
        self._run()

    def _run(self) -> None:
        self._set_gdis_merop()
        self._set_bounds_permeability()
        self._set_bounds_length_hor_wellbore()
        self._set_bounds_length_half_fracture()

    def _set_gdis_merop(self) -> None:
        self._df_gdis = self._data['gdis'].copy()
        self._df_merop = self._data['merop'].loc[self._data['merop']['well.ois'] == self._well_name].copy()

    def _set_frac(self) -> None:
        self._df_frac = self._data['frac'].copy()
        self._df_frac.dropna(subset=['well.ois', 'frac_date'], inplace=True)
        self._df_frac = self._df_frac.loc[
            (self._df_frac['well.ois'] == self._well_name) &
            (self._df_frac['frac_date'] <= self._date)
        ]

    def _set_bounds_permeability(self) -> None:
        self.k = []
        if not self._get_k_well_plast():
            if not self._get_k_plast():
                self.k = self._get_default_permeability(self._field_name)

    def _set_bounds_length_hor_wellbore(self) -> None:
        self.l_hor = []
        if self._kind_code in [1, 3]:
            if not self._get_l_hor():
                self.l_hor = [100, 500, 1000]

    def _set_bounds_length_half_fracture(self) -> None:
        self.xf = []
        if self._kind_code in [2, 3]:
            self._set_frac()
            if not self._get_xf_gdis():
                if not self._get_xf_frac():
                    self.xf = self._get_default_length_half_fracture(self._field_name)

    def _get_k_well_plast(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Кпр, мД',
            'Кгидр, Д*см/сПз',
            'k цвет',
            'Пласт ОИС',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['k цвет'] != self._mark_code['not count']]
        two_years = datetime.timedelta(2 * 365)
        past_bound_date = self._date - two_years
        future_bound_date = min(self._date + two_years, self._date_test)
        df = self._get_df_gdis_segment(
            df,
            past_bound_date,
            future_bound_date,
            well_filter=True,
            plast_filter=True,
        )
        if df.empty:
            return False
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(self._get_abs_time_interval)
        df = df.loc[df['Дата окончания исследования'] == df['Дата окончания исследования'].min()]

        new_df = df.loc[df['k цвет'] == self._mark_code['excellent']].copy()
        if not new_df.empty:
            k_gidr_init = new_df.iloc[0]['Кгидр, Д*см/сПз']
            k_init = 10 * self._viscosity_liq * k_gidr_init / self._thickness_oil_saturated
            self.k = [0.7 * k_init, k_init, 1.3 * k_init]
            return True

        new_df = df.loc[df['k цвет'] == self._mark_code['good']].copy()
        if not new_df.empty:
            k_gidr_init = new_df.iloc[0]['Кгидр, Д*см/сПз']
            k_init = 10 * self._viscosity_liq * k_gidr_init / self._thickness_oil_saturated
            self.k = [0.3 * k_init, k_init, 1.7 * k_init]
            return True

        new_df = df.loc[df['k цвет'] == self._mark_code['bad']].copy()
        if not new_df.empty:
            k_gidr_init = new_df.iloc[0]['Кгидр, Д*см/сПз']
            k_init = 10 * self._viscosity_liq * k_gidr_init / self._thickness_oil_saturated
            self.k = [1 / 3 * k_init, k_init, 3 * k_init]
            return True

    def _get_k_plast(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Кпр, мД',
            'Кгидр, Д*см/сПз',
            'k цвет',
            'Пласт ОИС',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['k цвет'] != self._mark_code['not count']]
        df = self._get_df_gdis_segment(
            df,
            past_bound_date=None,
            future_bound_date=self._date_test,
            well_filter=False,
            plast_filter=True,
        )
        if df.empty:
            return False
        k_gidr_gdis_max = df['Кгидр, Д*см/сПз'].max()
        k_gidr_gdis_min = df['Кгидр, Д*см/сПз'].min()
        k_max = 10 * self._viscosity_liq * k_gidr_gdis_max / self._thickness_oil_saturated * 1.1
        k_min = 10 * self._viscosity_liq * k_gidr_gdis_min / self._thickness_oil_saturated / 1.1
        self.k = [k_min, (k_min + k_max) / 2, k_max]
        return True

    def _get_l_hor(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Lэфф,м',
            'l цвет',
            'Пласт ОИС',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['l цвет'] != self._mark_code['not count']]
        two_years = datetime.timedelta(2 * 365)
        there_is_fracture = self._kind_code == 3
        if there_is_fracture:
            l_hor_gtms_list = ['ГРП']
            time_bounds_list = [self._date - two_years, self._date + two_years, self._date_test]
            past_bound_date, future_bound_date = self._get_bounds_dates_counting_gtms(l_hor_gtms_list, time_bounds_list)
        else:
            past_bound_date = self._date - two_years
            future_bound_date = min(self._date + two_years, self._date_test)
        df = self._get_df_gdis_segment(
            df,
            past_bound_date,
            future_bound_date,
            well_filter=True,
            plast_filter=True,
        )
        if df.empty:
            return False
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(self._get_abs_time_interval)
        df = df.loc[df['Дата окончания исследования'] == df['Дата окончания исследования'].min()]

        new_df = df.loc[df['l цвет'] == self._mark_code['excellent']].copy()
        if not new_df.empty:
            l_hor_init = new_df.iloc[0]['Lэфф,м']
            if there_is_fracture:
                self.l_hor = [0.4 * l_hor_init, l_hor_init, 1.6 * l_hor_init]
            else:
                self.l_hor = [0.7 * l_hor_init, l_hor_init, 1.3 * l_hor_init]
            return True

        new_df = df.loc[df['l цвет'] == self._mark_code['good']].copy()
        if not new_df.empty:
            l_hor_init = new_df.iloc[0]['Lэфф,м']
            if there_is_fracture:
                self.l_hor = [1 / 2 * l_hor_init, l_hor_init, 2 * l_hor_init]
            else:
                self.l_hor = [0.3 * l_hor_init, l_hor_init, 1.7 * l_hor_init]
            return True

        new_df = df.loc[df['l цвет'] == self._mark_code['bad']].copy()
        if not new_df.empty:
            l_hor_init = new_df.iloc[0]['Lэфф,м']
            self.l_hor = [1 / 3 * l_hor_init, l_hor_init, 3 * l_hor_init]
            return True

    def _get_xf_gdis(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Xf',
            'xf цвет',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['xf цвет'] != self._mark_code['not count']]
        xf_gtms_list = self._get_changing_everything_gtms_list() + ['ГРП']
        time_bounds_list = [self._date_test]
        past_bound_date, future_bound_date = self._get_bounds_dates_counting_gtms(xf_gtms_list, time_bounds_list)
        df = self._get_df_gdis_segment(
            df,
            past_bound_date,
            future_bound_date,
            well_filter=True,
            plast_filter=False,
        )
        if df.empty:
            return False
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(self._get_abs_time_interval)
        df = df.loc[df['Дата окончания исследования'] == df['Дата окончания исследования'].min()]
        it_is_hor = self._kind_code == 3

        new_df = df.loc[df['xf цвет'] == self._mark_code['excellent']].copy()
        if not new_df.empty:
            xf_init = new_df.iloc[0]['Xf']
            if it_is_hor:
                self.xf = [0.4 * xf_init, xf_init, 1.6 * xf_init]
            else:
                self.xf = [0.7 * xf_init, xf_init, 1.3 * xf_init]
            return True

        new_df = df.loc[df['xf цвет'] == self._mark_code['good']].copy()
        if not new_df.empty:
            xf_init = new_df.iloc[0]['Xf']
            if it_is_hor:
                self.xf = [1 / 2 * xf_init, xf_init, 2 * xf_init]
            else:
                self.xf = [0.3 * xf_init, xf_init, 1.7 * xf_init]
            return True

        new_df = df.loc[df['xf цвет'] == self._mark_code['bad']].copy()
        if not new_df.empty:
            xf_init = new_df.iloc[0]['Xf']
            self.xf = [1 / 3 * xf_init, xf_init, 3 * xf_init]
            return True

    def _get_xf_frac(self) -> bool:
        df = self._df_frac.copy()
        df.dropna(subset=['xf'], inplace=True)
        df = df.loc[df['xf'] != 0]
        if df.empty:
            return False
        else:
            xf_frac = df.iloc[-1]['xf']
            self.xf = [1 / 5 * xf_frac, 3 / 5 * xf_frac, xf_frac]
            return True

    def _get_bounds_dates_counting_gtms(
            self,
            needed_gtms_list: List[str],
            time_bounds_list: List[datetime.date],
    ) -> Tuple[datetime.date, datetime.date]:
        df = self._df_merop.loc[self._df_merop['merid.name'].isin(needed_gtms_list)]
        potential_bounds_dates = df['dtend'].tolist() + time_bounds_list
        potential_bounds_intervals = list(map(lambda date: (date - self._date).days, potential_bounds_dates))
        past_bound_index = self._get_negative_max_or_zero_index(potential_bounds_intervals)
        future_bound_index = self._get_positive_min_index(potential_bounds_intervals)
        if past_bound_index is None:
            past_bound_date = None
        else:
            past_bound_date = potential_bounds_dates[past_bound_index]
        if future_bound_index is None:
            future_bound_date = None
        else:
            future_bound_date = potential_bounds_dates[future_bound_index]
        return past_bound_date, future_bound_date

    def _get_df_gdis_segment(
            self,
            df_gdis: pd.DataFrame,
            past_bound_date: datetime.date or None,
            future_bound_date: datetime.date or None,
            well_filter: bool,
            plast_filter: bool,
    ) -> pd.DataFrame:
        if past_bound_date is not None:
            past_filter = past_bound_date < df_gdis['Дата окончания исследования']
        else:
            past_filter = True
        if future_bound_date is not None:
            future_filter = df_gdis['Дата окончания исследования'] <= future_bound_date
        else:
            future_filter = True
        if plast_filter is True:
            plast_filter = df_gdis['Пласт ОИС'] == set(self._formation_names)
        else:
            plast_filter = True
        if well_filter is True:
            well_filter = df_gdis['Скважина'] == self._well_name_geo
        else:
            well_filter = True
        df = df_gdis.loc[past_filter & future_filter & plast_filter & well_filter]
        return df

    def _get_abs_time_interval(self, table_date: datetime.date) -> int:
        return abs((table_date - self._date).days)

    @staticmethod
    def _get_default_permeability(field_name: str) -> List[float]:
        values = {
            'Валынтойское': [0.01, 1, 20],
            'Вынгаяхинское': [0.01, 2, 120],
            'Крайнее': [0.01, 3, 80],
        }
        return values[field_name]

    @staticmethod
    def _get_default_length_half_fracture(field_name: str) -> List[float]:
        values = {
            'Валынтойское': [10, 100, 210],
            'Вынгаяхинское': [10, 100, 300],
            'Крайнее': [10, 100, 250],
        }
        return values[field_name]

    @staticmethod
    def _get_changing_everything_gtms_list() -> List[str]:
        changing_everything_gtms_list = [
            'Ввод новых ГС',
            'Ввод новых ГС с МГРП',
            'Ввод новых ННС из ГРР',
            'Ввод новых ННС с ГРП',
            'ВПП',
            'Вывод из контрольного фонда',
            'Дострел',
            'Запуск скважин',
            'Зарезка боковых горизонтальных стволов',
            'Зарезка боковых горизонтальных стволов с МГРП',
            'Зарезка боковых стволов',
            'Изоляция заколонных перетоков',
            'Ликвидация негерметичности эксплуатационной колонны',
            'Перевод на вышележащий горизонт',
            'Перевод на нижележащий горизонт',
            'Перевод под закачку',
            'Приобщение пласта',
            'Расконсервация скважин',
            'Реперфорация',
        ]
        return changing_everything_gtms_list

    @staticmethod
    def _get_negative_max_or_zero_index(array: List[int]) -> int or None:
        if len(array) == 0:
            return None
        i = 0
        while array[i] > 0:
            i += 1
            if i == len(array):
                return None
        neg_max = array[i]
        neg_max_index = i
        while i < len(array):
            if neg_max < array[i] <= 0:
                neg_max = array[i]
                neg_max_index = i
            i += 1
        return neg_max_index

    @staticmethod
    def _get_positive_min_index(array: List[int]) -> int or None:
        if len(array) == 0:
            return None
        i = 0
        while array[i] <= 0:
            i += 1
            if i == len(array):
                return None
        pos_min = array[i]
        pos_min_index = i
        while i < len(array):
            if 0 < array[i] < pos_min:
                pos_min = array[i]
                pos_min_index = i
            i += 1
        return pos_min_index
