from typing import List, Tuple, Dict
import warnings
from datetime import date
from pathlib import Path
import numpy as np
from loguru import logger
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from frameworks_crm.class_CRM.calculator import Calculator as CalculatorCRM
from frameworks_crm.class_CRM.config import ConfigCRM
from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from frameworks_ftor.ftor.config import Config as ConfigFtor
from frameworks_wolfram.wolfram.calculator import Calculator as CalculatorWolfram
from frameworks_wolfram.wolfram.config import Config as ConfigWolfram
from models_ensemble.calculator import Calculator as CalculatorEnsemble
from models_ensemble.config import Config as ConfigEnsemble
from statistics_explorer.config import ConfigStatistics
from statistics_explorer.main import calculate_statistics
from tools_preprocessor.config import Config as ConfigPreprocessor
from tools_preprocessor.preprocessor import Preprocessor
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from frameworks_crm.class_Fedot.plot_fedot import plot_graph

warnings.filterwarnings('ignore')


class FedotModel:
    def __init__(self,
                 oilfield: str,
                 train_start: date,
                 train_end: date,
                 predict_start: date,
                 predict_end: date,
                 wells_to_calc: list[str or int],
                 coeff: pd.DataFrame,
                 save_plots: bool = False):
        self.oilfield = oilfield
        self.train_start = train_start
        self.train_end = train_end
        self.predict_start = predict_start
        self.predict_end = predict_end
        self.wells_to_calc = wells_to_calc
        self.coeff = coeff
        self.save_plots = save_plots
        # Для статистики
        self.statistics = pd.DataFrame(index=pd.date_range(self.predict_start, self.predict_end, freq='D'))

        self.run()

    def run(self):
        self._read_data()
        self._restructure_data()
        self._create_fedot_models()
        self._calculate()

    def _read_data(self):
        # подгружаем эксель мессояхи
        project_root = Path(__file__).parent.parent
        path_read = project_root / 'tools_preprocessor' / 'data' / f'{self.oilfield}'

        xls = pd.ExcelFile(path_read / f'Input_data_CRM_{self.oilfield}.xlsx')
        self.prod_info = pd.read_excel(xls, sheet_name=0, dtype={'Скважина': str})
        self.inj_info = pd.read_excel(xls, sheet_name=1, dtype={'Скважина': str})

    def _restructure_data(self):
        self.data_prod = pd.pivot_table(self.prod_info,
                                        values=['Дебит жидкости', 'Дебит нефти', 'Забойное давление'],
                                        index='Дата',
                                        columns='Скважина',
                                        dropna=False)[self.train_start:self.predict_end]

        self.data_inj = pd.pivot_table(self.inj_info,
                                       values='Приемистость',
                                       index='Дата',
                                       columns='Скважина',
                                       dropna=False)[self.train_start:self.predict_end]
        # заполняем пустые значения
        self.data_prod = self.data_prod.interpolate(method='linear', axis=1)
        self.data_prod = self.data_prod.fillna(0)
        self.data_inj = self.data_inj.fillna(1)

    def _create_fedot_models(self):
        # Ml обучается отдельно для жидкости и для нефти
        self.fedot_model_water = Fedot(problem='regression', timeout=0.5,
                                       seed=42, verbose_level=-1)
        # API для automl, ставим тип задачи(тут регрессия, в будущем когда
        # когда разрабы временные ряды допилят - на них перейду), timeout - время построения пайплайна,
        # seed - рандом построения пайплайна, verbose_level - логгирование, тут возможен баг.
        # Но это разрабы пофиксят в будущем обновлении.
        # аналогично для нефти
        self.fedot_model_oil = Fedot(problem='regression', timeout=0.5,
                                     seed=42, verbose_level=-1)

    def _calculate(self):
        available_wells = [name for name in self.wells_to_calc if name in self.coeff.columns]
        for wellname in available_wells:
            try:
                self._calculate_well(wellname)
                logger.success(f'FEDOT: success {wellname}')
            except Exception as err:
                logger.exception(f'FEDOT: FAIL {wellname}', err)

    def _calculate_well(self, prod_well):
        # Создание массива фич на трэйн
        inj_well_water = self.injection(prod_well, self.data_inj, self.coeff)[:self.train_end]
        # список с значениями для приемистостей. Вложенные списки хранит
        inj_to_feature_water = []
        for name_inj_well_water in inj_well_water.columns:
            inj_to_feature_water.append(inj_well_water[name_inj_well_water].values)

            # создаём размерность фичи
        n = len(self.data_inj[:self.train_end])  # rows
        m = inj_well_water.shape[1] + 1  # columns
        feature_train_water = np.zeros(shape=(n, m))
        # oil
        feature_train_oil = np.zeros(shape=(n, m))

        # записываем значения в фичу
        for i in range(feature_train_water.shape[0]):

            feature_train_water[i][0] = self.data_prod['Забойное давление'][prod_well][i]
            feature_train_oil[i][0] = self.data_prod['Забойное давление'][prod_well][i]
            for j in range(1, m):
                feature_train_water[i][j] = inj_to_feature_water[j - 1][i]
                feature_train_oil[i][j] = inj_to_feature_water[j - 1][i]

        # water тут оборачивается всё в InputData, это не обязательно. Но т.к. .fit съедает и numpy массив.
        # Но я делал это по туториалы их и они оборачивали :)
        target_train_water = self.data_prod[:self.train_end]['Дебит жидкости'][prod_well].values
        train_data_water = InputData(idx=np.arange(0, len(feature_train_water)), features=feature_train_water,
                                     target=target_train_water, task=Task(TaskTypesEnum.regression),
                                     data_type=DataTypesEnum.table)

        # oil
        target_train_oil = self.data_prod[:self.train_end]['Дебит нефти'][prod_well].values
        train_data_oil = InputData(idx=np.arange(0, len(feature_train_oil)), features=feature_train_oil,
                                   target=target_train_oil, task=Task(TaskTypesEnum.regression),
                                   data_type=DataTypesEnum.table)

        # собственно обучение воды и нефти. можно выбрать любой вариант отработает одинаково
        pipeline_water = self.fedot_model_water.fit(train_data_water)
        #     pipeline_water = fedot_model_water.fit(feature_train_water,target_train_water )

        pipeline_oil = self.fedot_model_oil.fit(train_data_oil)
        #     pipeline_oil = fedot_model_oil.fit(feature_train_oil, target_train_oil )

        # Создание массива фич на предикт
        inj_well_water_pred = self.injection(prod_well, self.data_inj, self.coeff)[self.predict_start:]
        inject_well = list(inj_well_water_pred.columns)

        # список с значениями для приемистостей. Вложенные списки хранит
        inj_to_feature_water_pred = []
        for name_inj_well_water in inj_well_water_pred.columns:
            inj_to_feature_water_pred.append(inj_well_water_pred[name_inj_well_water].values)

        # создаём размерность фичи
        k = len(self.data_inj[self.predict_start:])  # rows
        l = inj_well_water_pred.shape[1] + 1  # columns
        feature_pred_water = np.zeros(shape=(k, l))

        # oil
        feature_pred_oil = np.zeros(shape=(k, l))

        # записываем значения в фичу
        for i in range(feature_pred_water.shape[0]):
            feature_pred_water[i][0] = self.data_prod[self.predict_start:]['Забойное давление'][prod_well][i]
            feature_pred_oil[i][0] = self.data_prod[self.predict_start:]['Забойное давление'][prod_well][i]
            for j in range(1, l):
                feature_pred_water[i][j] = inj_to_feature_water_pred[j - 1][i]
                feature_pred_oil[i][j] = inj_to_feature_water_pred[j - 1][i]

        target_pred_water = self.data_prod[self.predict_start:]['Дебит жидкости'][prod_well].values
        target_pred_oil = self.data_prod[self.predict_start:]['Дебит нефти'][prod_well].values

        # water
        test_data_water = InputData(idx=np.arange(0, len(feature_pred_water)), features=feature_pred_water,
                                    target=target_pred_water, task=Task(TaskTypesEnum.regression),
                                    data_type=DataTypesEnum.table)
        # oil
        test_data_oil = InputData(idx=np.arange(0, len(feature_pred_oil)), features=feature_pred_oil,
                                  target=target_pred_oil, task=Task(TaskTypesEnum.regression),
                                  data_type=DataTypesEnum.table)

        prediction_water = self.fedot_model_water.predict(test_data_water)
        #     prediction_water = fedot_model_water.predict(feature_pred_water)
        # выводим как обучалась модель
        prediction_fact_water = self.fedot_model_water.predict(train_data_water)
        # oil
        prediction_oil = self.fedot_model_oil.predict(test_data_oil)
        #     prediction_oil = fedot_model_oil.predict(feature_pred_oil)
        prediction_fact_oil = self.fedot_model_oil.predict(train_data_oil)

        # обернём всё в numpy массивы, чтобы plotly не ругалась,
        # да и в статистику всё можно было адекватно сохранить
        prediction_water = np.array(prediction_water)
        prediction_fact_water = np.array(prediction_fact_water)
        prediction_oil = np.array(prediction_oil)
        prediction_fact_oil = np.array(prediction_fact_oil)

        self._extract_data(prod_well, prediction_water, prediction_oil)

        # вызов функции отрисовки графиков
        if self.save_plots:
            plot_graph(prod_well, self.data_prod, prediction_water,
                       prediction_fact_water, self.train_end, self.predict_start, 'water')

            plot_graph(prod_well, self.data_prod, prediction_oil,
                       prediction_fact_oil, self.train_end, self.predict_start, 'oil')

    def _extract_data(self,
                      prod_well: str,
                      prediction_water: np.array,
                      prediction_oil: np.array):
        liq_true = self.data_prod[self.predict_start:]['Дебит жидкости'][prod_well]
        oil_true = self.data_prod[self.predict_start:]['Дебит нефти'][prod_well]
        self.statistics[f'{prod_well}_liq_true'] = liq_true
        self.statistics[f'{prod_well}_liq_pred'] = prediction_water
        self.statistics[f'{prod_well}_oil_true'] = oil_true
        self.statistics[f'{prod_well}_oil_pred'] = prediction_oil

    @staticmethod
    def injection(prod_well: str, injection: pd.DataFrame, coefficient: pd.DataFrame) -> pd.DataFrame:
        """Возвращает значения с пересчитанными приемистостями с учётом коэффициентов взаимовлияния.

        args:
            prod_well - добывающая скважина в формате str
            injection - DataFrame с приемистостями
            coefficient - матрица коэффициентов взаимовлияния
        return:
            hybrid_injection - DataFrame со значениями приемистости
        """
        new_injection = pd.DataFrame(index=injection.index)
        coefficient = coefficient.T
        for inj_well in coefficient:
            if coefficient[inj_well][prod_well] != 0:
                new_injection[inj_well] = injection[str(inj_well)] * coefficient[inj_well][prod_well]

        return new_injection


@st.cache(show_spinner=False)
def run_preprocessor(config: ConfigPreprocessor) -> Preprocessor:
    _preprocessor = Preprocessor(config)
    return _preprocessor


@st.cache
def calculate_ftor(_preprocessor: Preprocessor,
                   well_names: List[int],
                   constraints: dict) -> CalculatorFtor:
    config_ftor = ConfigFtor()
    # Если пользователь задал границы\значение параметра, которым производится адаптация на
    # последние точки, то эти значения применяются и для самой адаптации на последние точки
    param_last_point = config_ftor.param_name_last_points_adaptation
    if param_last_point in constraints.keys():
        if type(constraints[param_last_point]) == dict:
            config_ftor.param_bounds_last_points_adaptation = constraints[param_last_point]['bounds']
        else:
            config_ftor.apply_last_points_adaptation = False

    ftor = CalculatorFtor(
        config_ftor,
        _preprocessor.create_wells_ftor(
            well_names,
            user_constraints_for_adap_period=constraints,
        ),
        logging=True
    )
    return ftor


@st.experimental_singleton
def calculate_wolfram(_preprocessor: Preprocessor,
                      well_names: List[int],
                      forecast_days_number: int,
                      estimator_name_group: str,
                      estimator_name_well: str,
                      is_deep_grid_search: bool,
                      window_sizes: List[int],
                      quantiles: List[float]) -> CalculatorWolfram:
    wolfram = CalculatorWolfram(
        ConfigWolfram(
            forecast_days_number,
            estimator_name_group,
            estimator_name_well,
            is_deep_grid_search,
            window_sizes,
            quantiles,
        ),
        _preprocessor.create_wells_wolfram(well_names),
    )
    return wolfram


@st.experimental_singleton
def calculate_CRM(date_start_adapt: date,
                  date_end_adapt: date,
                  date_end_forecast: date,
                  oilfield: str,
                  calc_CRM: bool = True,
                  calc_CRMIP: bool = False,
                  grad_format_data: bool = True,
                  influence_R: int = 1300,
                  maxiter: int = 100,
                  p_res: int = 220) -> CalculatorCRM or None:
    config_CRM = ConfigCRM(date_start_adapt=date_start_adapt,
                           date_end_adapt=date_end_adapt,
                           date_end_forecast=date_end_forecast,
                           calc_CRM=calc_CRM,
                           calc_CRMIP=calc_CRMIP,
                           grad_format_data=grad_format_data,
                           oilfield=oilfield)
    config_CRM.INFLUENCE_R = influence_R
    config_CRM.options_SLSQP_CRM['maxiter'] = maxiter
    config_CRM.p_res = p_res
    try:
        logger.info(f'CRM: start calculations')
        calculator_CRM = CalculatorCRM(config_CRM)
        logger.success(f'CRM: success')
        return calculator_CRM
    except Exception as exc:
        logger.exception('CRM: FAIL', exc)
        return None


@st.experimental_singleton
def calculate_fedot(oilfield: str,
                    train_start: date,
                    train_end: date,
                    predict_start: date,
                    predict_end: date,
                    wells_to_calc: list[str],
                    coeff: pd.DataFrame) -> FedotModel:
    calculator_fedot = FedotModel(oilfield=oilfield,
                                  train_start=train_start,
                                  train_end=train_end,
                                  predict_start=predict_start,
                                  predict_end=predict_end,
                                  wells_to_calc=wells_to_calc,
                                  coeff=coeff)
    return calculator_fedot


@st.experimental_singleton
def calculate_ensemble(input_data: list[dict],
                       adaptation_days_number: int,
                       interval_probability: float,
                       draws: int,
                       tune: int,
                       chains: int,
                       target_accept: float,
                       name_of_y_true: str) -> dict[str, pd.DataFrame]:
    calculator_ensemble = CalculatorEnsemble(
        ConfigEnsemble(adaptation_days_number=adaptation_days_number,
                       interval_probability=interval_probability,
                       draws=draws,
                       tune=tune,
                       chains=chains,
                       target_accept=target_accept,
                       name_of_y_true=name_of_y_true),
        input_data,
        logging=True
    )
    return calculator_ensemble.result_test


@st.experimental_memo
def calculate_statistics_plots(
        statistics: dict,
        field_name: str,
        date_start: date,
        date_end: date,
        well_names: tuple,
        use_abs: bool,
        exclude_wells: list,
        bin_size: int
) -> Tuple[Dict[str, go.Figure], ConfigStatistics]:
    config_stat = ConfigStatistics(
        oilfield=field_name,
        dates=pd.date_range(date_start, date_end, freq='D').date,
        well_names=well_names,
        use_abs=use_abs,
        bin_size=bin_size,
    )
    config_stat.exclude_wells(exclude_wells)
    analytics_plots = calculate_statistics(statistics, config_stat)
    return analytics_plots, config_stat
