import datetime
from copy import deepcopy
from typing import List

from config import Config
from models.gtm_test.api.config import Config as ConfigByFtor
from models.wolfram.api.config import Config as ConfigByWolfram
from models.gtm_test.main import run as run_ftor
from models.wolfram.main import run as run_wolfram
from preprocessor import Preprocessor


configs_ftor = [
    Config(
        field_name='Вынгаяхинское',
        shops=['ЦДНГ-10'],
        date_start=datetime.date(2018, 1, 1),
        date_test=datetime.date(2019, 1, 1),
        date_end=datetime.date(2019, 3, 31),
    ),
]
configs_wolfram = []
for config in configs_ftor:
    config_new = deepcopy(config)
    config_new.model_name = 'wolfram'
    config_new.date_start = datetime.date(2018, 1, 1)
    configs_wolfram.append(config_new)


def _run_configs_ftor(configs: List[Config]) -> None:
    for config in configs:
        preprocessor = Preprocessor(config)
        config_ftor = ConfigByFtor(
            path_save=preprocessor._path_current,
            wells=preprocessor._wells_ftor,
        )
        run_ftor(config_ftor)


def _run_configs_wolfram(configs: List[Config]) -> None:
    drop_predicate_pairs = {
        'РАСЧЕТ ДЕБИТА ЖИДКОСТИ': {
            'drop': 'Дебит нефти расчетный',
            'predicate': 'Дебит жидкости среднесуточный',
        },
        'РАСЧЕТ ДЕБИТА НЕФТИ': {
            'drop': 'Дебит жидкости среднесуточный',
            'predicate': 'Дебит нефти расчетный',
        },
    }
    for key, drop_predicate in drop_predicate_pairs.items():
        print(key)
        for config in configs:
            preprocessor = Preprocessor(config)
            #  Выбрасываем из df каждой скважины ненужный столбец.
            for well in preprocessor._wells_ftor:
                df_chess = well.df.drop(columns=drop_predicate['drop'])
                well.df = df_chess.copy()
            config_wolfram = ConfigByWolfram(
                path_save=preprocessor._path_current,
                wells=preprocessor._wells_ftor,
                predicate=drop_predicate['predicate'],
                forecast_days_number=(config.date_end - config.date_test).days + 1,
            )
            run_wolfram(config_wolfram)


_run_configs_ftor(configs_ftor)
_run_configs_wolfram(configs_wolfram)
