import datetime
from dateutil.relativedelta import relativedelta

FIELDS_SHOPS = {
    'Валынтойское': ['ЦДНГ-12'],
    'Вынгаяхинское': ['ЦДНГ-10'],
    'Крайнее': ['ЦДНГ-4', 'ЦДНГ-2'],
    'Отдельное': ['ЦДНГ-1'],
    'Романовское': ['ЦДНГ-3'],
    'Холмогорское': ['ЦДHГ-1'],  # H - латинская
}

# Диапазон дат выгрузки sh таблицы
DATE_MIN = datetime.date(2018, 1, 1)
DATE_MAX = datetime.date(2021, 4, 30)

PERIOD_TRAIN_MIN = relativedelta(months=3)
PERIOD_TEST_MIN = relativedelta(months=1)

ML_FULL_ABBR = {
    'ElasticNet': 'ela',
    'LinearSVR': 'svr',
    'XGBoost': 'xgb',
}
YES_NO = {
    'Да': True,
    'Нет': False,
}

DEFAULT_FTOR_BOUNDS = {
    'boundary_code': {
        'label': 'Тип границ',
        'lower_val': 0,
        'default_val': 0,
        'upper_val': 5,
        'step': 1,
        'min': 0,
        'max': 5,
        'help': """Тип границ - непротекания или постоянного давления.
            0: RECTANGLE_BOUNDARY_CCCC
            1: RECTANGLE_BOUNDARY_NNNN
            2: RECTANGLE_BOUNDARY_NCNC
            3: RECTANGLE_BOUNDARY_NNCC
            4: RECTANGLE_BOUNDARY_NNCN
            5: RECTANGLE_BOUNDARY_NCCC""",
    },
    'permeability': {
        'label': 'Проницаемость k, мД',
        'lower_val': 0.1,
        'default_val': 5.,
        'upper_val': 10.,
        'step': 0.01,
        'min': 0.1,
        'max': 1000.,
        'help': '',
    },
    'skin': {
        'label': 'Skin',
        'lower_val': 0.1,
        'default_val': 1.,
        'upper_val': 2.,
        'step': 0.1,
        'min': 0.1,
        'max': 20.,
        'help': """Скин-фактор""",
    },
    'res_width': {
        'label': 'Ширина резервуара, м',
        'lower_val': 200,
        'default_val': 600,
        'upper_val': 700,
        'step': 50,
        'min': 10,
        'max': 100000,
        'help': """Ширина прямоугольного резервуара, м""",
    },
    'res_length': {
        'label': 'Длина резервуара, м',
        'lower_val': 200,
        'default_val': 600,
        'upper_val': 700,
        'step': 50,
        'min': 10,
        'max': 100000,
        'help': """Длина прямоугольного резервуара, м""",
    },
    'pressure_initial': {
        'label': 'Начальное пластовое давление, атм',
        'lower_val': 100,
        'default_val': 500,
        'upper_val': 700,
        'step': 5,
        'min': 1,
        'max': 2000,
        'help': '',
    },
    'length_hor_well_bore': {
        'label': 'Длина горизонт. ствола, м',
        'lower_val': 100,
        'default_val': 200,
        'upper_val': 300,
        'step': 5,
        'min': 1,
        'max': 100000,
        'help': """Длина горизонтального ствола скважины, м""",
    },
    'length_half_fracture': {
        'label': 'Полудлина трещины ГРП xf, м',
        'lower_val': 10,
        'default_val': 60,
        'upper_val': 150,
        'step': 1,
        'min': 1,
        'max': 10000,
        'help': '',
    },
    'number_fractures': {
        'label': 'Число трещин ГРП',
        'lower_val': 2,
        'default_val': 3,
        'upper_val': 6,
        'step': 1,
        'min': 2,
        'max': 100,
        'help': """Число трещин ГРП (для скважин МГРП)""",
    },
}

FTOR_DECODE = {
    "boundary_code": {
        'label': 'Тип границ резервуара',
        0: "CCCC",
        1: "NNNN",
        2: "NCNC",
        3: "NNCC",
        4: "NNCN",
        5: "NCCC",
    },
    'permeability': {
        'label': 'Проницаемость k, мД',
    },
    'skin': {
        'label': 'Skin',
    },
    'res_width': {
        'label': 'Ширина резервуара, м',
    },
    'res_length': {
        'label': 'Длина резервуара, м',
    },
    'pressure_initial': {
        'label': 'Начальное пластовое давление, атм',
    },
    'length_hor_well_bore': {
        'label': 'Длина горизонт. ствола, м',
    },
    'length_half_fracture': {
        'label': 'Полудлина трещины ГРП xf, м',
    },
    'number_fractures': {
        'label': 'Число трещин ГРП',
    },
    "kind_code": {
        'label': 'Тип скважины',
        0: "Вертикальная скважина",
        1: "Горизонтальная скважина",
        2: "Вертикальная скважина с ГРП",
        3: "Горизонт. скв. МГРП",
    },
}

ignore_plots = [
    'Распределение ошибки (жидкость) "CRM"',
    'Ошибка прогноза (жидкость) "CRM"',
    'Распределение ошибки (жидкость) "Ансамбль"',
    'Ошибка прогноза (жидкость) "Ансамбль"',
]
