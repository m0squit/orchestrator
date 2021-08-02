import datetime
from typing import List


class Config(object):
    """Настройки расчета для конкретного месторождения.
    """

    def __init__(
            self,
            field_name: str,
            shops: List[str],
            date_start: datetime.date,
            date_test: datetime.date,
            date_end: datetime.date,
    ):
        """Инициализация класса Config.

        Args:
            field_name: Название месторождения на русском.
            shops: Список названий цехов на русском.
            date_start: Дата начала адаптации моделей (начало с 00:00 этой даты).
            date_test: Дата начала прогноза моделей (начало с 00:00 этой даты).
            date_end: Дата конца прогноза моделей (конец по 23:59 этой даты).
        """
        self.field_name = field_name
        self.shops = shops
        self.date_start = date_start
        self.date_test = date_test
        self.date_end = date_end
