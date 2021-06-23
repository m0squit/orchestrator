import datetime
from typing import List


class Config(object):

    def __init__(
            self,
            field_name: str,
            model_name: str,
            folder_name: str,
            date_start: datetime.date,
            date_test: datetime.date,
            date_end: datetime.date,
            shops: List[str] = None,
    ):
        self.field_name = field_name
        self.model_name = model_name
        self.folder_name = folder_name
        self.date_start = date_start
        self.date_test = date_test
        self.date_end = date_end
        self.shops = shops
