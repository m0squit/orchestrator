import sys


class Worker:
    """Класс для логирования внутри процессов Multiprocessing.Pool()"""
    logger = None

    @staticmethod
    def set_logger(logger_):
        Worker.logger = logger_
        Worker.logger.add(sys.stderr, level='DEBUG', enqueue=True)
