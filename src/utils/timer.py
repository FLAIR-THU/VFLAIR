import sys
import time
from loguru import logger
from typing import Union
import datetime


class Record:
    def __init__(self, func_name: str, **kwargs):
        self.func_name = func_name
        self.time_cost = 0.0
        self.time_count = 0
        self.kwargs = kwargs  # type:dict

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return self.kwargs.get(key)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            else:
                self.kwargs.update({k: v})

    def __str__(self):
        return str(self.__dict__)
        # return f"{self.func_name} took {self.time_cost} seconds"


class Recorder:
    def __init__(self):
        self.summary = {}  # type: [str, Record]

    def add_record(self, record: dict) -> None:
        start_time = record.get('start_time', time.time())
        end_time = record.get('end_time', time.time())
        func_name = record.get('func_name')
        time_delta = end_time - start_time
        if not (record := self.summary.get(func_name)):
            record = Record(func_name)
            self.summary.update({func_name: record})
        record.time_cost += time_delta
        record.time_count += 1

    @property
    def time_str(self, time_stamp) -> str:
        return datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')

    def __call__(self, kwargs):
        return self.add_record(kwargs)

    def __str__(self):
        return '\n'.join([str(v) for v in self.summary.values()])


recorder = Recorder()


def timer(recorder=recorder):
    def decorate(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            _info = {'took': f"{end_time - start_time}s", 'func_name': func.__name__}
            _info.update({'start_time': start_time, 'end_time': end_time})
            recorder(_info)
            # recorder('\n'.join([f"{str(k)}: {str(v)}" for k, v in _info.items()]))
            return result

        return wrapper

    return decorate
