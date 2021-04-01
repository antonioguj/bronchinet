
from typing import Tuple
import inspect
import sys


class PrintFrameManager(object):

    @staticmethod
    def get_frame_record(level: int):
        return inspect.stack()[level]
        # level = 0 : for this line (not relevant) ( in PrintFrameManager.get_info_caller_frame() )
        # level = 1 : for line at direct caller ( in catch_error_exception(message) )
        # level = 2 : for line at external caller

    @staticmethod
    def get_info_current_frame() -> Tuple[str, int]:
        caller_frame_record = inspect.stack()[1]
        frame = caller_frame_record[0]
        info = inspect.getframeinfo(frame)
        return (info.filename, info.lineno)

    @staticmethod
    def get_info_caller_frame() -> Tuple[str, int]:
        caller_frame_record = inspect.stack()[2]
        frame = caller_frame_record[0]
        info = inspect.getframeinfo(frame)
        return (info.filename, info.lineno)


def catch_error_exception(message: str) -> None:
    filename, lineno = PrintFrameManager.get_info_caller_frame()
    print("In FILE \'%s\' and LINE \'%s\':" % (filename, lineno))
    print("ERROR: %s... EXIT" % (message))
    sys.exit(0)


def catch_warning_exception(message: str) -> None:
    filename, lineno = PrintFrameManager.get_info_caller_frame()
    print("In FILE \'%s\' and LINE \'%s\':" % (filename, lineno))
    print("WARNING: %s... CONTINUE" % (message))


def catch_error_exception_old(message: str) -> None:
    print("ERROR: %s... EXIT" % (message))
    sys.exit(0)


def catch_warning_exception_old(message: str) -> None:
    print("WARNING: %s... CONTINUE" % (message))
