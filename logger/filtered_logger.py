import os
from enum import Enum

from env_utils import parse_bool_env

class LogChannel(Enum):
    GLOBAL = "GLOBAL"
    YOLO = "YOLO"
    DENSITY = "DENSITY"
    NVDEC = "NVDEC"


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARN"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class FilteredLogger:
    def __init__(self):
        self.extreme_debug = parse_bool_env('EXTREME_DEBUG', '0')
        self.yolo_debug = parse_bool_env('YOLO_DEBUG_LOGS', '0')
        self.density_debug = parse_bool_env('DENSITY_DEBUG_LOGS', '0')
        self.nvdec_debug = parse_bool_env('NVDEC_DEBUG_LOGS', '1')

    def configure(self, *, extreme_debug=None, yolo_debug=None, density_debug=None, nvdec_debug=None):
        if extreme_debug is not None:
            self.extreme_debug = extreme_debug
        if yolo_debug is not None:
            self.yolo_debug = yolo_debug
        if density_debug is not None:
            self.density_debug = density_debug
        if nvdec_debug is not None:
            self.nvdec_debug = nvdec_debug

    def should_log_debug(self, channel):
        if channel == LogChannel.GLOBAL:
            return self.extreme_debug or self.yolo_debug or self.density_debug or self.nvdec_debug
        if channel == LogChannel.YOLO:
            return self.yolo_debug
        if channel == LogChannel.DENSITY:
            return self.density_debug
        if channel == LogChannel.NVDEC:
            return self.nvdec_debug
        return False

    def _print(self, level, channel, message):
        prefix = f"[{level.value}]"
        channel_tag = f"[{channel.value}]"
        for line in str(message).splitlines():
            print(f"{prefix} {channel_tag} {line}")

    def info(self, channel, message):
        self._print(LogLevel.INFO, channel, message)

    def warning(self, channel, message):
        self._print(LogLevel.WARNING, channel, message)

    def error(self, channel, message):
        self._print(LogLevel.ERROR, channel, message)

    def debug(self, channel, message):
        if not self.should_log_debug(channel):
            return
        self._print(LogLevel.DEBUG, channel, message)


_shared_logger = FilteredLogger()


def configure_logger(**kwargs):
    _shared_logger.configure(**kwargs)


def info(channel, message):
    _shared_logger.info(channel, message)


def warning(channel, message):
    _shared_logger.warning(channel, message)


def error(channel, message):
    _shared_logger.error(channel, message)


def debug(channel, message):
    _shared_logger.debug(channel, message)
