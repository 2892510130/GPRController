import logging
import os


_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class Logger:
    """One-line-per-call file logger backed by the standard logging module."""

    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        parent = os.path.dirname(file_name)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._logger = logging.getLogger(f"gprcontroller.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._handler = logging.FileHandler(file_name, mode="w", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(self._handler)
        self._closed = False

    def write_data_to_file(self, level: str, message: str, data: object) -> None:
        self._logger.log(
            _LOG_LEVELS.get(level, logging.DEBUG), f"[{level}] {message}{data}"
        )

    def info(self, message: str, data: object) -> None:
        self.write_data_to_file("INFO", message, data)

    def debug(self, message: str, data: object) -> None:
        self.write_data_to_file("DEBUG", message, data)

    def empty_line(self, times: int) -> None:
        for _ in range(times):
            self._logger.debug("")

    def close_log(self) -> None:
        if self._closed:
            return
        print("[Logger] closing the log file")
        self._handler.flush()
        self._handler.close()
        self._logger.removeHandler(self._handler)
        self._closed = True
