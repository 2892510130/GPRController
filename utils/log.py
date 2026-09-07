class Logger:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.log_handle = open(self.file_name, 'w')
    
    def write_data_to_file(self, level: str, message: str, data: object) -> None:
        output = f"[{level}-{message}{data}\n"
        self.log_handle.write(output)
    
    def info(self, message: str, data: object) -> None:
        self.write_data_to_file("INFO", message, data)
    
    def debug(self, message: str, data: object) -> None:
        self.write_data_to_file("DEBUG", message, data)
    
    def empty_line(self, times: int) -> None:
        self.log_handle.write("\n" * times)

    def close_log(self) -> None:
        print("[Logger] closing the log file")
        self.log_handle.close()