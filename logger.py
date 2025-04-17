import logging
import os
import sys
import traceback
from datetime import datetime

# Создаем директорию для логов, если она не существует
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Имя файла лога с текущей датой и временем
log_file = os.path.join(log_dir, f'3ddfa_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Цвета для форматирования в терминале
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Обычные цвета
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Яркие цвета
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Фоновые цвета
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

# Эмодзи для разных уровней логирования
class LogEmoji:
    DEBUG = "🔍"
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🔥"
    SUCCESS = "✅"
    PROCESSING = "⚙️"
    FACE = "👤"
    CAMERA = "📷"
    SAVE = "💾"
    METRICS = "📊"
    TIME = "⏱️"

# Кастомный форматтер с поддержкой цветов и эмодзи
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: LogColors.BRIGHT_BLACK + LogEmoji.DEBUG + " %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s" + LogColors.RESET,
        logging.INFO: LogColors.BRIGHT_BLUE + LogEmoji.INFO + " %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s" + LogColors.RESET,
        logging.WARNING: LogColors.YELLOW + LogEmoji.WARNING + " %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s" + LogColors.RESET,
        logging.ERROR: LogColors.RED + LogEmoji.ERROR + " %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s" + LogColors.RESET,
        logging.CRITICAL: LogColors.BG_RED + LogColors.WHITE + LogEmoji.CRITICAL + " %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s" + LogColors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Настройка корневого логгера
def setup_logger():
    # Создаем логгер
    logger = logging.getLogger('3ddfa')
    logger.setLevel(logging.DEBUG)
    
    # Очищаем существующие обработчики, если они есть
    if logger.handlers:
        logger.handlers.clear()

    # Создаем обработчик для записи в файл (без цветов, но с эмодзи)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Создаем обработчик для вывода в консоль (с цветами и эмодзи)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Глобальный логгер для использования во всем проекте
logger = setup_logger()

# Функция для логирования исключений с полным трейсбеком
def log_exception(e, message="Произошла ошибка"):
    """
    Логирует исключение с полным трейсбеком
    """
    logger.error(f"{LogEmoji.ERROR} {message}: {str(e)}")
    logger.error(f"{LogEmoji.ERROR} Трейсбек:\n" + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))

# Декоратор для логирования вызовов функций
def log_function_call(func):
    """
    Декоратор для логирования вызовов функций
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"{LogEmoji.PROCESSING} Вызов функции {func.__name__} с аргументами: args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{LogEmoji.SUCCESS} Функция {func.__name__} успешно выполнена")
            return result
        except Exception as e:
            log_exception(e, f"Ошибка в функции {func.__name__}")
            raise
    return wrapper

# Дополнительные функции для удобного логирования с эмодзи
def log_success(message):
    """Логирует успешное завершение операции"""
    logger.info(f"{LogEmoji.SUCCESS} {message}")

def log_processing(message):
    """Логирует информацию о текущем процессе"""
    logger.info(f"{LogEmoji.PROCESSING} {message}")

def log_face(message):
    """Логирует информацию о лице"""
    logger.info(f"{LogEmoji.FACE} {message}")

def log_metrics(message):
    """Логирует информацию о метриках"""
    logger.info(f"{LogEmoji.METRICS} {message}")

def log_save(message):
    """Логирует информацию о сохранении файлов"""
    logger.info(f"{LogEmoji.SAVE} {message}")

def log_time(message):
    """Логирует информацию о времени выполнения"""
    logger.info(f"{LogEmoji.TIME} {message}")
