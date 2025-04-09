# aiquantr_tokenizer/utils/logging_utils.py
"""
Loglama yardımcı işlevleri.

Bu modül, uygulama genelinde tutarlı loglama yapılandırması için
gerekli fonksiyonları ve yardımcı araçları sağlar.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Union


# Varsayılan log formatı
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log seviye eşleşmeleri
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


def setup_logger(
    name: str = "aiquantr_tokenizer",
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    file_level: Optional[Union[str, int]] = None,
    console_level: Optional[Union[str, int]] = None,
    propagate: bool = False
) -> logging.Logger:
    """
    Belirtilen isim için yapılandırılmış bir logger döndürür.
    
    Args:
        name: Logger adı
        level: Genel log seviyesi (varsayılan: "INFO")
        log_file: Log dosyasının yolu (varsayılan: None - dosyaya yazılmaz)
        log_format: Log formatı (varsayılan: "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        date_format: Tarih formatı (varsayılan: "%Y-%m-%d %H:%M:%S")
        file_level: Dosya logları için seviye (varsayılan: None - genel seviye kullanılır)
        console_level: Konsol logları için seviye (varsayılan: None - genel seviye kullanılır)
        propagate: Logger'ın üst loggerlara iletilip iletilmeyeceği (varsayılan: False)
        
    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi
    """
    # Seviyeyi çözümle
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    # Dosya ve konsol seviyelerini belirle
    if file_level is None:
        file_level = level
    elif isinstance(file_level, str):
        file_level = LOG_LEVELS.get(file_level.upper(), level)
        
    if console_level is None:
        console_level = level
    elif isinstance(console_level, str):
        console_level = LOG_LEVELS.get(console_level.upper(), level)
    
    # Logger'ı oluştur
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Önceki handler'ları temizle (çift log önlemek için)
    if logger.handlers:
        logger.handlers.clear()
    
    # Formatlayıcıyı oluştur
    formatter = logging.Formatter(log_format, date_format)
    
    # Konsola log yazma handler'ı
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Dosyaya log yazma handler'ı (eğer log_file belirtilmişse)
    if log_file:
        # Log dosyasının dizininin varlığını kontrol et ve oluştur
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"Log dizini oluşturulamadı ({log_dir}): {e}")
                
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (PermissionError, IOError) as e:
            logger.error(f"Log dosyası oluşturulamadı ({log_file}): {e}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Mevcut bir logger'ı alır veya yeni bir logger oluşturur.
    
    Args:
        name: Logger adı (varsayılan: None - root logger)
        
    Returns:
        logging.Logger: İstenen logger
    """
    if name is None:
        return logging.getLogger("aiquantr_tokenizer")
    return logging.getLogger(name)


def log_execution_time(logger: logging.Logger, level: int = logging.INFO):
    """
    Bir fonksiyonun çalışma süresini ölçmek için dekoratör.
    
    Args:
        logger: Kullanılacak logger
        level: Log seviyesi (varsayılan: logging.INFO)
        
    Returns:
        callable: Dekoratör fonksiyon
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = end_time - start_time
            logger.log(level, f"{func.__name__} çalışma süresi: {duration}")
            return result
        return wrapper
    return decorator


def log_config(logger: logging.Logger, config: Dict[str, Any], level: int = logging.DEBUG):
    """
    Konfigürasyon detaylarını loglar.
    
    Args:
        logger: Kullanılacak logger
        config: Loglanacak konfigürasyon sözlüğü
        level: Log seviyesi (varsayılan: logging.DEBUG)
    """
    logger.log(level, "Konfigürasyon:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.log(level, f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (list, dict)):
                    logger.log(level, f"    {sub_key}: {type(sub_value).__name__} (len={len(sub_value)})")
                else:
                    logger.log(level, f"    {sub_key}: {sub_value}")
        elif isinstance(value, list):
            logger.log(level, f"  {key}: Liste (len={len(value)})")
        else:
            logger.log(level, f"  {key}: {value}")


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adaptörü, loglara ek bağlam ekler.
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """
        LoggerAdapter sınıfı başlatıcısı.
        
        Args:
            logger: Temel logger
            extra: Ekstra bağlam bilgileri (varsayılan: None)
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """
        Log mesajını işle ve ek bağlam bilgilerini ekle.
        
        Args:
            msg: Log mesajı
            kwargs: Ek parametreler
            
        Returns:
            tuple: (formatlanmış mesaj, kwargs)
        """
        context_str = " ".join(f"[{k}={v}]" for k, v in self.extra.items())
        if context_str:
            return f"{context_str} {msg}", kwargs
        return msg, kwargs


def add_timestamp_to_logs():
    """
    Mevcut loglama yapılandırmasına zaman damgası ekler.
    
    Bu fonksiyon, özellikle yapılandırma dosyası üzerinden
    loglama yapılandırılırken kullanışlıdır.
    """
    # Varsayılan tarih formatını al
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return record
    
    logging.setLogRecordFactory(record_factory)


# Uygulama başlangıcında zaman damgası ekle
add_timestamp_to_logs()