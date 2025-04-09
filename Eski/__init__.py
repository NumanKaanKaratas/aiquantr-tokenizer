"""
Tokenizer Prep: Tokenizer eğitimi için veri hazırlama kütüphanesi.
"""

from .version import __version__

# Alt paketleri içe aktar
import aiquantr_tokenizer.data
import aiquantr_tokenizer.processors
import aiquantr_tokenizer.metrics
import aiquantr_tokenizer.config
import aiquantr_tokenizer.core
import aiquantr_tokenizer.utils

# Temel sınıf ve fonksiyonları doğrudan erişilebilir yap
from aiquantr_tokenizer.data import clean_text, clean_code, filter_dataset
from aiquantr_tokenizer.processors import BaseProcessor
from aiquantr_tokenizer.processors.language_processor import TextProcessor
from aiquantr_tokenizer.processors.code import CodeProcessor, PythonProcessor, PhpProcessor
from aiquantr_tokenizer.metrics import calculate_text_diversity, calculate_token_distribution

# Dışa aktarılacak isimler
__all__ = [
    "__version__",
    "data",
    "processors", 
    "metrics",
    "config",
    "core",
    "utils",
    # Sık kullanılan fonksiyonlar
    "clean_text",
    "clean_code",
    "filter_dataset",
    "BaseProcessor",
    "TextProcessor", 
    "CodeProcessor",
    "PythonProcessor",
    "PhpProcessor",
    "calculate_text_diversity",
    "calculate_token_distribution"
]