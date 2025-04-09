# aiquantr_tokenizer/tokenizers/__init__.py
"""
Tokenizer tanımları ve eğitim fonksiyonları.

Bu modül, çeşitli tokenizer modellerini ve bunların
eğitim ve değerlendirme fonksiyonlarını sağlar.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

from .base import BaseTokenizer, TokenizerTrainer
from .byte_level import ByteLevelTokenizer
from .bpe import BPETokenizer
from .wordpiece import WordPieceTokenizer
from .unigram import UnigramTokenizer
from .sentencepiece import SentencePieceTokenizer
from .mixed import MixedTokenizer

__all__ = [
    # Temel sınıflar
    "BaseTokenizer",
    "TokenizerTrainer",
    
    # Tokenizer türleri
    "ByteLevelTokenizer",
    "BPETokenizer",
    "WordPieceTokenizer",
    "UnigramTokenizer",
    "SentencePieceTokenizer",
    "MixedTokenizer",
    
    # Yardımcı işlevler
    "load_tokenizer",
    "create_tokenizer",
    "evaluate_tokenizer"
]

# Logger oluştur
logger = logging.getLogger(__name__)


def create_tokenizer(config: Dict[str, Any]) -> BaseTokenizer:
    """
    Yapılandırmaya göre bir tokenizer oluşturur.
    
    Args:
        config: Tokenizer yapılandırması
        
    Returns:
        BaseTokenizer: Oluşturulan tokenizer
        
    Raises:
        ValueError: Yapılandırma geçerli değilse
    """
    from .factory import create_tokenizer_from_config
    
    tokenizer = create_tokenizer_from_config(config)
    logger.info(f"Tokenizer oluşturuldu: {tokenizer.__class__.__name__}")
    return tokenizer


def load_tokenizer(path: Union[str, Path], tokenizer_type: Optional[str] = None) -> BaseTokenizer:
    """
    Disk veya Hugging Face Hub'dan bir tokenizer yükler.
    
    Args:
        path: Tokenizer yolu veya Hugging Face model kimliği
        tokenizer_type: Tokenizer türü (varsayılan: None - otomatik tespit)
        
    Returns:
        BaseTokenizer: Yüklenen tokenizer
        
    Raises:
        ValueError: Tokenizer yüklenemezse
    """
    from .factory import load_tokenizer_from_path
    
    tokenizer = load_tokenizer_from_path(path, tokenizer_type)
    logger.info(f"Tokenizer yüklendi: {tokenizer.__class__.__name__}")
    return tokenizer


def evaluate_tokenizer(
    tokenizer: BaseTokenizer,
    texts: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Tokenizer'ı değerlendirir.
    
    Args:
        tokenizer: Değerlendirilecek tokenizer
        texts: Test metinleri
        metrics: Hesaplanacak metrikler (varsayılan: None - tüm metrikler)
        
    Returns:
        Dict[str, Any]: Değerlendirme sonuçları
    """
    from .evaluation import evaluate_tokenizer as evaluate_func
    
    results = evaluate_func(tokenizer, texts, metrics)
    logger.info(f"Tokenizer değerlendirmesi tamamlandı: {len(texts)} örnek üzerinde")
    return results