# aiquantr_tokenizer/__init__.py
"""
Tokenizer hazırlama araçları paketi.

Bu paket, çeşitli tokenizer modellerini eğitmek, 
değerlendirmek ve kullanmak için araçlar sağlar.

Desteklenen tokenizer türleri:
- Byte-Level Tokenizer
- Byte-Pair Encoding (BPE) Tokenizer
- WordPiece Tokenizer
- Unigram Tokenizer
- SentencePiece (sarmalayıcı)
- Mixed (karma) Tokenizer
"""

__version__ = "0.1.0"
__author__ = "NumanKaanKaratas"
__date__ = "2025-04-08"

# Alt modülleri dışa aktar
from .tokenizers.base import BaseTokenizer, TokenizerTrainer
from .tokenizers.byte_level import ByteLevelTokenizer
from .tokenizers.bpe import BPETokenizer
from .tokenizers.wordpiece import WordPieceTokenizer
from .tokenizers.unigram import UnigramTokenizer
from .tokenizers.sentencepiece import SentencePieceTokenizer
from .tokenizers.mixed import MixedTokenizer
from .tokenizers.factory import (
    create_tokenizer_from_config,
    load_tokenizer_from_path,
    register_tokenizer_type
)

# Yardımcı işlevleri dışa aktar
from .utils import (
    # Loglama
    setup_logger,
    get_logger,
    
    # Genel yardımcı işlevler
    set_seed,
    
    # Dosya ve veri işlemleri
    ensure_dir,
    file_exists,
    read_file,
    write_file,
    load_json,
    save_json,
    
    # Veri yükleme ve işleme
    extract_texts,
    split_data,
    save_texts,
    count_tokens
)

# Eski load_data fonksiyonu yerine extract_texts kullanılmalıdır
load_data = extract_texts

# setup_logging fonksiyonu setup_logger ile değiştirildi
setup_logging = setup_logger