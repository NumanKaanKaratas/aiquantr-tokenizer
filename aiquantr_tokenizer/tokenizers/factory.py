# aiquantr_tokenizer/tokenizers/factory.py
"""
Tokenizer fabrikası işlevleri.

Bu modül, yapılandırma sözlüklerinden veya dosyalardan
tokenizer nesneleri oluşturmak için fabrika işlevleri sağlar.
"""

import os
import time
import json
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Union, Optional, Type, List, Callable

from .base import BaseTokenizer
from .byte_level import ByteLevelTokenizer
from .bpe import BPETokenizer
from .wordpiece import WordPieceTokenizer
from .unigram import UnigramTokenizer
from .sentencepiece import SentencePieceTokenizer
from .mixed import MixedTokenizer

# Logger oluştur
logger = logging.getLogger(__name__)

# Desteklenen tokenizer türleri
TOKENIZER_TYPES = {
    "bytelevel": ByteLevelTokenizer,
    "byte_level": ByteLevelTokenizer,
    "bpe": BPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
    "sentencepiece": SentencePieceTokenizer,
    "mixed": MixedTokenizer
}


def create_tokenizer_from_config(config: Dict[str, Any]) -> BaseTokenizer:
    """
    Yapılandırma sözlüğünden bir tokenizer oluşturur.
    
    Args:
        config: Tokenizer yapılandırması
        
    Returns:
        BaseTokenizer: Oluşturulan tokenizer
        
    Raises:
        ValueError: Yapılandırma geçerli değilse
    """
    if not isinstance(config, dict):
        raise ValueError("Yapılandırma bir sözlük olmalıdır.")
    
    # Tokenizer türünü belirle
    tokenizer_type = config.get("type", "").lower()
    if not tokenizer_type:
        raise ValueError("Yapılandırmada 'type' alanı eksik.")
    
    # Tokenizer sınıfını bul
    tokenizer_class = TOKENIZER_TYPES.get(tokenizer_type)
    if tokenizer_class is None:
        raise ValueError(f"Bilinmeyen tokenizer türü: {tokenizer_type}")
    
    # Tokenizer'ı oluştur
    try:
        # Özel parametre durumları
        if tokenizer_type == "mixed":
            # Karma tokenizer için alt tokenizer'ları oluştur
            sub_tokenizers = {}
            tokenizer_configs = config.get("tokenizers", {})
            
            for tokenizer_name, tokenizer_config in tokenizer_configs.items():
                sub_tokenizers[tokenizer_name] = create_tokenizer_from_config(tokenizer_config)
                
            # Router fonksiyonu oluştur
            router = None
            router_config = config.get("router")
            if router_config:
                router = _create_router_from_config(router_config)
                
            return MixedTokenizer(
                tokenizers=sub_tokenizers,
                default_tokenizer=config.get("default_tokenizer"),
                router=router,
                merged_vocab=config.get("merged_vocab", False),
                special_tokens=config.get("special_tokens"),
                name=config.get("name")
            )
        
        # Diğer tokenizer türleri için standart parametre geçişi
        kwargs = {k: v for k, v in config.items() if k != "type"}
        return tokenizer_class(**kwargs)
        
    except Exception as e:
        logger.error(f"Tokenizer oluşturulurken hata: {e}")
        raise


def load_tokenizer_from_path(path: Union[str, Path], tokenizer_type: Optional[str] = None) -> BaseTokenizer:
    """
    Belirtilen yoldan bir tokenizer yükler.
    
    Args:
        path: Tokenizer yolu
        tokenizer_type: Tokenizer türü (varsayılan: None - otomatik tespit)
        
    Returns:
        BaseTokenizer: Yüklenen tokenizer
        
    Raises:
        ValueError: Tokenizer yüklenemezse
    """
    path = Path(path)
    
    # Dosya türünü belirle
    if path.is_dir():
        # Dizin - yapılandırma dosyasını bul
        config_path = path / "tokenizer.json"
        if not config_path.exists():
            # SentencePiece modeli olabilir mi?
            model_path = path / "tokenizer.model"
            if model_path.exists():
                return SentencePieceTokenizer.load(path)
                
            # Diğer tokenizer türlerini dene
            for tokenizer_class in TOKENIZER_TYPES.values():
                try:
                    return tokenizer_class.load(path)
                except Exception:
                    continue
                    
            raise ValueError(f"Dizinde tanınabilir bir tokenizer bulunamadı: {path}")
    else:
        # Tek dosya
        if not path.exists():
            raise ValueError(f"Tokenizer dosyası bulunamadı: {path}")
            
        # Dosya uzantısına göre yüklemeyi dene
        if path.suffix == ".model":
            return SentencePieceTokenizer.load(path)
            
        elif path.suffix in [".json", ".jsonl"]:
            # JSON yapılandırmasını oku
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    
                # Tokenizer türünü belirle
                detected_type = config.get("type", "").lower()
                if detected_type:
                    tokenizer_type = detected_type
            except Exception as e:
                logger.warning(f"JSON yapılandırması okunamadı: {e}")
    
    # Tokenizer türü belirtilmişse veya tespit edildiyse
    if tokenizer_type:
        tokenizer_class = TOKENIZER_TYPES.get(tokenizer_type.lower())
        if tokenizer_class:
            return tokenizer_class.load(path)
            
    # Tüm tokenizer sınıflarını dene
    for t_type, tokenizer_class in TOKENIZER_TYPES.items():
        try:
            logger.debug(f"Tokenizer tipini deneme: {t_type}")
            return tokenizer_class.load(path)
        except Exception as e:
            logger.debug(f"{t_type} tokenizer yükleme başarısız: {e}")
            
    raise ValueError(f"Tokenizer yüklenemedi: {path}")


def _create_router_from_config(router_config: Union[str, Dict[str, Any]]) -> Callable[[str], str]:
    """
    Router fonksiyonu oluşturur.
    
    Args:
        router_config: Router yapılandırması
        
    Returns:
        Callable[[str], str]: Router fonksiyonu
        
    Raises:
        ValueError: Router oluşturulamazsa
    """
    if isinstance(router_config, str):
        # Basit router türünü belirle
        router_type = router_config
        router_params = {}
    elif isinstance(router_config, dict):
        # Yapılandırma sözlüğünden router bilgilerini çıkar
        router_type = router_config.get("type", "")
        router_params = {k: v for k, v in router_config.items() if k != "type"}
    else:
        raise ValueError("Geçersiz router yapılandırması")
    
    # Router türüne göre fonksiyon oluştur
    if router_type == "regex":
        return _create_regex_router(router_params)
    elif router_type == "length":
        return _create_length_router(router_params)
    elif router_type == "language":
        return _create_language_router(router_params)
    elif router_type == "custom":
        return _create_custom_router(router_params)
    else:
        raise ValueError(f"Bilinmeyen router türü: {router_type}")


def _create_regex_router(params: Dict[str, Any]) -> Callable[[str], str]:
    """
    Regex tabanlı router oluşturur.
    
    Args:
        params: Router parametreleri
        
    Returns:
        Callable[[str], str]: Router fonksiyonu
    """
    import re
    
    patterns = params.get("patterns", {})
    default = params.get("default", "")
    
    compiled_patterns = {
        tokenizer_name: re.compile(pattern) 
        for tokenizer_name, pattern in patterns.items()
    }
    
    def router(text: str) -> str:
        for tokenizer_name, pattern in compiled_patterns.items():
            if pattern.search(text):
                return tokenizer_name
        return default
        
    return router


def _create_length_router(params: Dict[str, Any]) -> Callable[[str], str]:
    """
    Uzunluk tabanlı router oluşturur.
    
    Args:
        params: Router parametreleri
        
    Returns:
        Callable[[str], str]: Router fonksiyonu
    """
    thresholds = sorted(params.get("thresholds", {}).items(), key=lambda x: x[1])
    default = params.get("default", "")
    
    def router(text: str) -> str:
        text_length = len(text)
        
        for tokenizer_name, threshold in thresholds:
            if text_length <= threshold:
                return tokenizer_name
                
        return default
        
    return router


def _create_language_router(params: Dict[str, Any]) -> Callable[[str], str]:
    """
    Dil tespiti tabanlı router oluşturur.
    
    Args:
        params: Router parametreleri
        
    Returns:
        Callable[[str], str]: Router fonksiyonu
        
    Raises:
        ImportError: Gerekli dil tespit kütüphanesi yoksa
    """
    try:
        import langdetect
    except ImportError:
        raise ImportError("Dil tespiti için 'langdetect' paketi gereklidir.")
        
    lang_map = params.get("language_map", {})
    default = params.get("default", "")
    
    def router(text: str) -> str:
        try:
            lang = langdetect.detect(text)
            return lang_map.get(lang, default)
        except Exception:
            return default
            
    return router


def _create_custom_router(params: Dict[str, Any]) -> Callable[[str], str]:
    """
    Özel bir modül ve fonksiyondan router oluşturur.
    
    Args:
        params: Router parametreleri
        
    Returns:
        Callable[[str], str]: Router fonksiyonu
        
    Raises:
        ValueError: Özel router yüklenemezse
    """
    module_path = params.get("module")
    function_name = params.get("function")
    
    if not module_path or not function_name:
        raise ValueError("Özel router için 'module' ve 'function' alanları gereklidir.")
        
    try:
        module = importlib.import_module(module_path)
        router_func = getattr(module, function_name)
        return router_func
    except Exception as e:
        raise ValueError(f"Özel router yüklenemedi: {e}")


def register_tokenizer_type(name: str, tokenizer_class: Type[BaseTokenizer]) -> None:
    """
    Yeni bir tokenizer türünü kaydeder.
    
    Args:
        name: Tokenizer türü adı
        tokenizer_class: Tokenizer sınıfı
    """
    global TOKENIZER_TYPES
    TOKENIZER_TYPES[name.lower()] = tokenizer_class
    logger.info(f"Tokenizer türü kaydedildi: {name}")