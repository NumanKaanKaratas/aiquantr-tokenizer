"""
Veri filtreleme modülleri.

Bu modül, veri kümeleri üzerinde filtreleme işlemleri yapmak
için fonksiyonlar ve sınıflar sağlar.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Iterable, Set
from pathlib import Path

from aiquantr_tokenizer.processors import BaseProcessor
from aiquantr_tokenizer.processors.language_processor import DuplicateRemover

# Logger oluştur
logger = logging.getLogger(__name__)


class TextFilter(BaseProcessor):
    """
    Metin filtreleme işlemcisi.
    
    Bu sınıf, çeşitli kriterlere göre metinleri filtrelemek için kullanılır.
    """
    
    def __init__(
        self,
        min_length: int = 0,
        max_length: Optional[int] = None,
        min_words: int = 0,
        max_words: Optional[int] = None,
        required_words: Optional[List[str]] = None,
        excluded_words: Optional[List[str]] = None,
        required_patterns: Optional[List[str]] = None,
        excluded_patterns: Optional[List[str]] = None,
        language_detection: bool = False,
        allowed_languages: Optional[List[str]] = None,
        custom_filter: Optional[Callable[[str], bool]] = None,
        name: Optional[str] = None
    ):
        """
        TextFilter sınıfı başlatıcısı.
        
        Args:
            min_length: Minimum karakter sayısı (varsayılan: 0)
            max_length: Maksimum karakter sayısı (varsayılan: None)
            min_words: Minimum kelime sayısı (varsayılan: 0)
            max_words: Maksimum kelime sayısı (varsayılan: None)
            required_words: İçermesi gereken kelimeler (varsayılan: None)
            excluded_words: İçermemesi gereken kelimeler (varsayılan: None)
            required_patterns: İçermesi gereken regex desenleri (varsayılan: None)
            excluded_patterns: İçermemesi gereken regex desenleri (varsayılan: None)
            language_detection: Dil algılama kullan (varsayılan: False)
            allowed_languages: İzin verilen diller (varsayılan: None)
            custom_filter: Özel filtre fonksiyonu (varsayılan: None)
            name: İşlemci adı (varsayılan: None)
        """
        super().__init__(name=name)
        
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_words = max_words
        self.required_words = set(w.lower() for w in required_words) if required_words else None
        self.excluded_words = set(w.lower() for w in excluded_words) if excluded_words else None
        self.custom_filter = custom_filter
        
        # Regex desenlerini derle
        self.required_patterns = [re.compile(p) for p in required_patterns] if required_patterns else None
        self.excluded_patterns = [re.compile(p) for p in excluded_patterns] if excluded_patterns else None
        
        # Dil algılama
        self.language_detection = language_detection
        self.allowed_languages = set(allowed_languages) if allowed_languages else None
        
        if language_detection:
            try:
                import fasttext
                # Model dosyasını kontrol et veya indir
                model_path = Path(__file__).parent / "lid.176.bin"
                if not model_path.exists():
                    logger.warning("Dil algılama modeli bulunamadı. İndiriliyor...")
                    # Model indirme kodları burada olacak
                    # Şimdilik sadece uyarı ver
                    logger.error("Dil algılama modeli bulunamadı. Lütfen manuel olarak yükleyin.")
                    self.language_detection = False
                else:
                    self.lang_model = fasttext.load_model(str(model_path))
            except ImportError:
                logger.warning("fasttext paketi bulunamadı. Dil algılama devre dışı.")
                self.language_detection = False
        
        # İstatistikler
        self.stats.update({
            "filtered_length": 0,
            "filtered_words": 0,
            "filtered_required_words": 0,
            "filtered_excluded_words": 0,
            "filtered_required_patterns": 0,
            "filtered_excluded_patterns": 0,
            "filtered_language": 0,
            "filtered_custom": 0
        })
    
    def detect_language(self, text: str) -> str:
        """
        Metnin dilini algılar.
        
        Args:
            text: Dili algılanacak metin
            
        Returns:
            str: ISO dil kodu (örn. "en", "tr")
        """
        if not hasattr(self, "lang_model"):
            return "unknown"
        
        # En az 20 karakter gerekli
        if len(text) < 20:
            return "unknown"
            
        # fasttext ile dil algıla
        predictions = self.lang_model.predict(text.replace("\n", " "))
        lang_code = predictions[0][0].replace("__label__", "")
        return lang_code
    
    def process(self, text: str) -> str:
        """
        Metni filtreleme ölçütlerine göre işler.
        
        Filtreden geçemeyen metinler boş string olarak döndürülür.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Filtreden geçen metin veya boş string
        """
        if not text:
            return ""
        
        # Uzunluk filtresi
        if self.min_length > 0 and len(text) < self.min_length:
            self.stats["filtered_length"] += 1
            return ""
            
        if self.max_length and len(text) > self.max_length:
            self.stats["filtered_length"] += 1
            return ""
        
        # Kelime sayısı filtresi
        words = text.split()
        if self.min_words > 0 and len(words) < self.min_words:
            self.stats["filtered_words"] += 1
            return ""
            
        if self.max_words and len(words) > self.max_words:
            self.stats["filtered_words"] += 1
            return ""
        
        # Gereken kelimeler filtresi
        if self.required_words:
            text_lower = text.lower()
            if not all(word in text_lower for word in self.required_words):
                self.stats["filtered_required_words"] += 1
                return ""
        
        # Dışlanan kelimeler filtresi
        if self.excluded_words:
            text_lower = text.lower()
            if any(word in text_lower for word in self.excluded_words):
                self.stats["filtered_excluded_words"] += 1
                return ""
        
        # Gereken desen filtresi
        if self.required_patterns:
            if not all(pattern.search(text) for pattern in self.required_patterns):
                self.stats["filtered_required_patterns"] += 1
                return ""
                
        # Dışlanan desen filtresi
        if self.excluded_patterns:
            if any(pattern.search(text) for pattern in self.excluded_patterns):
                self.stats["filtered_excluded_patterns"] += 1
                return ""
        
        # Dil filtresi
        if self.language_detection and self.allowed_languages:
            lang = self.detect_language(text)
            if lang not in self.allowed_languages:
                self.stats["filtered_language"] += 1
                return ""
        
        # Özel filtre
        if self.custom_filter and not self.custom_filter(text):
            self.stats["filtered_custom"] += 1
            return ""
        
        return text


def filter_dataset(
    texts: Iterable[str],
    min_length: int = 0,
    max_length: Optional[int] = None,
    deduplication: bool = True,
    dedup_similarity: float = 1.0,
    language_filter: bool = False,
    languages: Optional[List[str]] = None
) -> List[str]:
    """
    Hızlı veri kümesi filtreleme fonksiyonu.
    
    Args:
        texts: Metin koleksiyonu
        min_length: Minimum metin uzunluğu (varsayılan: 0)
        max_length: Maksimum metin uzunluğu (varsayılan: None)
        deduplication: Yinelenen örnekleri kaldır (varsayılan: True)
        dedup_similarity: Yineleme benzerlik eşiği (varsayılan: 1.0)
        language_filter: Dil filtrelemesi yap (varsayılan: False)
        languages: İzin verilen diller (varsayılan: None)
        
    Returns:
        List[str]: Filtrelenmiş metin listesi
    """
    # Filtre işlemcisi oluştur
    text_filter = TextFilter(
        min_length=min_length,
        max_length=max_length,
        language_detection=language_filter,
        allowed_languages=languages
    )
    
    # İlk filtreleme
    filtered_texts = []
    for text in texts:
        result = text_filter(text)
        if result:
            filtered_texts.append(result)
    
    # Yineleme kaldırma
    if deduplication:
        deduplicator = DuplicateRemover(min_similarity=dedup_similarity)
        dedup_texts = []
        
        for text in filtered_texts:
            result = deduplicator(text)
            if result:
                dedup_texts.append(result)
        
        return dedup_texts
    
    return filtered_texts


def remove_similar_samples(
    texts: List[str], 
    similarity_threshold: float = 0.85,
    method: str = "minhash"
) -> List[str]:
    """
    Benzer örnekleri kaldır.
    
    Args:
        texts: Metin listesi
        similarity_threshold: Benzerlik eşiği (varsayılan: 0.85)
        method: Benzerlik hesaplama yöntemi (varsayılan: "minhash")
    
    Returns:
        List[str]: Benzerlikleri kaldırılmış liste
    """
    deduplicator = DuplicateRemover(
        hash_method=method,
        min_similarity=similarity_threshold
    )
    
    result = []
    for text in texts:
        processed = deduplicator(text)
        if processed:
            result.append(processed)
            
    return result