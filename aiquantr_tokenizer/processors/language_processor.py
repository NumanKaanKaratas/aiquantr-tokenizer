"""
Doğal dil işleme işlemcileri.

Bu modül, metin verilerini (doğal dil) işlemek ve temizlemek için 
çeşitli sınıflar sağlar.
"""

import re
import html
import unicodedata
import logging
from typing import List, Dict, Any, Optional, Tuple, Pattern

from aiquantr_tokenizer.processors.base_processor import BaseProcessor

# Logger oluştur
logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    """
    Genel metin temizleme ve normalizasyon işleyicisi.
    
    Bu sınıf, metinleri temizlemek, normalize etmek ve
    tokenizer eğitimi için hazırlamak için çeşitli işlemleri sağlar.
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        fix_unicode: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        replace_urls: bool = False,
        replace_emails: bool = False,
        replace_numbers: bool = False,
        replace_digits: bool = False,
        replace_currency_symbols: bool = False,
        remove_punct: bool = False,
        remove_line_breaks: bool = False,
        fix_html: bool = True,
        min_text_length: int = 0,
        max_text_length: Optional[int] = None,
        # Test dosyasının beklediği eski parametreleri ekleyin
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        custom_replacements: Optional[List[Tuple[str, str]]] = None,
        name: Optional[str] = None
    ):
        """
        TextProcessor sınıfı başlatıcısı.
        
        Args:
            lowercase: Metni küçük harfe çevir (varsayılan: False)
            fix_unicode: Unicode karakterleri düzelt (varsayılan: True)
            normalize_unicode: Unicode karakterleri normalize et (varsayılan: True)
            normalize_whitespace: Boşlukları normalize et (varsayılan: True)
            replace_urls: URL'leri özel bir token ile değiştir (varsayılan: False)
            replace_emails: E-postaları özel bir token ile değiştir (varsayılan: False)
            replace_numbers: Sayıları özel bir token ile değiştir (varsayılan: False)
            replace_digits: Rakamları özel bir token ile değiştir (varsayılan: False)
            replace_currency_symbols: Para birimi sembollerini değiştir (varsayılan: False)
            remove_punct: Noktalama işaretlerini kaldır (varsayılan: False)
            remove_line_breaks: Satır sonlarını kaldır (varsayılan: False)
            fix_html: HTML karakterlerini düzelt (varsayılan: True)
            min_text_length: Minimum metin uzunluğu (varsayılan: 0)
            max_text_length: Maksimum metin uzunluğu (varsayılan: None)
            min_length: Eski API uyumluluğu için minimum metin uzunluğu (varsayılan: None)
            max_length: Eski API uyumluluğu için maksimum metin uzunluğu (varsayılan: None)
            custom_replacements: Özel değiştirme kuralları (varsayılan: None)
            name: İşleyici adı (varsayılan: None)
        """
        super().__init__(name=name)
        
        self.lowercase = lowercase
        self.fix_unicode = fix_unicode
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.replace_urls = replace_urls
        self.replace_emails = replace_emails
        self.replace_numbers = replace_numbers
        self.replace_digits = replace_digits
        self.replace_currency_symbols = replace_currency_symbols
        self.remove_punct = remove_punct
        self.remove_line_breaks = remove_line_breaks
        self.fix_html = fix_html
        
        # Hem eski hem yeni parametreleri destekle
        self.min_text_length = min_length if min_length is not None else min_text_length
        self.max_text_length = max_length if max_length is not None else max_text_length
        
        # min_length ve max_length takma adları için
        self.min_length = self.min_text_length
        self.max_length = self.max_text_length
        
        self.custom_replacements = custom_replacements or []
        
        # İstatistikler için ek alanlar
        self.stats.update({
            "texts_filtered_length": 0,
            "replaced_urls": 0,
            "replaced_emails": 0, 
            "replaced_numbers": 0,
            "replaced_currency": 0
        })
            
        # Düzenli ifadeler
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b')
        self.digit_pattern = re.compile(r'\d')
        self.currency_pattern = re.compile(r'[$€£¥₹₽₩]')
        self.punct_pattern = re.compile(r'[^\w\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.line_break_pattern = re.compile(r'[\r\n]+')
    
    def process(self, text: str) -> str:
        """
        Metin üzerinde işleme yapar.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: İşlenmiş metin
        """
        if not text:
            return ""
        
        # HTML düzeltme
        if self.fix_html:
            text = html.unescape(text)
        
        # Unicode düzeltme ve normalizasyon
        if self.fix_unicode:
            text = unicodedata.normalize('NFC', text)
        
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # URL değiştirme
        if self.replace_urls:
            url_count = len(re.findall(self.url_pattern, text))
            text = re.sub(self.url_pattern, ' [URL] ', text)
            self.stats["replaced_urls"] += url_count
        
        # E-posta değiştirme
        if self.replace_emails:
            email_count = len(re.findall(self.email_pattern, text))
            text = re.sub(self.email_pattern, ' [EMAIL] ', text)
            self.stats["replaced_emails"] += email_count
        
        # Sayı değiştirme
        if self.replace_numbers:
            number_count = len(re.findall(self.number_pattern, text))
            text = re.sub(self.number_pattern, ' [NUMBER] ', text)
            self.stats["replaced_numbers"] += number_count
        
        # Rakam değiştirme
        if self.replace_digits:
            text = re.sub(self.digit_pattern, '0', text)
        
        # Para birimi sembollerini değiştirme
        if self.replace_currency_symbols:
            currency_count = len(re.findall(self.currency_pattern, text))
            text = re.sub(self.currency_pattern, ' [CURRENCY] ', text)
            self.stats["replaced_currency"] += currency_count
        
        # Noktalama işaretlerini kaldır
        if self.remove_punct:
            text = re.sub(self.punct_pattern, ' ', text)
        
        # Satır sonlarını kaldır
        if self.remove_line_breaks:
            text = re.sub(self.line_break_pattern, ' ', text)
        
        # Boşlukları normalize et
        if self.normalize_whitespace:
            text = re.sub(self.whitespace_pattern, ' ', text)
            text = text.strip()
        
        # Küçük harfe çevir
        if self.lowercase:
            text = text.lower()
        
        # Özel değiştirmeler
        for pattern, replacement in self.custom_replacements:
            text = re.sub(pattern, replacement, text)
        
        # Uzunluk kontrolü
        if self.min_text_length > 0 and len(text) < self.min_text_length:
            self.stats["texts_filtered_length"] += 1
            return ""
            
        if self.max_text_length and len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            self.stats["texts_filtered_length"] += 1
        
        return text


class DuplicateRemover(BaseProcessor):
    """
    Metin örnekleri arasındaki yinelenenleri kaldıran işleyici.
    
    Bu sınıf, veri kümesindeki yinelenen örnekleri tanımlamak
    ve kaldırmak için kullanılır.
    """
    
    def __init__(
        self,
        hash_method: str = "exact",
        min_similarity: float = 1.0,
        case_sensitive: bool = True,
        whitespace_sensitive: bool = False,
        name: Optional[str] = None
    ):
        """
        DuplicateRemover sınıfı başlatıcısı.
        
        Args:
            hash_method: Metinleri karşılaştırma yöntemi (varsayılan: "exact")
                         Seçenekler: "exact", "lowercase", "normalized", "minhash"
            min_similarity: Bir metin kopya sayılması için minimum benzerlik (varsayılan: 1.0)
            case_sensitive: Büyük/küçük harf duyarlı eşleşmeler (varsayılan: True)
            whitespace_sensitive: Boşluk duyarlı eşleşmeler (varsayılan: False)
            name: İşleyici adı (varsayılan: None)
            
        Note:
            min_similarity 1.0 ise sadece tam eşleşmeler kaldırılır.
            1.0'dan küçük değerler için bulanık eşleşme yöntemleri kullanılır.
        """
        super().__init__(name=name)
        
        self.hash_method = hash_method.lower()
        self.min_similarity = min_similarity
        self.case_sensitive = case_sensitive
        self.whitespace_sensitive = whitespace_sensitive
        
        # Görülen metinleri izle
        self.seen_hashes = set()
        
        # MinHash kullan
        if self.hash_method == "minhash" and self.min_similarity < 1.0:
            try:
                import datasketch
                self.minhash_seed = 42
                self.minhash_hash_funcs = 128
                self.minhash_lsh = datasketch.MinHashLSH(threshold=self.min_similarity, num_perm=self.minhash_hash_funcs)
            except ImportError:
                logger.warning("datasketch paketi bulunamadı. MinHash metodu yerine 'normalized' kullanılıyor.")
                self.hash_method = "normalized"
        
        # İstatistikler
        self.stats.update({
            "duplicates_found": 0,
            "unique_texts": 0,
        })
    
    def _hash_text(self, text: str):
        """
        Metne özel bir hash değeri üretir.
        
        Args:
            text: Hash değeri üretilecek metin
            
        Returns:
            Union[str, Any]: Hash değeri
        """
        if not self.case_sensitive:
            text = text.lower()
            
        if not self.whitespace_sensitive:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Hash yöntemini seç
        if self.hash_method == "exact":
            return text
        
        elif self.hash_method == "lowercase":
            return text.lower()
            
        elif self.hash_method == "normalized":
            # Boşlukları ve noktalama işaretlerini kaldır
            normalized = re.sub(r'[^\w]', '', text.lower())
            return normalized
            
        elif self.hash_method == "minhash" and hasattr(self, "minhash_lsh"):
            import datasketch
            # Metin için MinHash oluştur
            m = datasketch.MinHash(num_perm=self.minhash_hash_funcs, seed=self.minhash_seed)
            # Kelimeleri ekle
            for word in re.split(r'\s+', text.lower()):
                m.update(word.encode('utf-8'))
            return m
            
        else:
            # Varsayılan olarak tam eşleşme kullan
            return text
    
    def is_duplicate(self, text: str) -> bool:
        """
        Metnin daha önce görülüp görülmediğini kontrol eder.
        
        Args:
            text: Kontrol edilecek metin
            
        Returns:
            bool: Metin bir yineleme ise True, değilse False
        """
        if not text:
            return False
            
        text_hash = self._hash_text(text)
        
        if self.hash_method == "minhash" and hasattr(self, "minhash_lsh"):
            # MinHash LSH'de benzer öğeleri sorgula
            result = self.minhash_lsh.query(text_hash)
            return len(result) > 0
        else:
            # Basit hash seti kontrolü
            return text_hash in self.seen_hashes
    
    def add_to_seen(self, text: str) -> None:
        """
        Bir metni görülen metinler kümesine ekler.
        
        Args:
            text: Eklenecek metin
        """
        if not text:
            return
            
        text_hash = self._hash_text(text)
        
        if self.hash_method == "minhash" and hasattr(self, "minhash_lsh"):
            # Eğer bu özel bir veri yapısı ise, doğru ekleme yöntemini kullan
            key = str(hash(text))  # Benzersiz bir anahtar oluştur
            self.minhash_lsh.insert(key, text_hash)
            self.seen_hashes.add(key)
        else:
            # Basit hash setine ekle
            self.seen_hashes.add(text_hash)
    
    def process(self, text: str) -> str:
        """
        Eğer metin yinelenen bir örnek ise boş metin döndürür.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Metin yineleniyorsa boş metin, değilse metnin kendisi
        """
        if not text:
            return ""
            
        if self.is_duplicate(text):
            self.stats["duplicates_found"] += 1
            return ""
            
        self.add_to_seen(text)
        self.stats["unique_texts"] += 1
        return text
    
    def reset(self) -> None:
        """Görülen metinler kümesini ve istatistikleri sıfırlar."""
        self.seen_hashes = set()
        if self.hash_method == "minhash" and hasattr(self, "minhash_lsh"):
            import datasketch
            self.minhash_lsh = datasketch.MinHashLSH(threshold=self.min_similarity, num_perm=self.minhash_hash_funcs)
        self.reset_stats()