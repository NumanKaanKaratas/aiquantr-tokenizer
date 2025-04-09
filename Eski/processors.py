# aiquantr_tokenizer/data/processors.py
"""
Metin verilerini işleme ve temizleme sınıfları.

Bu modül, ham metin verilerini tokenizer eğitimi için
hazırlayan işleme ve temizleme sınıflarını sağlar.
"""

import re
import html
import unicodedata
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Callable, Set

# Logger oluştur
logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Tüm metin işleyicileri için temel sınıf.
    
    Bu soyut temel sınıf, metin işleyicileri için ortak
    arayüzü ve işlevselliği tanımlar.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        BaseProcessor sınıfı başlatıcısı.
        
        Args:
            name: İşleyici adı (varsayılan: None - sınıf adı kullanılır)
        """
        self.name = name or self.__class__.__name__
        self.stats = {
            "processed_count": 0,
            "total_chars_in": 0,
            "total_chars_out": 0
        }
    
    @abstractmethod
    def process(self, text: str) -> str:
        """
        Metin üzerinde işleme yapar.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: İşlenmiş metin
        """
        pass
    
    def __call__(self, text: str) -> str:
        """
        Processor'u doğrudan bir fonksiyon gibi çağırılabilir yapar.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: İşlenmiş metin
        """
        # İstatistikleri güncelle
        self.stats["processed_count"] += 1
        self.stats["total_chars_in"] += len(text)
        
        # İşleme yap
        result = self.process(text)
        
        # Çıktı istatistiklerini güncelle
        self.stats["total_chars_out"] += len(result)
        
        return result
    
    def reset_stats(self) -> None:
        """İstatistikleri sıfırlar."""
        self.stats = {
            "processed_count": 0,
            "total_chars_in": 0,
            "total_chars_out": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        İşleyici istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistikler sözlüğü
        """
        stats = dict(self.stats)
        
        # Ortalama karakter değişikliği
        if stats["processed_count"] > 0:
            stats["avg_chars_in"] = stats["total_chars_in"] / stats["processed_count"]
            stats["avg_chars_out"] = stats["total_chars_out"] / stats["processed_count"]
            
            # Toplam karakter değişimi yüzdesi
            if stats["total_chars_in"] > 0:
                stats["char_change_percent"] = (stats["total_chars_out"] - stats["total_chars_in"]) / stats["total_chars_in"] * 100
            
        return stats


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
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
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


class CodeProcessor(BaseProcessor):
    """
    Kod dosyaları için özel işleme sınıfı.
    
    Bu sınıf, programlama dili dosyalarını işleyerek
    tokenizer eğitimi için hazırlar.
    """
    
    def __init__(
        self,
        remove_comments: bool = False,
        normalize_whitespace: bool = True,
        remove_docstrings: bool = False,
        remove_string_literals: bool = False,
        keep_indentation: bool = True,
        remove_shebang: bool = True,
        min_code_length: int = 0,
        max_code_length: Optional[int] = None,
        language_specific_rules: Optional[Dict[str, Dict[str, Any]]] = None,
        name: Optional[str] = None
    ):
        """
        CodeProcessor sınıfı başlatıcısı.
        
        Args:
            remove_comments: Yorumları kaldır (varsayılan: False)
            normalize_whitespace: Boşlukları normalize et (varsayılan: True)
            remove_docstrings: Belge dizilerini kaldır (varsayılan: False)
            remove_string_literals: Dize değişmezleri kaldır (varsayılan: False)
            keep_indentation: Girintileri koru (varsayılan: True)
            remove_shebang: Shebang satırlarını kaldır (varsayılan: True)
            min_code_length: Minimum kod uzunluğu (varsayılan: 0)
            max_code_length: Maksimum kod uzunluğu (varsayılan: None)
            language_specific_rules: Dil özel kurallar (varsayılan: None)
            name: İşleyici adı (varsayılan: None)
        """
        super().__init__(name=name)
        
        self.remove_comments = remove_comments
        self.normalize_whitespace = normalize_whitespace
        self.remove_docstrings = remove_docstrings
        self.remove_string_literals = remove_string_literals
        self.keep_indentation = keep_indentation
        self.remove_shebang = remove_shebang
        self.min_code_length = min_code_length
        self.max_code_length = max_code_length
        self.language_specific_rules = language_specific_rules or {}
        
        # Düzenli ifadeler
        self.shebang_pattern = re.compile(r'^#!.*?$', re.MULTILINE)
        self.line_comment_patterns = {
            "python": re.compile(r'#.*?$', re.MULTILINE),
            "javascript": re.compile(r'//.*?$', re.MULTILINE),
            "c": re.compile(r'//.*?$', re.MULTILINE),
            "java": re.compile(r'//.*?$', re.MULTILINE)
        }
        self.block_comment_patterns = {
            "javascript": (re.compile(r'/\*.*?\*/', re.DOTALL), ''),
            "c": (re.compile(r'/\*.*?\*/', re.DOTALL), ''),
            "java": (re.compile(r'/\*.*?\*/', re.DOTALL), ''),
            "css": (re.compile(r'/\*.*?\*/', re.DOTALL), '')
        }
        self.docstring_patterns = {
            "python": (re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL), ''),
        }
        self.whitespace_pattern = re.compile(r'[ \t]+')
    
    def process(self, text: str, language: str = None) -> str:
        """
        Kod üzerinde işleme yapar.
        
        Args:
            text: İşlenecek kod metni
            language: Programlama dili (varsayılan: None)
            
        Returns:
            str: İşlenmiş kod metni
        """
        if not text:
            return ""
        
        # Shebang satırlarını kaldır
        if self.remove_shebang:
            text = re.sub(self.shebang_pattern, '', text)
        
        # Dil özel kuralları uygula
        if language and language in self.language_specific_rules:
            rules = self.language_specific_rules[language]
            for rule_name, rule_value in rules.items():
                if hasattr(self, rule_name) and isinstance(getattr(self, rule_name), bool):
                    setattr(self, rule_name, rule_value)
        
        # Yorumları kaldır
        if self.remove_comments:
            # Satır yorumları
            if language in self.line_comment_patterns:
                text = re.sub(self.line_comment_patterns[language], '', text)
            
            # Blok yorumları
            if language in self.block_comment_patterns:
                pattern, replacement = self.block_comment_patterns[language]
                text = re.sub(pattern, replacement, text)
        
        # Belgeleme dizilerini kaldır
        if self.remove_docstrings and language in self.docstring_patterns:
            pattern, replacement = self.docstring_patterns[language]
            text = re.sub(pattern, replacement, text)
        
        # Dize değişmezlerini kaldır
        if self.remove_string_literals:
            # Basit dize değişmezi deseni - daha karmaşık olanlar için dil özel işleyiciler gerekli
            text = re.sub(r'".*?"|\'.*?\'', '""', text)
        
        # Boşlukları normalize et
        if self.normalize_whitespace:
            if self.keep_indentation:
                # Satırlara böl, her satırda boşlukları normalize et ve tekrar birleştir
                lines = text.splitlines()
                normalized_lines = []
                for line in lines:
                    # Satır başındaki boşlukları koru
                    indent_match = re.match(r'^[ \t]*', line)
                    indent = indent_match.group(0) if indent_match else ""
                    
                    # Satırın geri kalanında boşlukları normalize et
                    rest_of_line = line[len(indent):].strip()
                    rest_of_line = re.sub(self.whitespace_pattern, ' ', rest_of_line)
                    
                    normalized_lines.append(f"{indent}{rest_of_line}")
                
                text = '\n'.join(normalized_lines)
            else:
                # Tüm boşlukları normalize et
                text = re.sub(self.whitespace_pattern, ' ', text)
                text = text.strip()
        
        # Uzunluk kontrolü
        if self.min_code_length > 0 and len(text) < self.min_code_length:
            return ""
            
        if self.max_code_length and len(text) > self.max_code_length:
            text = text[:self.max_code_length]
        
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
    
    def _hash_text(self, text: str) -> Union[str, Any]:
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


class ProcessingPipeline(BaseProcessor):
    """
    Birden fazla işleyiciyi sırayla uygulayan işleme hattı.
    
    Bu sınıf, bir dizi işleyiciyi zincir halinde uygulayarak
    metinleri adım adım işler.
    """
    
    def __init__(
        self,
        processors: List[BaseProcessor],
        skip_empty: bool = True,
        name: Optional[str] = None
    ):
        """
        ProcessingPipeline sınıfı başlatıcısı.
        
        Args:
            processors: Uygulanacak işleyiciler listesi
            skip_empty: Boş metinleri işlemeden geç (varsayılan: True)
            name: İşleme hattı adı (varsayılan: None)
        """
        super().__init__(name=name or "ProcessingPipeline")
        
        if not processors:
            raise ValueError("İşleyiciler listesi boş olamaz.")
            
        self.processors = processors
        self.skip_empty = skip_empty
        
        # İşleme hattı istatistikleri
        self.stats.update({
            "skipped_empty": 0,
            "final_empty": 0
        })
    
    def process(self, text: str) -> str:
        """
        Metni işleme hattından geçirir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Tüm işleyicilerden geçirilmiş metin
        """
        if not text and self.skip_empty:
            self.stats["skipped_empty"] += 1
            return ""
            
        result = text
        
        # Her işleyiciyi sırayla uygula
        for processor in self.processors:
            result = processor(result)
            
            # Eğer metin boş hale geldiyse ve boş metinleri atla seçeneği etkinse
            if not result and self.skip_empty:
                self.stats["final_empty"] += 1
                break
                
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        İşleme hattı ve içerdiği işleyicilerin istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İşleme hattı istatistikleri
        """
        stats = super().get_stats()
        
        # Her işleyicinin istatistiklerini ekle
        for i, processor in enumerate(self.processors):
            stats[f"processor_{i}"] = {
                "name": processor.name,
                "type": processor.__class__.__name__,
                "stats": processor.get_stats()
            }
            
        return stats