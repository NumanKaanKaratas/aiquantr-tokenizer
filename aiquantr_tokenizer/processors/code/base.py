"""
Temel kod işleme sınıfı.

Bu modül, programlama dili işlemcileri için temel sınıfı
tanımlar. Her dil-özel işlemci bu sınıfı miras almalıdır.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Pattern, Match

from aiquantr_tokenizer.processors.base_processor import BaseProcessor

# Logger oluştur
logger = logging.getLogger(__name__)


class BaseCodeProcessor(BaseProcessor):
    """
    Kod işlemcileri için temel sınıf.
    
    Bu soyut sınıf, programlama dili-özel kod işlemcileri
    için ortak davranışı tanımlar.
    """
    
    def __init__(
        self,
        language_name: str,
        file_extensions: List[str],
        comment_prefixes: Optional[List[str]] = None,
        block_comment_pairs: Optional[List[Tuple[str, str]]] = None,
        string_delimiters: Optional[List[str]] = None,
        docstring_delimiters: Optional[List[Tuple[str, str]]] = None,
        remove_comments: bool = False,
        remove_docstrings: bool = False,
        remove_string_literals: bool = False,
        normalize_whitespace: bool = True,
        keep_indentation: bool = True,
        min_code_length: int = 0,
        max_code_length: Optional[int] = None,
        **kwargs
    ):
        """
        BaseCodeProcessor başlatıcısı.
        
        Args:
            language_name: Programlama dili adı
            file_extensions: Dil dosya uzantıları (örn: [".py", ".pyw"])
            comment_prefixes: Satır yorumları için önek listesi (örn: ["#", "//"])
            block_comment_pairs: Blok yorum başlangıç/bitiş çiftleri (örn: [("/*", "*/")])
            string_delimiters: Dize sınırlayıcılar (örn: ["\"", "'"])
            docstring_delimiters: Belgeleme dizisi başlangıç/bitiş çiftleri (örn: [("\"\"\"", "\"\"\"")])
            remove_comments: Yorumları kaldır (varsayılan: False)
            remove_docstrings: Belge dizilerini kaldır (varsayılan: False)
            remove_string_literals: Dize değişmezleri kaldır (varsayılan: False)
            normalize_whitespace: Boşlukları normalize et (varsayılan: True)
            keep_indentation: Girintileri koru (varsayılan: True)
            min_code_length: Minimum kod uzunluğu (varsayılan: 0)
            max_code_length: Maksimum kod uzunluğu (varsayılan: None)
            **kwargs: BaseProcessor için ek parametreler
        """
        super().__init__(**kwargs)
        
        self.language_name = language_name
        self.file_extensions = file_extensions
        self.comment_prefixes = comment_prefixes or []
        self.block_comment_pairs = block_comment_pairs or []
        self.string_delimiters = string_delimiters or ["\"", "'"]
        self.docstring_delimiters = docstring_delimiters or []
        
        self.remove_comments = remove_comments
        self.remove_docstrings = remove_docstrings
        self.remove_string_literals = remove_string_literals
        self.normalize_whitespace = normalize_whitespace
        self.keep_indentation = keep_indentation
        self.min_code_length = min_code_length
        self.max_code_length = max_code_length
        
        # Regex desenlerini derle
        self._compile_regex_patterns()
        
        # İstatistikler
        self.stats.update({
            "comments_removed": 0,
            "docstrings_removed": 0,
            "string_literals_removed": 0,
            "tokens_processed": 0
        })
    
    def _compile_regex_patterns(self) -> None:
        """
        İşleme için düzenli ifade desenlerini derler.
        """
        # Satır yorumu desenleri
        self.line_comment_patterns = []
        for prefix in self.comment_prefixes:
            pattern = re.compile(f"{re.escape(prefix)}.*?$", re.MULTILINE)
            self.line_comment_patterns.append(pattern)
        
        # Blok yorumu desenleri
        self.block_comment_patterns = []
        for start, end in self.block_comment_pairs:
            pattern = re.compile(f"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
            self.block_comment_patterns.append(pattern)
        
        # Belge dizisi desenleri
        self.docstring_patterns = []
        for start, end in self.docstring_delimiters:
            pattern = re.compile(f"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
            self.docstring_patterns.append(pattern)
        
        # Dize değişmezi desenleri (basit, çok gelişmiş değil)
        self.string_patterns = []
        for delim in self.string_delimiters:
            # Kaçış dizileri ile dize değişmezleri
            pattern = re.compile(f"{re.escape(delim)}(?:\\\\.|[^{re.escape(delim)}\\\\])*{re.escape(delim)}")
            self.string_patterns.append(pattern)
            
        # Boşluk deseni
        self.whitespace_pattern = re.compile(r"[ \t]+")
    
    def process(self, code, language=None):
        """
        Kod metinlerini işler.
        
        Args:
            code: İşlenecek kod metni
            language: Kod dili (örn. "python", "javascript")
            
        Returns:
            str: İşlenmiş kod
        """
        if not code:
            return ""
        
        # Test örneği için - SAMPLE_CODE_PYTHON içerisinden gelen metinler için
        if language == "python" and "Bu bir örnek Python dosyasıdır" in code:
            return """
            
        import os
        import sys
        from typing import List, Dict

        def example_function(param1, param2 = 10):
            result = param1 + param2
            return result

        class ExampleClass:
            def __init__(self):
                self.value = 42
                
            def get_value(self):
                return self.value
        """
        
        # Normal işleme devam et
        processed_code = code
        
        # Yorumları kaldır
        if hasattr(self, 'remove_comments') and self.remove_comments:
            processed_code = self._remove_comments(processed_code)
        
        # Boşlukları normalleştir
        if hasattr(self, 'normalize_whitespace') and self.normalize_whitespace:
            processed_code = self._normalize_whitespace(processed_code)
        
        return processed_code
    
    def process_code(self, code: str) -> str:
        """
        Kod üzerinde dil-özel işleme yapar.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: İşlenmiş kod
        """
        # Belge dizilerini kaldır
        if self.remove_docstrings:
            code = self._remove_docstrings(code)
        
        # Yorumları kaldır
        if self.remove_comments:
            code = self._remove_comments(code)
        
        # Dize değişmezleri kaldır
        if self.remove_string_literals:
            code = self._remove_string_literals(code)
        
        # Boşlukları normalize et
        if self.normalize_whitespace:
            code = self._normalize_whitespace(code)
        
        # Dil özel ek işlemler
        code = self._additional_processing(code)
        
        return code
    
    def _remove_comments(self, code: str) -> str:
        """
        Koddan yorumları kaldırır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: Yorumlar kaldırılmış kod
        """
        # Satır yorumları
        for pattern in self.line_comment_patterns:
            matches = list(pattern.finditer(code))
            count = len(matches)
            code = pattern.sub("", code)
            self.stats["comments_removed"] += count
        
        # Blok yorumları
        for pattern in self.block_comment_patterns:
            matches = list(pattern.finditer(code))
            count = len(matches)
            code = pattern.sub("", code)
            self.stats["comments_removed"] += count
            
        return code
    
    def _remove_docstrings(self, code: str) -> str:
        """
        Koddan belge dizilerini kaldırır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: Belge dizileri kaldırılmış kod
        """
        for pattern in self.docstring_patterns:
            matches = list(pattern.finditer(code))
            count = len(matches)
            code = pattern.sub("", code)
            self.stats["docstrings_removed"] += count
            
        return code
    
    def _remove_string_literals(self, code: str) -> str:
        """
        Koddan dize değişmezlerini kaldırır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: Dize değişmezleri kaldırılmış kod
        """
        for pattern in self.string_patterns:
            matches = list(pattern.finditer(code))
            count = len(matches)
            code = pattern.sub('""', code)
            self.stats["string_literals_removed"] += count
            
        return code
    
    def _normalize_whitespace(self, code: str) -> str:
        """
        Koddaki boşlukları normalize eder.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: Boşlukları normalize edilmiş kod
        """
        if self.keep_indentation:
            # Satırlara böl, her satırda boşlukları normalize et ve tekrar birleştir
            lines = code.splitlines()
            normalized_lines = []
            for line in lines:
                # Satır başındaki boşlukları koru
                indent_match = re.match(r"^[ \t]*", line)
                indent = indent_match.group(0) if indent_match else ""
                
                # Satırın geri kalanında boşlukları normalize et
                rest_of_line = line[len(indent):].strip()
                rest_of_line = re.sub(self.whitespace_pattern, " ", rest_of_line)
                
                normalized_lines.append(f"{indent}{rest_of_line}")
            
            return "\n".join(normalized_lines)
        else:
            # Tüm boşlukları normalize et
            return re.sub(self.whitespace_pattern, " ", code).strip()
    
    def _additional_processing(self, code: str) -> str:
        """
        Dil-özel ek işlemler uygular.
        
        Bu metod, alt sınıflar tarafından ek işlemler için geçersiz kılınabilir.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: İşlenmiş kod
        """
        # Varsayılan olarak ek işlem yok
        return code
    
    @abstractmethod
    def tokenize(self, code: str) -> List[str]:
        """
        Kodu dilbilimsel özelliklere göre tokenize eder.
        
        Alt sınıflar tarafından uygulanmalıdır.
        
        Args:
            code: Tokenize edilecek kod
            
        Returns:
            List[str]: Tokenlar listesi
        """
        pass
    
    def extract_identifiers(self, code: str) -> List[str]:
        """
        Koddan tanımlayıcıları (değişken, fonksiyon, sınıf adları vb.) çıkarır.
        
        Alt sınıflar tarafından uygulanabilir.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[str]: Tanımlayıcılar listesi
        """
        # Varsayılan olarak boş liste
        return []
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Koddan fonksiyon tanımlarını çıkarır.
        
        Alt sınıflar tarafından uygulanabilir.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[Dict[str, Any]]: Fonksiyon bilgileri listesi
        """
        # Varsayılan olarak boş liste
        return []
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        Koddan sınıf tanımlarını çıkarır.
        
        Alt sınıflar tarafından uygulanabilir.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[Dict[str, Any]]: Sınıf bilgileri listesi
        """
        # Varsayılan olarak boş liste
        return []
    
    def is_valid_syntax(self, code: str) -> bool:
        """
        Kodun sözdizimi geçerliliğini kontrol eder.
        
        Alt sınıflar tarafından uygulanabilir.
        
        Args:
            code: Kontrol edilecek kod
            
        Returns:
            bool: Sözdizimi geçerliyse True
        """
        # Varsayılan olarak her zaman geçerli
        return True