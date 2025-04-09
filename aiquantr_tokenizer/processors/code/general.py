"""
Genel kod işleme sınıfları.

Bu modül, çeşitli programlama dilleri için kod dosyalarını işleyen
ve tokenizer eğitimi için hazırlayan sınıfları içerir.
"""

import re
import logging
from typing import Dict, Any, Optional

from aiquantr_tokenizer.processors.base_processor import BaseProcessor

# Logger oluştur
logger = logging.getLogger(__name__)


class CodeProcessor(BaseProcessor):
    """
    Kod dosyaları için genel işleme sınıfı.
    
    Bu sınıf, farklı programlama dili dosyalarını işleyerek
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