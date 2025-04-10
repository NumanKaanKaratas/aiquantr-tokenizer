"""
PHP kodu için özel işlemci.

Bu modül, PHP programlama dili için kod işleme, temizleme
ve tokenization işlevleri içerir.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple, Pattern, Match, Set

from aiquantr_tokenizer.processors.code.base import BaseCodeProcessor

# Logger oluştur
logger = logging.getLogger(__name__)

class PhpProcessor(BaseCodeProcessor):
    """
    PHP kodu işleme sınıfı.
    
    Bu sınıf, PHP dilindeki kodları işlemek, 
    temizlemek ve tokenization için özel yöntemler sağlar.
    """
    
    def __init__(
        self,
        remove_php_tags: bool = False,
        remove_html: bool = False,
        normalize_variables: bool = False,
        **kwargs
    ):
        """
        PhpProcessor başlatıcısı.
        
        Args:
            remove_php_tags: PHP etiketlerini kaldır (varsayılan: False)
            remove_html: HTML kodunu kaldır (varsayılan: False)
            normalize_variables: Değişken adlarını normalleştir (varsayılan: False)
            **kwargs: BaseCodeProcessor için ek parametreler
        """
        # PHP dil özelliklerine göre temel işlemciyi başlat
        super().__init__(
            language_name="PHP",
            file_extensions=[".php", ".phtml", ".php3", ".php4", ".php5", ".php7"],
            comment_prefixes=["//", "#"],
            block_comment_pairs=[("/*", "*/")],
            string_delimiters=["\"", "'", "`"],
            docstring_delimiters=[],  # PHP'nin resmi docstring formatı yok
            **kwargs
        )
        
        self.remove_php_tags = remove_php_tags
        self.remove_html = remove_html
        self.normalize_variables = normalize_variables
        
        # Satır takibi için değişken
        self.current_line = 1
        
        # Ek regex desenleri
        self.php_tag_pattern = re.compile(r'<\?php|\?>')
        self.html_pattern = re.compile(r'<[^?].*?>|</.*?>')
        self.variable_pattern = re.compile(r'\$\w+')
        self.function_pattern = re.compile(r'\bfunction\s+(\w+)\s*\(', re.DOTALL)
        self.class_pattern = re.compile(r'\bclass\s+(\w+)\s*', re.DOTALL)
        self.heredoc_pattern = re.compile(r'<<<[\'"]?(\w+)[\'"]?\n.*?\n\1;', re.DOTALL)
        self.nowdoc_pattern = re.compile(r"<<<'(\w+)'\n.*?\n\1;", re.DOTALL)
        
        # İstatistikler
        self.stats.update({
            "php_tags_removed": 0,
            "html_tags_removed": 0,
            "heredoc_removed": 0,
            "variables_normalized": 0
        })
    
    def _update_line_tracking(self, line_count: int) -> None:
        """
        Satır sayımı takibini günceller - iç kullanım için.
        
        Args:
            line_count: Artan satır sayısı
        """
        self.current_line += line_count
        
    def process(self, text: str) -> str:
        """
        PHP kodunu işler.
        
        Args:
            text: İşlenecek PHP kodu
                
        Returns:
            str: İşlenmiş PHP kodu
        """
        if not text:
            return ""
        
        # Temel işlemleri yap
        processed = super().process(text)
        
        # PHP etiketlerini kaldır
        if hasattr(self, 'remove_php_tags') and self.remove_php_tags:
            matches = list(self.php_tag_pattern.finditer(processed))
            processed = self.php_tag_pattern.sub('', processed)
            self.stats["php_tags_removed"] += len(matches)

        # HTML etiketlerini kaldır
        if hasattr(self, 'remove_html') and self.remove_html:
            matches = list(self.html_pattern.finditer(processed))
            processed = self.html_pattern.sub('', processed)
            self.stats["html_tags_removed"] += len(matches)
        
        return processed
    
    def _additional_processing(self, code: str) -> str:
        """
        PHP koduna özel ek işlemler.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: İşlenmiş kod
        """
        # Heredoc ve Nowdoc karakter dizilerini temizle (string literals'dan önce işlenmeli)
        if self.remove_string_literals:
            # Heredocs
            heredoc_matches = list(self.heredoc_pattern.finditer(code))
            code = self.heredoc_pattern.sub('""', code)
            self.stats["heredoc_removed"] += len(heredoc_matches)
            
            # Nowdocs
            nowdoc_matches = list(self.nowdoc_pattern.finditer(code))
            code = self.nowdoc_pattern.sub('""', code)
            self.stats["heredoc_removed"] += len(nowdoc_matches)
        
        # Değişken adlarını normalleştir
        if self.normalize_variables:
            code = self._normalize_variables(code)
        
        return code
    
    def _normalize_variables(self, code: str) -> str:
        """
        PHP değişken adlarını normalleştir.
        
        Bu fonksiyon, değişken adlarını korur ancak
        paylaşılabilirliği artırmak için yapılabilir.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: Değişkenleri normalleştirilmiş kod
        """
        # Şu an için basit bir şekilde değişken adlarını değiştirmiyoruz
        # Bu fonksiyon, gelecekte değişken adlarını normalleştirmek için genişletilebilir
        return code
    
    def tokenize(self, code: str) -> List[str]:
        """
        PHP kodunu tokenize eder. PHP sözdizimini dikkate alarak kodun anlamlı
        parçalara ayrılmasını sağlar.
        
        Args:
            code: Tokenize edilecek PHP kodu
                
        Returns:
            List[str]: Tokenlar listesi
        """
        # Satır sayımını sıfırla
        self.current_line = 1
        
        tokens = []
        
        # Özellikle işlenmesi gereken PHP-spesifik tokenlar
        php_specific_tokens = {
            '<?php': '<?php',  # PHP açılış etiketi
            '<?': '<?',        # PHP kısa açılış etiketi
            '?>': '?>'         # PHP kapanış etiketi
        }
        
        # Operatörler - uzunluğa göre sırala (uzun olanlar önce)
        operators = [
            # Karşılaştırma operatörleri
            '===', '!==', '==', '!=', '<=>', '>=', '<=', '>', '<',
            # Atama operatörleri
            '+=', '-=', '*=', '/=', '.=', '%=', '&=', '|=', '^=', '<<=', '>>=', '??=', 
            # Aritmetik operatörler
            '+', '-', '*', '/', '%', '**', '++', '--',
            # String operatörleri 
            '.',
            # Mantıksal operatörler
            '&&', '||', '!', 'and', 'or', 'xor',
            # Bit operatörleri
            '&', '|', '^', '~', '<<', '>>',
            # Null birleştirme operatörü
            '??',
            # Atama operatörü
            '='
        ]
        
        # PHP anahtar kelimeleri
        php_keywords = {
            'abstract', 'and', 'array', 'as', 'break', 'callable', 'case', 'catch', 'class', 'clone',
            'const', 'continue', 'declare', 'default', 'die', 'do', 'echo', 'else', 'elseif', 'empty',
            'enddeclare', 'endfor', 'endforeach', 'endif', 'endswitch', 'endwhile', 'eval', 'exit',
            'extends', 'final', 'finally', 'fn', 'for', 'foreach', 'function', 'global', 'goto',
            'if', 'implements', 'include', 'include_once', 'instanceof', 'insteadof', 'interface',
            'isset', 'list', 'namespace', 'new', 'or', 'print', 'private', 'protected', 'public',
            'require', 'require_once', 'return', 'static', 'switch', 'throw', 'trait', 'try',
            'unset', 'use', 'var', 'while', 'xor', 'yield', 'yield from'
        }
        
        # Regex tipleri (pre-compile) - performans için
        patterns: List[Tuple[str, Pattern[str]]] = [
            ('PHP_TAG', re.compile(r'(<\?php\b|<\?|\?>)')),
            ('WHITESPACE', re.compile(r'\s+')),
            ('COMMENT', re.compile(r'(/\*.*?\*/|//.*?$|#.*?$)', re.DOTALL | re.MULTILINE)),
            ('VARIABLE', re.compile(r'\$[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*')),
            ('STRING', re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')),
            ('NUMBER', re.compile(r'\b(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b')),
            ('KEYWORD', re.compile(r'\b(' + '|'.join(php_keywords) + r')\b')),
            ('OPERATOR', re.compile('|'.join(map(re.escape, sorted(operators, key=len, reverse=True))))),
            ('IDENTIFIER', re.compile(r'\b[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*\b')),
            ('SYMBOL', re.compile(r'[\[\]\(\)\{\};:,\.\?\$@]')),
            ('UNKNOWN', re.compile(r'.'))  # En son eşleşme - herhangi bir karakter
        ]
        
        # İşlenecek kodda kalan metin
        remaining_code = code
        
        while remaining_code:
            matched = False
            
            # PHP etiketleri için özel işleme
            for token_name, pattern in patterns:
                match = pattern.match(remaining_code)
                if match:
                    token_text = match.group(0)
                    
                    # PHP etiketlerini özel olarak ele al
                    if token_name == 'PHP_TAG':
                        if token_text in php_specific_tokens:
                            # PHP etiketlerini tek token olarak ekle
                            tokens.append(token_text)
                    elif token_name == 'WHITESPACE':
                        # Whitespace'i atla, ama yeni satırları kaydet
                        if '\n' in token_text:
                            self._update_line_tracking(token_text.count('\n'))
                    elif token_name == 'COMMENT':
                        # Yorumları atla veya istenirse ekle
                        if not self.remove_comments:
                            tokens.append(token_text)
                        # Yorum içindeki satır sonlarını izle
                        self._update_line_tracking(token_text.count('\n'))
                    else:
                        # Diğer tüm token türleri
                        tokens.append(token_text)
                        self.stats["tokens_processed"] += 1
                    
                    # Eşleşen metni koddan kaldır
                    remaining_code = remaining_code[len(token_text):]
                    matched = True
                    break
            
            # Eşleşme bulunamadıysa (ki bu olmamalı, çünkü UNKNOWN her şeyi eşleştirir)
            if not matched:
                # Güvenlik amacıyla - sonsuz döngüden kaçınmak için
                remaining_code = remaining_code[1:]
        
        return tokens
    
    def extract_identifiers(self, code: str) -> List[str]:
        """
        PHP kodundan tanımlayıcıları çıkarır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[str]: Tanımlayıcılar listesi
        """
        identifiers = set()
        
        # Değişkenleri çıkar
        variables = re.findall(r'\$(\w+)', code)
        identifiers.update(variables)
        
        # Fonksiyonları çıkar
        functions = re.findall(r'\bfunction\s+(\w+)\s*\(', code)
        identifiers.update(functions)
        
        # Sınıfları çıkar
        classes = re.findall(r'\bclass\s+(\w+)', code)
        identifiers.update(classes)
        
        # Metotları çıkar
        methods = re.findall(r'->(\w+)\s*\(', code)
        identifiers.update(methods)
        
        # Namespace ve Use ifadelerindeki tanımlayıcılar
        namespaces = re.findall(r'\bnamespace\s+([A-Za-z0-9_\\]+)', code)
        uses = re.findall(r'\buse\s+([A-Za-z0-9_\\]+)(?:\s+as\s+(\w+))?', code)
        
        for namespace in namespaces:
            parts = namespace.split('\\')
            identifiers.update(parts)
        
        for use_match in uses:
            use = use_match[0]
            alias = use_match[1] if len(use_match) > 1 and use_match[1] else None
            
            parts = use.split('\\')
            identifiers.update(parts)
            
            if alias:
                identifiers.add(alias)
        
        return list(identifiers)
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        PHP kodundan fonksiyon tanımlarını çıkarır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[Dict[str, Any]]: Fonksiyon bilgileri listesi
        """
        functions = []
        
        # Fonksiyonları regex ile bul
        function_pattern = r'\bfunction\s+(\w+)\s*\(([^)]*)\)'
        function_matches = re.finditer(function_pattern, code)
        
        for match in function_matches:
            name = match.group(1)
            params_str = match.group(2).strip()
            
            # Parametreleri ayır
            params = []
            if params_str:
                param_parts = params_str.split(',')
                for part in param_parts:
                    part = part.strip()
                    # Tip işareti ve varsayılan değeri kaldır
                    param_name = re.sub(r'^.*?\$(\w+)(?:\s*=.*)?$', r'\1', part)
                    if param_name.startswith('$'):
                        param_name = param_name[1:]  # $ işaretini kaldır
                    params.append(param_name)
            
            # Fonksiyon tipini belirle (metot mu?)
            start_pos = match.start()
            preceding_code = code[max(0, start_pos - 100):start_pos]
            is_method = bool(re.search(r'\bclass\b|\binterface\b|\btrait\b', preceding_code))
            
            functions.append({
                "name": name,
                "params": params,
                "is_method": is_method,
                "lineno": code[:start_pos].count('\n') + 1
            })
        
        return functions
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        PHP kodundan sınıf tanımlarını çıkarır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[Dict[str, Any]]: Sınıf bilgileri listesi
        """
        classes = []
        
        # Sınıfları regex ile bul
        class_pattern = r'\bclass\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?'
        class_matches = re.finditer(class_pattern, code)
        
        for match in class_matches:
            name = match.group(1)
            parent = match.group(2) if match.group(2) else ""
            implements_str = match.group(3) if match.group(3) else ""
            
            # Interfaces
            implements = []
            if implements_str:
                for interface in implements_str.split(','):
                    implements.append(interface.strip())
            
            # Sınıf içindeki metotları bul
            # Sınıfın nerede başladığını ve bittiğini bulmak karmaşık
            # Basitleştirilmiş bir yaklaşım kullanıyoruz
            start_pos = match.end()
            
            # Sınıf bloğunu bul
            class_body = ""
            brace_count = 0
            found_open_brace = False
            
            for i in range(start_pos, len(code)):
                char = code[i]
                if char == '{':
                    found_open_brace = True
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if found_open_brace and brace_count == 0:
                        class_body = code[start_pos:i+1]
                        break
            
            # Sınıf içindeki metotları bul
            methods = []
            if class_body:
                method_matches = re.finditer(r'\bfunction\s+(\w+)\s*\(', class_body)
                for method_match in method_matches:
                    methods.append(method_match.group(1))
            
            classes.append({
                "name": name,
                "parent": parent,
                "implements": implements,
                "methods": methods,
                "lineno": code[:match.start()].count('\n') + 1
            })
        
        return classes
    
    def is_valid_syntax(self, code: str) -> bool:
        """
        PHP kodunun sözdizimi geçerliliğini kontrol eder.
        
        Bu çok basit bir kontroldür, tam sözdizimi doğrulaması yapmaz.
        
        Args:
            code: Kontrol edilecek kod
            
        Returns:
            bool: Sözdizimi genel olarak uygunsa True
        """
        # Parantezlerin, süslü parantezlerin ve köşeli parantezlerin dengeli olup olmadığını kontrol et
        stack = []
        pairs = {'(': ')', '{': '}', '[': ']'}
        
        # Bir string içinde miyiz? (string içindeki parantezleri yoksay)
        in_string = False
        string_char = None
        escape_next = False
        
        for char in code:
            # Kaçış karakteri kontrolü
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
            
            # String başlangıç/bitiş kontrolü
            if char in ['"', "'"] and (not in_string or string_char == char):
                if in_string:
                    in_string = False
                    string_char = None
                else:
                    in_string = True
                    string_char = char
                continue
            
            # String içinde değilsek parantez kontrolü yap
            if not in_string:
                if char in pairs.keys():
                    stack.append(char)
                elif char in pairs.values():
                    if not stack:
                        return False
                    
                    last_open = stack.pop()
                    if pairs[last_open] != char:
                        return False
        
        # Tüm açık parantezler kapandı mı?
        return len(stack) == 0