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
        PHP kodunu tokenize eder.
        
        Args:
            code: Tokenize edilecek kod
            
        Returns:
            List[str]: Tokenlar listesi
        """
        # Basit bir tokenization uygula
        # PHP için daha gelişmiş bir tokenizer kullanılabilir (ör. phpparser kütüphanesi)
        
        # PHP'nin kendi token türlerini taklit eden basit tokenization
        tokens = []
        
        # Operatörler ve sınırlayıcılar
        operators = ['+', '-', '*', '/', '%', '=', '==', '===', '!=', '!==', '<', '>', '<=', '>=', 
                    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '.', '.=', 
                    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=']
        
        # Daha büyük operatörleri önce kontrol et (örneğin === önce == yerine)
        operators.sort(key=len, reverse=True)
        
        # Operatör kalıpları
        operator_pattern = '|'.join(map(re.escape, operators))
        
        # Kod üzerinde genel bir tokenizasyon yap
        pattern = re.compile(
            r'(' + operator_pattern + r')|'  # operatörler
            r'(\$\w+)|'                      # değişkenler
            r'(\b\d+(?:\.\d+)?)|'            # sayılar
            r'(\b\w+\b)|'                    # tanımlayıcılar
            r'(\s+)|'                        # boşluklar
            r'([^\w\s])'                     # diğer karakterler
        )
        
        matches = pattern.finditer(code)
        for match in matches:
            if match.group(1) or match.group(2) or match.group(3) or match.group(4) or match.group(6):
                # Boş olmayan token
                token = match.group(0)
                if token.strip():  # sadece boşluk olmayan tokenları ekle
                    tokens.append(token)
                    self.stats["tokens_processed"] += 1
        
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