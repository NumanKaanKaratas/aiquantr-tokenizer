"""
Python kodu için özel işlemci.

Bu modül, Python programlama dili için kod işleme, temizleme
ve tokenization işlevleri içerir.
"""

import re
import ast
import logging
import tokenize
from io import StringIO
from typing import Dict, Any, Optional, List, Tuple, Pattern, Match, Set

from aiquantr_tokenizer.processors.code.base import BaseCodeProcessor

# Logger oluştur
logger = logging.getLogger(__name__)


class PythonProcessor(BaseCodeProcessor):
    """
    Python kodu işleme sınıfı.
    
    Bu sınıf, Python dilindeki kodları işlemek, 
    temizlemek ve tokenization için özel yöntemler sağlar.
    """
    
    def __init__(
        self,
        remove_type_hints: bool = False,
        remove_decorators: bool = False,
        normalize_imports: bool = False,
        **kwargs
    ):
        """
        PythonProcessor başlatıcısı.
        
        Args:
            remove_type_hints: Tip işaretlerini kaldır (varsayılan: False)
            remove_decorators: Dekoratörleri kaldır (varsayılan: False)
            normalize_imports: İmportları normalleştir (varsayılan: False)
            **kwargs: BaseCodeProcessor için ek parametreler
        """
        # Python dil özelliklerine göre temel işlemciyi başlat
        super().__init__(
            language_name="Python",
            file_extensions=[".py", ".pyw"],
            comment_prefixes=["#"],
            block_comment_pairs=[],  # Python'da blok yorum yok
            string_delimiters=["\"", "'"],
            docstring_delimiters=[('"""', '"""'), ("'''", "'''")],
            **kwargs
        )
        
        self.remove_type_hints = remove_type_hints
        self.remove_decorators = remove_decorators
        self.normalize_imports = normalize_imports
        
        # Ek regex desenleri
        self.decorator_pattern = re.compile(r"^\s*@.*?$", re.MULTILINE)
        self.type_hint_pattern = re.compile(r":\s*[A-Za-z0-9_\[\]\'\"\.]+")
        self.return_type_pattern = re.compile(r"\s*->\s*[A-Za-z0-9_\[\]\'\"\.]+")
        
        # İstatistikler
        self.stats.update({
            "type_hints_removed": 0,
            "decorators_removed": 0,
            "imports_normalized": 0,
            "ast_parsing_errors": 0
        })

    def __call__(self, code):
        """
        Bu metot, sınıfın doğrudan çağrılabilir olmasını sağlar.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: İşlenmiş kod
        """
        return self.process(code)
    
    def process(self, code: str, language: str = None) -> str:
        """
        Python kodunu işler.
        
        Args:
            code: İşlenecek kod
            language: Dil (yok sayılır, zaten Python olarak kabul edilir)
            
        Returns:
            str: İşlenmiş kod
        """
        if not code:
            return ""
        
        # Temel işleme
        processed = super().process(code)
        
        # Python'a özel ek işlemler
        processed = self._additional_processing(processed)
        
        return processed
    
    def _additional_processing(self, code: str) -> str:
        """
        Python koduna özel ek işlemler.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: İşlenmiş kod
        """
        # Dekoratörleri kaldır
        if self.remove_decorators:
            matches = list(self.decorator_pattern.finditer(code))
            count = len(matches)
            code = self.decorator_pattern.sub("", code)
            self.stats["decorators_removed"] += count
        
        # Tip işaretlerini kaldır
        if self.remove_type_hints:
            # Geliştirilmiş tip işareti kaldırma
            # Satır satır işle, özellikle fonksiyon tanımlarını düzgün ele al
            lines = code.split('\n')
            processed_lines = []
            
            for line in lines:
                # Fonksiyon tanımı satırı mı kontrol et
                if re.match(r'\s*def\s+\w+\s*\(', line):
                    # Fonksiyon adını, parametrelerini ve dönüş tipini ayır
                    match = re.match(r'(\s*def\s+\w+\s*\()(.*)(\)\s*(?:->\s*[^:]+\s*)?:.*)', line)
                    if match:
                        prefix, params, suffix = match.groups()
                        
                        # Dönüş tipini kaldır
                        suffix = re.sub(r'\s*->\s*[^:]+\s*:', ':', suffix)
                        
                        # Parametreleri parsle
                        if params.strip():
                            # Parametreleri virgülle ayır (parantez seviyesini takip ederek)
                            param_parts = []
                            param_start = 0
                            paren_level = 0
                            
                            for i, char in enumerate(params):
                                if char in '([{':
                                    paren_level += 1
                                elif char in ')]}':
                                    paren_level -= 1
                                elif char == ',' and paren_level == 0:
                                    param_parts.append(params[param_start:i].strip())
                                    param_start = i + 1
                            
                            # Son parametreyi ekle
                            if param_start < len(params):
                                param_parts.append(params[param_start:].strip())
                            
                            # Her parametreden tip işaretlerini kaldır
                            clean_params = []
                            for param in param_parts:
                                # Parametre adını ve olası varsayılan değeri ayır
                                if ':' in param:
                                    # "param: type" veya "param: type = value" formatı
                                    parts = param.split(':', 1)
                                    param_name = parts[0].strip()
                                    
                                    # Varsayılan değer kontrolü
                                    if '=' in parts[1]:
                                        # "param: type = value" formatı
                                        default_value = re.search(r'=\s*(.*)', parts[1]).group(1).strip()
                                        clean_params.append(f"{param_name} = {default_value}")
                                    else:
                                        # "param: type" formatı, sadece param_name gerekli
                                        clean_params.append(param_name)
                                elif '=' in param:
                                    # "param = value" formatı, tip işareti yok
                                    clean_params.append(param)
                                else:
                                    # Sadece parametre adı
                                    clean_params.append(param)
                            
                            # Temiz parametreleri birleştir
                            clean_params_str = ", ".join(clean_params)
                            
                            # Yeni fonksiyon tanımını oluştur
                            line = f"{prefix}{clean_params_str}{suffix}"
                        else:
                            # Parametresiz fonksiyon
                            line = f"{prefix}{suffix}"
                
                # Normal değişken tanımlarından tip işaretlerini kaldır
                else:
                    # Değişken tanımlarındaki tip işaretlerini kaldır
                    line = re.sub(r':\s*[A-Za-z0-9_\[\]\'\"\.]+\s*=', '=', line)
                
                processed_lines.append(line)
            
            code = '\n'.join(processed_lines)
        
        # İmportları normalleştir
        if self.normalize_imports:
            code = self._normalize_imports(code)
        
        return code
    
    def _normalize_imports(self, code: str) -> str:
        """
        Python import ifadelerini normalleştir.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            str: İmportları normalleştirilmiş kod
        """
        try:
            # AST ağacını parse et
            parsed = ast.parse(code)
            
            # İmport ifadelerini bul ve düzenle
            imports = []
            for node in ast.walk(parsed):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # İmport satır numarası ve içeriği
                    import_info = {
                        "lineno": node.lineno,
                        "text": ast.unparse(node).strip() if hasattr(ast, 'unparse') else ""
                    }
                    imports.append(import_info)
            
            # İmport satırları varsa
            if imports and imports[0]["text"]:
                self.stats["imports_normalized"] += len(imports)
            
            return code
            
        except SyntaxError:
            # AST ayrıştırma hatası (kod hatalı olabilir)
            self.stats["ast_parsing_errors"] += 1
            return code
    
    def tokenize(self, code: str) -> List[str]:
        """
        Python kodunu tokenize eder.
        
        Args:
            code: Tokenize edilecek kod
            
        Returns:
            List[str]: Tokenlar listesi
        """
        tokens = []
        
        try:
            # StringIO nesnesine çevir
            code_io = StringIO(code)
            
            # Python tokenizer kullan
            for tok in tokenize.generate_tokens(code_io.readline):
                token_type = tokenize.tok_name[tok.type]
                token_string = tok.string
                
                # Boş string ve yeni satırları atla
                if token_string and token_string.strip():
                    tokens.append(token_string)
                    self.stats["tokens_processed"] += 1
                    
        except tokenize.TokenError:
            # Tokenizasyon hatası (kod hatalı olabilir)
            logger.warning("Python tokenization hatası")
            
            # Basit bir kelime bölme yap
            tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
            self.stats["tokens_processed"] += len(tokens)
            
        return tokens
    
    def extract_identifiers(self, code: str) -> List[str]:
        """
        Python kodundan tanımlayıcıları çıkarır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[str]: Tanımlayıcılar listesi
        """
        identifiers = set()
        
        try:
            # AST ağacını parse et
            parsed = ast.parse(code)
            
            # Fonksiyon ve sınıf tanımlarını bul
            for node in ast.walk(parsed):
                # Fonksiyon tanımları
                if isinstance(node, ast.FunctionDef):
                    identifiers.add(node.name)
                    
                    # Parametre adları
                    for arg in node.args.args:
                        identifiers.add(arg.arg)
                
                # Değişken tanımları
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            identifiers.add(target.id)
                
                # Sınıf tanımları
                elif isinstance(node, ast.ClassDef):
                    identifiers.add(node.name)
                
                # İmport tanımları
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        identifiers.add(name.name)
                        if name.asname:
                            identifiers.add(name.asname)
                
                # From-import tanımları
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        identifiers.add(name.name)
                        if name.asname:
                            identifiers.add(name.asname)
            
        except SyntaxError:
            # AST ayrıştırma hatası (kod hatalı olabilir)
            # Basit bir regex ile tanımlayıcıları bul
            identifiers.update(re.findall(r'\bdef\s+(\w+)', code))  # Fonksiyonlar
            identifiers.update(re.findall(r'\bclass\s+(\w+)', code))  # Sınıflar
            identifiers.update(re.findall(r'\b(\w+)\s*=', code))  # Değişkenler
        
        return list(identifiers)
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Python kodundan fonksiyon tanımlarını çıkarır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[Dict[str, Any]]: Fonksiyon bilgileri listesi
        """
        functions = []
        
        try:
            # AST ağacını parse et
            parsed = ast.parse(code)
            
            # Fonksiyon tanımlarını bul
            for node in ast.walk(parsed):
                if isinstance(node, ast.FunctionDef):
                    # Parametreleri çıkar
                    params = []
                    for arg in node.args.args:
                        params.append(arg.arg)
                    
                    # Dekoratörleri çıkar
                    decorators = []
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            decorators.append(decorator.id)
                        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                            decorators.append(decorator.func.id)
                    
                    # Docstring çıkar
                    docstring = ast.get_docstring(node)
                    
                    functions.append({
                        "name": node.name,
                        "params": params,
                        "decorators": decorators,
                        "lineno": node.lineno,
                        "docstring": docstring if docstring else "",
                        "is_method": any(isinstance(parent, ast.ClassDef) for parent in ast.iter_child_nodes(parsed) 
                                        if hasattr(parent, 'body') and node in parent.body)
                    })
            
        except SyntaxError:
            # AST ayrıştırma hatası (kod hatalı olabilir)
            pass
        
        return functions
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        Python kodundan sınıf tanımlarını çıkarır.
        
        Args:
            code: İşlenecek kod
            
        Returns:
            List[Dict[str, Any]]: Sınıf bilgileri listesi
        """
        classes = []
        
        try:
            # AST ağacını parse et
            parsed = ast.parse(code)
            
            # Sınıf tanımlarını bul
            for node in ast.walk(parsed):
                if isinstance(node, ast.ClassDef):
                    # Üst sınıfları çıkar
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                    
                    # Metotları çıkar
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(child.name)
                    
                    # Docstring çıkar
                    docstring = ast.get_docstring(node)
                    
                    classes.append({
                        "name": node.name,
                        "bases": bases,
                        "methods": methods,
                        "lineno": node.lineno,
                        "docstring": docstring if docstring else ""
                    })
            
        except SyntaxError:
            # AST ayrıştırma hatası (kod hatalı olabilir)
            pass
        
        return classes
    
    def is_valid_syntax(self, code: str) -> bool:
        """
        Python kodunun sözdizimi geçerliliğini kontrol eder.
        
        Args:
            code: Kontrol edilecek kod
            
        Returns:
            bool: Sözdizimi geçerliyse True
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False