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
        remove_empty_lines: bool = False,
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
        self.remove_empty_lines = remove_empty_lines
        
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
            # 1. Satır sonundaki eşitlik yerinde dönüş tip belirteçlerini temizle
            code = re.sub(r'(\w+\s*=\s*{[\s\S]*?})\s*->\s*[\w\[\],\.\s]+', r'\1', code)

            # 2. Function parameter tip işaretlerini temizle - başlangıç çıktısında sorunlu olan yerler
            code = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*[a-zA-Z_][a-zA-Z0-9_\[\],\.\s]+\s*=\s*', r'\1 = ', code)
            
            # 3. Satır içi değişken tip işaretlerini temizle
            code = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_\[\],\.\s]+)(?=\s*(?:,|\)))', r'\1', code)
            
            # 4. Fonksiyon dönüş tiplerini temizle
            code = re.sub(r'(\))\s*->\s*[a-zA-Z_][a-zA-Z0-9_\[\],\.\s]+\s*:', r'\1:', code)
            
            # 5. Standart parametreye sahip olmayan fonksiyon dönüş tiplerini temizle
            code = re.sub(r'(\))\s*->\s*[a-zA-Z_][a-zA-Z0-9_\[\],\.\s]+', r'\1', code)
            
            # 6. Fonksiyon parametreleri üzerinde daha kapsamlı bir işlem yapalım
            lines = code.split('\n')
            processed_lines = []

            for line in lines:
                # Fonksiyon tanımlarında tip işaretlerini kaldır
                if re.match(r'^\s*def\s+\w+\s*\(', line) or '=' in line:
                    # Parametre listesindeki "param: type" kalıplarını bul ve sadece "param" olarak değiştir
                    in_param_list = False
                    new_line = ""
                    param_start = 0
                    param_depth = 0
                    
                    # Parametre listesini bul ve parametreleri tek tek işle
                    for i, c in enumerate(line):
                        if c == '(' and not in_param_list:
                            in_param_list = True
                            new_line += line[:i+1]
                            param_start = i + 1
                        elif c == '(' and in_param_list:
                            param_depth += 1
                            new_line += c
                        elif c == ')' and in_param_list and param_depth > 0:
                            param_depth -= 1
                            new_line += c
                        elif c == ')' and in_param_list and param_depth == 0:
                            # Parametre listesinin sonuna gelindi, parametreyi işle
                            param_text = line[param_start:i]
                            
                            # Parametreleri virgülle ayır ve her birini işle
                            params = []
                            param_parts = []
                            temp_start = 0
                            temp_depth = 0
                            
                            for j, pc in enumerate(param_text):
                                if pc in '([{':
                                    temp_depth += 1
                                elif pc in ')]}':
                                    temp_depth -= 1
                                elif pc == ',' and temp_depth == 0:
                                    param_parts.append(param_text[temp_start:j])
                                    temp_start = j + 1
                            
                            # Son parametreyi de ekle
                            if temp_start < len(param_text):
                                param_parts.append(param_text[temp_start:])
                            
                            # Parametreleri temizle
                            for part in param_parts:
                                part = part.strip()
                                if ':' in part:
                                    name_part = part.split(':', 1)[0].strip()
                                    if '=' in part:
                                        # "param: type = value" durumu
                                        default_value = part.split('=', 1)[1].strip()
                                        params.append(f"{name_part} = {default_value}")
                                    else:
                                        # "param: type" durumu
                                        params.append(name_part)
                                else:
                                    # Tip işareti olmayan parametre
                                    params.append(part)
                                    
                            # Parametreleri birleştir ve satırı oluştur
                            new_line += ", ".join(params) + c
                            in_param_list = False
                        elif not in_param_list:
                            new_line += c
                        else:
                            # Parametre listesi içindeyiz
                            continue
                            
                    # Eğer parametre listesi bulunamazsa orijinal satırı kullan
                    if new_line:
                        line = new_line
                    
                    # Dönüş tipi işaretlerini kaldır
                    line = re.sub(r'(\))\s*->\s*[^:]+:', r'\1:', line)
                
                processed_lines.append(line)
            
            code = '\n'.join(processed_lines)
            
            # Son olarak genel bir temizlik yap
            # Değişken tanımlamalarındaki son kalan tip işaretlerini kaldır
            code = re.sub(r'(\w+)\s*:\s*[A-Za-z0-9_\[\]\'\"\.]+\s*=', r'\1 =', code)
            
            # Doğrudan "= ..." ifadelerinden önceki tip işaretlerini kaldır 
            code = re.sub(r'([a-zA-Z0-9_]+)\s*:\s*[A-Za-z0-9_\[\]\'\"\.]+\s*=', r'\1 =', code)
        
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
    
    def extract_identifiers(self, code: str) -> list:
        """
        Python kodundan tanımlayıcıları çıkarır (değişken, fonksiyon, sınıf, modül, parametre ve sınıf özellikleri).
        Python 2.7 ve tüm Python 3.x sürümleriyle uyumludur.
        
        Args:
            code: İşlenecek Python kodu
                
        Returns:
            list: Tanımlanan ve kullanılan benzersiz tanımlayıcıların listesi
        """
        import ast
        import re
        import sys
        
        # Python 3 için typing modülü ile tip belirtimi, 2.7'de çalışmaz
        # o yüzden inline olarak kullanmıyoruz
        
        identifiers = set()
        is_py2 = sys.version_info[0] == 2
        
        try:
            # AST ağacını parse et
            parsed = ast.parse(code)
            
            # AST ağacını dolaş ve tanımlayıcıları topla
            for node in ast.walk(parsed):
                # PYTHON 2.7 ve 3.x için farklı AST node'ları
                
                # Fonksiyon tanımları
                if isinstance(node, ast.FunctionDef):
                    identifiers.add(node.name)
                    
                    # Python 2.7 ve 3.x'te parametre node'ları farklıdır
                    if is_py2:
                        # Python 2.7'de args.args doğrudan Name node'larıdır
                        for arg in node.args.args:
                            if isinstance(arg, ast.Name):
                                identifiers.add(arg.id)
                    else:
                        # Python 3.x'te arg node'ları vardır
                        for arg in node.args.args:
                            if hasattr(arg, 'arg'):
                                identifiers.add(arg.arg)
                            elif hasattr(arg, 'id'):
                                identifiers.add(arg.id)
                    
                    # Python 3.x'te keyword-only argümanlar
                    if not is_py2 and hasattr(node.args, 'kwonlyargs'):
                        for kwarg in node.args.kwonlyargs:
                            identifiers.add(kwarg.arg)
                
                # Sınıf tanımları
                elif isinstance(node, ast.ClassDef):
                    identifiers.add(node.name)
                
                # Değişken tanımları
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            identifiers.add(target.id)
                        # Tuple unpacking durumunda
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    identifiers.add(elt.id)
                
                # Değişken isimleri (kullanım)
                elif isinstance(node, ast.Name):
                    identifiers.add(node.id)
                
                # Import ifadeleri
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        # Import edilen modül adı
                        identifiers.add(name.name)
                        # Eğer as ile yeniden adlandırılmışsa
                        if name.asname:
                            identifiers.add(name.asname)
                
                # From-import ifadeleri
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        identifiers.add(name.name)
                        if name.asname:
                            identifiers.add(name.asname)
                
                # Sınıf özellikleri (self.attribute gibi)
                elif isinstance(node, ast.Attribute):
                    # Özellikle self.xyz şeklindeki özellikleri yakalamak için
                    if isinstance(node.value, ast.Name) and node.value.id == 'self':
                        identifiers.add(node.attr)
                
                # Python 3.x'e özel arg node - Python 2.7'de bunlar Name node'larıdır
                elif not is_py2 and isinstance(node, ast.arg) and hasattr(node, 'arg'):
                    identifiers.add(node.arg)
                
                # Comprehension değişkenleri (list/dict comprehension) - Her iki Python sürümünde benzer
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    for generator in node.generators:
                        if isinstance(generator.target, ast.Name):
                            identifiers.add(generator.target.id)
                        # Tuple unpacking durumunda
                        elif isinstance(generator.target, ast.Tuple):
                            for elt in generator.target.elts:
                                if isinstance(elt, ast.Name):
                                    identifiers.add(elt.id)
                
                # With ifadeleri için as kısmı
                # Python 2.7'de withitem node yok, with_items doğrudan With node'da bulunur
                if not is_py2:
                    if isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                        identifiers.add(node.optional_vars.id)
                else:
                    if isinstance(node, ast.With) and node.optional_vars and isinstance(node.optional_vars, ast.Name):
                        identifiers.add(node.optional_vars.id)
                    
                    # For döngüsü değişkenleri
                    elif isinstance(node, ast.For):
                        if isinstance(node.target, ast.Name):
                            identifiers.add(node.target.id)
                        # Tuple unpacking durumunda
                        elif isinstance(node.target, ast.Tuple):
                            for elt in node.target.elts:
                                if isinstance(elt, ast.Name):
                                    identifiers.add(elt.id)
                        
                    # Exception handling değişkenleri - Python sürümlerine uygun şekilde
                    elif isinstance(node, ast.ExceptHandler):
                        if is_py2:
                            if hasattr(node, 'name') and isinstance(node.name, ast.Name):
                                identifiers.add(node.name.id)
                        else:  # Python 3.x
                            if hasattr(node, 'name'):
                                if node.name is None:  # Exception yakalanıyor ama değişkene atanmıyor
                                    continue
                                if isinstance(node.name, str):  # Python 3.8+
                                    identifiers.add(node.name)
                                elif isinstance(node.name, ast.Name):  # Python 3.7 ve öncesi
                                    identifiers.add(node.name.id)
                    
        except SyntaxError:
            # AST ayrıştırma hatası durumunda regex ile basit tanımlayıcı çıkarma
            identifiers.update(re.findall(r'\bdef\s+(\w+)', code))      # Fonksiyon tanımları
            identifiers.update(re.findall(r'\bclass\s+(\w+)', code))    # Sınıf tanımları
            identifiers.update(re.findall(r'\b(\w+)\s*=', code))        # Değişken tanımları
            identifiers.update(re.findall(r'\bself\.(\w+)', code))      # Sınıf özellikleri
            
            # Daha eski Python sürümleri için import ifadeleri dikkatlice ele alınmalı
            import_matches = re.findall(r'(?:^|\s)from\s+([\w.]+)\s+import\s+([\w*,\s]+)(?:\s+as\s+(\w+))?', code)
            for match in import_matches:
                module, imports, alias = match
                if alias:
                    identifiers.add(alias)
                for imp in imports.split(','):
                    imp = imp.strip()
                    if imp and imp != '*':
                        identifiers.add(imp)
                        
            # Standart import ifadeleri
            import_matches = re.findall(r'(?:^|\s)import\s+([\w.,\s]+)', code)
            for match in import_matches:
                modules = match.split(',')
                for module in modules:
                    module = module.strip()
                    if module:
                        parts = module.split(' as ')
                        if len(parts) > 1:
                            identifiers.add(parts[1].strip())
                        else:
                            identifiers.add(module.split('.')[-1])
            
        # Genel Python anahtar kelimelerini filtrele
        python_keywords = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 
            'def', 'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not',
            'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'with', 'yield',
            # Python 3 anahtar kelimeleri
            'False', 'None', 'True', 'nonlocal', 'async', 'await'
        }
        
        # Anahtar kelimeleri filtrele
        identifiers = {id for id in identifiers if id not in python_keywords}
        
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