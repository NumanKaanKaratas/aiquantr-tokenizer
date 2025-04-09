"""
Python kod işleme sınıfı için testler.
"""

from tests.test_processors import BaseProcessorTest

class TestPythonProcessor(BaseProcessorTest):
    """
    Python işlemcisi için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        super().setUp()
        try:
            from aiquantr_tokenizer.processors.code.python import PythonProcessor
            self.PythonProcessor = PythonProcessor
        except ImportError:
            self.skipTest("aiquantr_tokenizer.processors.code.python modülü bulunamadı")
    
    def test_basic_python_processing(self):
        """
        Temel Python kod işleme işlevselliğini test eder.
        """
        processor = self.PythonProcessor()
        
        # Python kodu
        python_code = """#!/usr/bin/env python3
# Bu bir yorum
def example_function():
    ""Bu bir docstring.""
    return True
"""
        
        self.assertProcessingResult(
            processor,
            python_code,
            expected_parts=["def example_function():", "return True"]
        )
        
    def test_remove_python_comments(self):
        """
        Python yorumlarını kaldırma işlevini test eder.
        """
        processor = self.PythonProcessor(remove_comments=True)
        
        python_code = """
# Bu bir yorum satırı
x = 10  # Satır sonu yorumu
y = 20
"""
        
        self.assertProcessingResult(
            processor,
            python_code,
            not_expected_parts=["# Bu bir yorum satırı", "# Satır sonu yorumu"],
            expected_parts=["x = 10", "y = 20"]
        )
        
    def test_remove_python_docstrings(self):
        """
        Python docstring kaldırma işlevini test eder.
        """
        processor = self.PythonProcessor(remove_docstrings=True)
        
        python_code = '''
def example_function():
    """
    Bu bir docstring.
    Çok satırlı.
    """
    x = 10
    return x

class ExampleClass:
    """Bu bir sınıf docstring'i."""
    pass
'''
        
        self.assertProcessingResult(
            processor,
            python_code,
            not_expected_parts=[
                '"""', 
                "Bu bir docstring.", 
                "Çok satırlı.", 
                "Bu bir sınıf docstring'i."
            ],
            expected_parts=[
                "def example_function():", 
                "x = 10", 
                "return x", 
                "class ExampleClass:", 
                "pass"
            ]
        )
        
    def test_remove_type_hints(self):
        """
        Python tip işaretlerini kaldırma işlevini test eder.
        """
        processor = self.PythonProcessor(remove_type_hints=True)
        
        python_code = """
def example_function(param1: str, param2: int = 10) -> list:
    x: int = 5
    return [x] * param2
"""
        
        result = self.assertProcessingResult(
            processor,
            python_code,
            expected_parts=[
                "def example_function(param1, param2 = 10):",
                "x = 5",
                "return [x] * param2"
            ]
        )
        
        # Tip işaretlerinin kaldırıldığını kontrol et
        self.assertNotIn("-> list", result)
        self.assertNotIn(": str", result)
        self.assertNotIn(": int", result)
        
    def test_tokenize_python(self):
        """
        Python tokenize işlevini test eder.
        """
        processor = self.PythonProcessor()
        
        python_code = """
def example():
    x = 10
    return x
"""
        
        tokens = processor.tokenize(python_code)
        
        # Tokenization doğru yapıldı mı kontrol et
        expected_tokens = ["def", "example", "(", ")", ":", "x", "=", "10", "return", "x"]
        for token in expected_tokens:
            self.assertIn(token, tokens)
            
        # Token sayısı doğru mu
        self.assertGreaterEqual(len(tokens), len(expected_tokens))
        
    def test_extract_python_identifiers(self):
        """
        Python tanımlayıcıları çıkarma işlevini test eder.
        """
        processor = self.PythonProcessor()
        
        python_code = """
def example_function(param1):
    x = 10
    return x + param1

class ExampleClass:
    def __init__(self):
        self.value = 42
"""
        
        identifiers = processor.extract_identifiers(python_code)
        
        # Tanımlayıcılar doğru çıkarıldı mı kontrol et
        expected_identifiers = ["example_function", "param1", "x", "ExampleClass", "__init__", "self", "value"]
        for identifier in expected_identifiers:
            self.assertIn(identifier, identifiers)