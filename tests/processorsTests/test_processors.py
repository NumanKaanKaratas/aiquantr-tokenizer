"""
Kod işleme modülleri için temel test sınıfı.

Bu modül, çeşitli kod işleme sınıflarını test etmek için
temel fonksiyonları ve ortak test durumlarını sağlar.
"""

import unittest
import tempfile
import os
from pathlib import Path

class BaseProcessorTest(unittest.TestCase):
    """
    Tüm işlemcilerin testleri için temel sınıf.
    
    Bu sınıf, işlemcilerin testleri için ortak yardımcı 
    metodları ve temel test durumlarını içerir.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Geçici dosya dizini oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        self.temp_dir.cleanup()
    
    def create_test_file(self, content, extension=".txt"):
        """
        Test dosyası oluşturur ve dosyanın yolunu döndürür.
        
        Args:
            content: Dosya içeriği
            extension: Dosya uzantısı
            
        Returns:
            Path: Oluşturulan dosyanın yolu
        """
        # Benzersiz bir dosya adı oluştur
        file_path = self.test_dir_path / f"test_file_{id(content)}{extension}"
        
        # Dosyayı oluştur ve içeriğini yaz
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return file_path
    
    def assertProcessingResult(self, processor, input_text, expected_result=None, 
                               not_expected_parts=None, expected_parts=None, language=None):
        """
        İşlemci sonucunu doğrular.
        
        Args:
            processor: Test edilecek işlemci
            input_text: İşlenecek girdi metni
            expected_result: Tam olarak beklenen çıktı (None ise kontrol edilmez)
            not_expected_parts: Çıktıda olmaması gereken parçalar listesi
            expected_parts: Çıktıda olması gereken parçalar listesi
            language: Dil parametresi (varsa)
        """
        # İşleme yap
        if language is not None:
            result = processor.process(input_text, language=language)
        else:
            result = processor.process(input_text)
            
        # Sonuçları kontrol et
        if expected_result is not None:
            self.assertEqual(result, expected_result)
            
        if not_expected_parts is not None:
            for part in not_expected_parts:
                self.assertNotIn(part, result)
                
        if expected_parts is not None:
            for part in expected_parts:
                self.assertIn(part, result)
                
        return result


class TestGeneralCodeProcessor(BaseProcessorTest):
    """
    Genel CodeProcessor sınıfının temel testleri.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        super().setUp()
        try:
            from aiquantr_tokenizer.processors.code.general import CodeProcessor
            self.CodeProcessor = CodeProcessor
        except ImportError:
            self.skipTest("aiquantr_tokenizer.processors.code.general modülü bulunamadı")
    
    def test_basic_processing(self):
        """
        Temel kod işleme işlevselliğini test eder.
        """
        processor = self.CodeProcessor()
        
        # Boş girdi testi
        self.assertEqual(processor.process(""), "")
        
        # Basit metin işleme
        text = "function test() { var x = 10; }"
        self.assertEqual(processor.process(text), text)
    
    def test_remove_comments(self):
        """
        Yorum kaldırma işlevini test eder.
        """
        processor = self.CodeProcessor(remove_comments=True)
        
        # JavaScript yorumları
        js_code = """
        // Bu bir satır yorumu
        function test() {
            /* Bu bir blok
               yorumu */
            var x = 10; // Satır sonu yorumu
        }
        """
        
        self.assertProcessingResult(
            processor, 
            js_code, 
            language="javascript",
            not_expected_parts=[
                "// Bu bir satır yorumu",
                "/* Bu bir blok",
                "// Satır sonu yorumu"
            ],
            expected_parts=[
                "function test()",
                "var x = 10;"
            ]
        )
    
    def test_normalize_whitespace(self):
        """
        Boşluk normalleştirme işlevini test eder.
        """
        processor = self.CodeProcessor(normalize_whitespace=True)
        
        # Boşluklu metin
        text = "function  test()   {\n    var  x  =  10;\n}"
        
        self.assertProcessingResult(
            processor,
            text,
            expected_parts=[
                "function test()", 
                "    var x = 10"
            ]
        )