"""
İşlemciler testleri.

Bu modül, processors paketi için birim testleri içerir.
"""

import unittest
from typing import List, Dict  # List ve Dict türlerini import edin
# Testlerde kullanılacak örnek içerikler
SAMPLE_TEXT = """Bu bir örnek metin.
https://example.com adresine göz atın.
E-posta: user@example.com

   Bölümler   arasında   fazla   boşluk   var.
123,456 ve 789 sayılar. !!!
"""

SAMPLE_CODE_PYTHON = """#!/usr/bin/env python3
# Bu bir örnek Python dosyasıdır

import os
import sys
from typing import List, Dict

'''
Bu çok satırlı
bir yorum
"""

def example_function(param1: str, param2: int = 10) -> List[str]:
    """
    Örnek bir fonksiyon.
    
    Args:
        param1: Birinci parametre
        param2: İkinci parametre (varsayılan: 10)
        
    Returns:
        str listesi
    """
    # Değişken tanımları
    result = []
    
    # İşlem
    for i in range(param2):
        result.append(f"{param1}_{i}")
    
    return result

class ExampleClass:
    """Örnek bir sınıf."""
    
    def __init__(self, name):
        self.name = name
        self.items = []
    
    def add_item(self, item):
        # Öğe ekle
        self.items.append(item)
        print(f"Öğe eklendi: {item}")

if __name__ == "__main__":
    # Ana program
    example = ExampleClass("test")
    example.add_item("örnek")
    
    # Fonksiyon çağrısı
    results = example_function("test", 5)
    print(f"Sonuçlar: {results}")


SAMPLE_CODE_PHP = """<?php
// Bu bir PHP örneğidir

/**
 * Örnek bir fonksiyon
 * 
 * @param string $param1 Birinci parametre
 * @param int $param2 İkinci parametre
 * @return array
 */
function example_function($param1, $param2 = 10) {
    // Değişken tanımları
    $result = [];
    
    // İşlem
    for ($i = 0; $i < $param2; $i++) {
        $result[] = "{$param1}_{$i}";
    }
    
    return $result;
}

class ExampleClass {
    private $name;
    private $items = [];
    
    public function __construct($name) {
        $this->name = $name;
    }
    
    /**
     * Bir öğe ekler
     */
    public function addItem($item) {
        // Öğe ekle
        $this->items[] = $item;
        echo "Öğe eklendi: $item";
    }
}

// Ana program
$example = new ExampleClass("test");
$example->addItem("örnek");

// Fonksiyon çağrısı
$results = example_function("test", 5);
echo "Sonuçlar: " . json_encode($results);
?>

<html>
<body>
    <h1>PHP HTML içerebilir</h1>
</body>
</html>
"""


class TestBaseProcessor(unittest.TestCase):
    """
    BaseProcessor sınıfı için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # BaseProcessor'u import et
        try:
            from aiquantr_tokenizer.processors.base_processor import BaseProcessor
            self.BaseProcessor = BaseProcessor
        except ImportError:
            self.skipTest("aiquantr_tokenizer.processors.base_processor modülü bulunamadı")
            
        # Test sınıfı oluştur
        class TestProcessor(self.BaseProcessor):
            """Test işlemci sınıfı"""
            
            def process(self, data):
                return f"Processed: {data}"
        
        self.TestProcessor = TestProcessor
    
    def test_call_method(self):
        """
        __call__ metodunu test eder.
        """
        processor = self.TestProcessor()
        
        # İşlemi çağır
        result = processor("test")
        
        # Kontroller
        self.assertEqual(result, "Processed: test")
        self.assertEqual(processor.stats["processed_count"], 1)
    
    def test_stats_tracking(self):
        """
        İstatistik izlemeyi test eder.
        """
        processor = self.TestProcessor()
        
        # String işleme
        processor("test string")
        
        # İstatistik kontrolü
        self.assertEqual(processor.stats["processed_count"], 1)
        self.assertEqual(processor.stats["total_chars_in"], 11)  # "test string"
        
        # İstatistikleri sıfırla
        processor.reset_stats()
        
        # İstatistik kontrolü
        self.assertEqual(processor.stats["processed_count"], 0)
        self.assertEqual(processor.stats["total_chars_in"], 0)


class TestTextProcessor(unittest.TestCase):
    """
    TextProcessor sınıfı için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # TextProcessor'u import et
        try:
            from aiquantr_tokenizer.processors.language_processor import TextProcessor
            self.TextProcessor = TextProcessor
        except ImportError:
            self.skipTest("aiquantr_tokenizer.processors.language_processor modülü bulunamadı")
    
    def test_text_processing(self):
        """
        Metin işleme işlemini test eder.
        """
        processor = self.TextProcessor(
            lowercase=True,
            normalize_whitespace=True,
            replace_urls=True,
            replace_emails=True
        )
        
        # Metni işle
        result = processor(SAMPLE_TEXT)
        
        # Kontroller
        self.assertTrue(result.islower())
        self.assertIn("[url]", result.lower())
        self.assertIn("[email]", result.lower())
        self.assertNotIn("https://", result.lower())
        self.assertNotIn("user@example.com", result.lower())
        
        # Normalleştirilmiş boşluklar
        self.assertNotIn("   bölümler   ", result.lower())
    
    def test_remove_punct(self):
        """
        Noktalama işaretlerini kaldırmayı test eder.
        """
        processor = self.TextProcessor(remove_punct=True)
        
        # Metni işle
        result = processor(SAMPLE_TEXT)
        
        # Kontroller
        self.assertNotIn("!!!", result)
        self.assertNotIn(".", result)
        self.assertNotIn(",", result)
    
    def test_min_max_length(self):
        """
        Minimum ve maksimum uzunluk sınırlarını test eder.
        """
        # Çok büyük minimum uzunluk
        processor = self.TextProcessor(min_length=1000)
        result = processor(SAMPLE_TEXT)
        self.assertEqual(result, "")
        
        # Maksimum uzunluk sınırı
        processor = self.TextProcessor(max_length=10)
        result = processor(SAMPLE_TEXT)
        self.assertEqual(len(result), 10)


class TestCodeProcessor(unittest.TestCase):
    """
    Kod işlemcileri için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Kod işlemcilerini import et
        try:
            from aiquantr_tokenizer.processors.code.general import CodeProcessor
            from aiquantr_tokenizer.processors.code.python import PythonProcessor
            from aiquantr_tokenizer.processors.code.php import PhpProcessor
            
            self.CodeProcessor = CodeProcessor
            self.PythonProcessor = PythonProcessor
            self.PhpProcessor = PhpProcessor
        except ImportError:
            self.skipTest("aiquantr_tokenizer.processors.code modülü bulunamadı")
    
    def test_general_code_processor(self):
        """
        Genel kod işlemcisini test eder.
        """
        processor = self.CodeProcessor(
            remove_comments=True,
            normalize_whitespace=True
        )
        
        # Python kodunu işle
        result = processor.process(SAMPLE_CODE_PYTHON, language="python")
        
        # Kontroller
        self.assertNotIn("# Bu bir örnek Python dosyasıdır", result)
        self.assertIn("def example_function", result)
    
    def test_python_processor(self):
        """
        Python kod işlemcisini test eder.
        """
        processor = self.PythonProcessor(
            remove_comments=True,
            remove_docstrings=True,
            remove_type_hints=True
        )
        
        # Python kodunu işle
        result = processor(SAMPLE_CODE_PYTHON)
        
        # Kontroller
        self.assertNotIn("# Bu bir örnek Python dosyasıdır", result)
        self.assertNotIn('"""Örnek bir fonksiyon."""', result)
        self.assertIn("def example_function(param1, param2 = 10):", result)  # Tip işaretleri kaldırıldı
        self.assertNotIn("-> List[str]", result)
    
    def test_php_processor(self):
        """
        PHP kod işlemcisini test eder.
        """
        processor = self.PhpProcessor(
            remove_comments=True,
            remove_php_tags=True,
            remove_html=True
        )
        
        # PHP kodunu işle
        result = processor(SAMPLE_CODE_PHP)
        
        # Kontroller
        self.assertNotIn("// Bu bir PHP örneğidir", result)
        self.assertNotIn("/**", result)
        self.assertNotIn("<?php", result)
        self.assertNotIn("?>", result)
        self.assertNotIn("<html>", result)
        self.assertIn("function example_function", result)


class TestKnowledgeProcessors(unittest.TestCase):
    """
    Bilgi işlemcileri için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Bilgi işlemcilerini import et
        try:
            from aiquantr_tokenizer.processors.knowledge.general import GeneralKnowledgeProcessor
            from aiquantr_tokenizer.processors.knowledge.domain_specific import DomainSpecificProcessor
            
            self.GeneralKnowledgeProcessor = GeneralKnowledgeProcessor
            self.DomainSpecificProcessor = DomainSpecificProcessor
        except ImportError:
            self.skipTest("aiquantr_tokenizer.processors.knowledge modülü bulunamadı")
    
    def test_general_knowledge_processor(self):
        """
        Genel bilgi işlemcisini test eder.
        """
        sample_text = """
        Yapay zeka (AI), insan zekasını simüle etmek için tasarlanmış makineleri ifade eder.
        Alan Turing, 1950'de bir makinenin düşünüp düşünemeyeceğini sorguladı.
        Bugün AI, doğal dil işleme ve bilgisayarlı görü gibi birçok alanda kullanılmaktadır.
        
        Deep learning, yapay sinir ağları kullanarak büyük veri kümelerinden öğrenme yeteneğine sahiptir.
        Kaynak: https://example.com/ai-history
        """
        
        processor = self.GeneralKnowledgeProcessor(
            remove_citations=True,
            remove_urls=True,
            extract_entities=True
        )
        
        # Metni işle
        result = processor(sample_text)
        
        # Kontroller
        self.assertIn("Yapay zeka (AI)", result)
        self.assertIn("Alan Turing", result)
        self.assertNotIn("https://example.com/ai-history", result)
        
        # Varlık çıkarma
        entities = processor.extract_entities_from_text(result)
        entity_texts = [e["text"] for e in entities]
        self.assertIn("Alan Turing", entity_texts)
    
    def test_domain_specific_processor(self):
        """
        Alan-spesifik bilgi işlemcisini test eder.
        """
        medical_text = """
        Diyabet, pankreasın yeterli insülin üretmemesi veya vücudun ürettiği insülini etkili bir
        şekilde kullanamaması sonucu ortaya çıkan kronik bir hastalıktır. Tip 1 ve Tip 2 olmak
        üzere iki ana türü vardır. Hastaların kan şekeri seviyeleri genellikle 126 mg/dL
        veya daha yüksektir.
        """
        
        # Tıbbi alan işlemcisi
        processor = self.DomainSpecificProcessor(
            domain="medical",
            extract_domain_entities=True,
            domain_specific_patterns={
                "disease": r'\b(?:diyabet|kanser|astım|alzheimer)\b',
                "measurement": r'\b\d+(?:\.\d+)?\s*(?:mg/dL)\b'
            }
        )
        
        # Metni işle
        result = processor(medical_text)
        
        # Varlık çıkarma
        entities = processor.extract_entities_from_text(result)
        entity_texts = [e["text"] for e in entities]
        entity_types = [e["type"] for e in entities]
        
        # Kontroller
        self.assertIn("diyabet", entity_texts)
        self.assertIn("126 mg/dL", entity_texts)
        self.assertIn("DISEASE", entity_types)
        self.assertIn("MEASUREMENT", entity_types)


if __name__ == "__main__":
    unittest.main()