"""
Veri kaynakları testleri.

Bu modül, data.sources paketi için birim testleri içerir.
"""

import os
import unittest
import tempfile
from pathlib import Path

# Testlerde kullanılacak örnek veri içeriği
SAMPLE_TEXT = """Bu bir örnek metin dosyasıdır.
Tokenizer eğitimi için kullanılabilir.
Birkaç paragraf içerir."""

SAMPLE_JSON = """
{
    "text": "Bu bir JSON örneğidir.",
    "metadata": {
        "source": "test",
        "language": "tr"
    }
}
"""

SAMPLE_JSONL = """{"text": "Bu bir JSONL dosyasının ilk satırıdır.", "id": 1}
{"text": "Bu ikinci satırdır.", "id": 2}
{"text": "Bu üçüncü ve son satırdır.", "id": 3}
"""

SAMPLE_CSV = """id,text,language
1,"Bu bir CSV dosyasının ilk satırıdır.","tr"
2,"Bu ikinci satırdır.","tr"
3,"Bu üçüncü ve son satırdır.","tr"
"""


class TestBaseDataSource(unittest.TestCase):
    """
    BaseDataSource sınıfı için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Test için geçici dizin oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Örnek dosyaları oluştur
        self.text_file = self.data_dir / "sample.txt"
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TEXT)
        
        self.json_file = self.data_dir / "sample.json"
        with open(self.json_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_JSON)
            
        self.jsonl_file = self.data_dir / "sample.jsonl"
        with open(self.jsonl_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_JSONL)
            
        self.csv_file = self.data_dir / "sample.csv"
        with open(self.csv_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CSV)
        
        # BaseDataSource'u import et
        try:
            from aiquantr_tokenizer.data.sources.base_source import BaseDataSource
            self.BaseDataSource = BaseDataSource
        except ImportError:
            self.skipTest("aiquantr_tokenizer.data.sources.base_source modülü bulunamadı")
            
        # Test sınıfı oluştur
        class TestSource(self.BaseDataSource):
            """Test kaynak sınıfı"""
            
            def load_data(self):
                return [{"text": "Test veri 1"}, {"text": "Test veri 2"}]
        
        self.TestSource = TestSource
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        # Geçici dizini temizle
        self.temp_dir.cleanup()
    
    def test_filter_text(self):
        """
        Metin filtreleme işlemini test eder.
        """
        source = self.TestSource(min_length=10, max_length=20)
        
        # Çok kısa metin
        self.assertIsNone(source.filter_text("Kısa"))
        
        # Uygun uzunlukta metin
        self.assertEqual(source.filter_text("Bu uygun uzunlukta."), "Bu uygun uzunlukta.")
        
        # Çok uzun metin (kesilir)
        long_text = "Bu çok uzun bir metin olduğu için kesilecektir."
        self.assertEqual(source.filter_text(long_text), long_text[:20])
        
        # Boş metin
        self.assertIsNone(source.filter_text(""))
    
    def test_process_item(self):
        """
        Veri öğesi işleme işlemini test eder.
        """
        source = self.TestSource(min_length=5)
        
        # Geçerli öğe
        item = {"text": "Bu geçerli bir öğedir."}
        processed = source.process_item(item)
        self.assertEqual(processed, item)
        
        # Text anahtarı olmayan öğe
        item = {"content": "Bu farklı bir anahtara sahip."}
        self.assertIsNone(source.process_item(item))
        
        # Çok kısa metin içeren öğe
        item = {"text": "Kısa"}
        self.assertIsNone(source.process_item(item))
    
    def test_metadata(self):
        """
        Üst veri işleme işlemini test eder.
        """
        source = self.TestSource(
            name="Test Kaynağı", 
            description="Test açıklaması", 
            metadata={"source_type": "test", "version": "1.0"}
        )
        
        # Üst verileri al
        metadata = source.get_metadata()
        
        # Kontroller
        self.assertEqual(metadata["source_name"], "Test Kaynağı")
        self.assertEqual(metadata["source_type"], "TestSource")
        self.assertEqual(metadata["description"], "Test açıklaması")
        self.assertEqual(metadata["version"], "1.0")


class TestLocalFileSource(unittest.TestCase):
    """
    LocalFileSource sınıfı için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Test için geçici dizin oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Örnek dosyaları oluştur
        self.text_file = self.data_dir / "sample.txt"
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TEXT)
        
        self.json_file = self.data_dir / "sample.json"
        with open(self.json_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_JSON)
            
        self.jsonl_file = self.data_dir / "sample.jsonl"
        with open(self.jsonl_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_JSONL)
            
        self.csv_file = self.data_dir / "sample.csv"
        with open(self.csv_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CSV)
        
        # LocalFileSource'u import et
        try:
            from aiquantr_tokenizer.data.sources.local_source import LocalFileSource
            self.LocalFileSource = LocalFileSource
        except ImportError:
            self.skipTest("aiquantr_tokenizer.data.sources.local_source modülü bulunamadı")
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        # Geçici dizini temizle
        self.temp_dir.cleanup()
    
    def test_load_text_file(self):
        """
        Metin dosyası yüklemeyi test eder.
        """
        source = self.LocalFileSource(file_path=self.text_file)
        
        # Veriyi yükle
        items = list(source.load_data())
        
        # Kontroller
        self.assertEqual(len(items), 1)
        self.assertIn("text", items[0])
        self.assertEqual(items[0]["text"], SAMPLE_TEXT)
    
    def test_load_json_file(self):
        """
        JSON dosyası yüklemeyi test eder.
        """
        source = self.LocalFileSource(file_path=self.json_file)
        
        # Veriyi yükle
        items = list(source.load_data())
        
        # Kontroller
        self.assertEqual(len(items), 1)
        self.assertIn("text", items[0])
        self.assertEqual(items[0]["text"], "Bu bir JSON örneğidir.")
        self.assertIn("metadata", items[0])
        self.assertEqual(items[0]["metadata"]["language"], "tr")
    
    def test_load_jsonl_file(self):
        """
        JSONL dosyası yüklemeyi test eder.
        """
        source = self.LocalFileSource(file_path=self.jsonl_file, jsonl_mode=True)
        
        # Veriyi yükle
        items = list(source.load_data())
        
        # Kontroller
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["id"], 1)
        self.assertEqual(items[1]["id"], 2)
        self.assertEqual(items[2]["id"], 3)
        self.assertEqual(items[0]["text"], "Bu bir JSONL dosyasının ilk satırıdır.")
    
    def test_load_csv_file(self):
        """
        CSV dosyası yüklemeyi test eder.
        """
        # pandas yüklü değilse testi atla
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas yüklü değil")
        
        source = self.LocalFileSource(file_path=self.csv_file)
        
        # Veriyi yükle
        items = list(source.load_data())
        
        # Kontroller
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["id"], 1)
        self.assertEqual(items[1]["id"], 2)
        self.assertEqual(items[2]["id"], 3)
        self.assertEqual(items[0]["text"], "Bu bir CSV dosyasının ilk satırıdır.")
        self.assertEqual(items[0]["language"], "tr")


class TestLocalDirSource(unittest.TestCase):
    """
    LocalDirSource sınıfı için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Test için geçici dizin oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Alt dizin oluştur
        self.sub_dir = self.data_dir / "subdir"
        self.sub_dir.mkdir(exist_ok=True)
        
        # Örnek dosyaları oluştur
        # Ana dizindeki dosyalar
        with open(self.data_dir / "file1.txt", "w", encoding="utf-8") as f:
            f.write("Bu ana dizindeki ilk dosya.")
        
        with open(self.data_dir / "file2.txt", "w", encoding="utf-8") as f:
            f.write("Bu ana dizindeki ikinci dosya.")
            
        # Alt dizindeki dosyalar
        with open(self.sub_dir / "file3.txt", "w", encoding="utf-8") as f:
            f.write("Bu alt dizindeki bir dosya.")
        
        with open(self.sub_dir / "data.json", "w", encoding="utf-8") as f:
            f.write("""{"text": "Bu bir JSON dosyası", "id": 1}""")
        
        # LocalDirSource'u import et
        try:
            from aiquantr_tokenizer.data.sources.local_source import LocalDirSource
            self.LocalDirSource = LocalDirSource
        except ImportError:
            self.skipTest("aiquantr_tokenizer.data.sources.local_source modülü bulunamadı")
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        # Geçici dizini temizle
        self.temp_dir.cleanup()
    
    def test_load_files_recursive(self):
        """
        Alt dizinleri içeren yüklemeyi test eder.
        """
        source = self.LocalDirSource(
            directory=self.data_dir,
            pattern="**/*.txt",
            recursive=True
        )
        
        # Veriyi yükle
        items = list(source.load_data())
        
        # Kontroller
        self.assertEqual(len(items), 3)  # 3 metin dosyası
        
        # Metin içeriklerini kontrol et
        texts = [item["text"] for item in items]
        self.assertIn("Bu ana dizindeki ilk dosya.", texts)
        self.assertIn("Bu ana dizindeki ikinci dosya.", texts)
        self.assertIn("Bu alt dizindeki bir dosya.", texts)
    
    def test_load_files_nonrecursive(self):
        """
        Sadece ana dizinden yüklemeyi test eder.
        """
        source = self.LocalDirSource(
            directory=self.data_dir,
            pattern="*.txt",
            recursive=False
        )
        
        # Veriyi yükle
        items = list(source.load_data())
        
        # Kontroller
        self.assertEqual(len(items), 2)  # Sadece ana dizindeki 2 metin dosyası
        
        # Metin içeriklerini kontrol et
        texts = [item["text"] for item in items]
        self.assertIn("Bu ana dizindeki ilk dosya.", texts)
        self.assertIn("Bu ana dizindeki ikinci dosya.", texts)
        self.assertNotIn("Bu alt dizindeki bir dosya.", texts)
    
    def test_load_with_file_types(self):
        """
        Dosya türü filtrelemeyi test eder.
        """
        source = self.LocalDirSource(
            directory=self.data_dir,
            pattern="**/*.*",
            recursive=True,
            file_types=[".json"]
        )
        
        # Veriyi yükle
        items = list(source.load_data())
        
        # Kontroller
        self.assertEqual(len(items), 1)  # Sadece 1 JSON dosyası
        self.assertEqual(items[0]["text"], "Bu bir JSON dosyası")
        self.assertEqual(items[0]["id"], 1)


if __name__ == "__main__":
    unittest.main()