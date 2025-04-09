"""
Konfigürasyon yönetimi testleri.

Bu modül, config paketi için birim testleri içerir.
"""

import os
import unittest
import tempfile
from pathlib import Path

# Testlerde kullanılacak örnek YAML içeriği
SAMPLE_CONFIG = """
# Örnek yapılandırma

tokenizer:
  name: test_tokenizer
  vocab_size: 30000
  special_tokens:
    - "[PAD]"
    - "[UNK]"
    - "[CLS]"
    - "[SEP]"

data:
  sources:
    - type: local
      path: "./data/corpus"
  processors:
    - name: text_cleaner
      type: text
      remove_urls: true
      lowercase: false
    - name: code_cleaner
      type: code
      remove_comments: true

training:
  max_samples: 1000000
  batch_size: 1000
  epochs: 2

output:
  path: "./output"
"""


class TestConfigManager(unittest.TestCase):
    """
    ConfigManager sınıfı için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Test için geçici dosya oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        
        # Örnek yapılandırmayı dosyaya yaz
        with open(self.config_path, "w") as f:
            f.write(SAMPLE_CONFIG)
        
        # ConfigManager'ı import et
        try:
            from aiquantr_tokenizer.config.config_manager import ConfigManager
            self.ConfigManager = ConfigManager
        except ImportError:
            self.skipTest("aiquantr_tokenizer.config.config_manager modülü bulunamadı")
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        # Geçici dizini temizle
        self.temp_dir.cleanup()
    
    def test_load_config(self):
        """
        YAML dosyasından yapılandırma yükleme işlemini test eder.
        """
        # ConfigManager örneği oluştur ve yapılandırmayı doğrudan yükle
        config_manager = self.ConfigManager(config_path=self.config_path)
        
        # Değerleri kontrol et
        self.assertEqual(config_manager.get("tokenizer.name"), "test_tokenizer")
        self.assertEqual(config_manager.get("tokenizer.vocab_size"), 30000)
        self.assertIn("[PAD]", config_manager.get("tokenizer.special_tokens"))
        self.assertEqual(len(config_manager.get("tokenizer.special_tokens")), 4)
        
        # Veri işlemcileri
        processors = config_manager.get("data.processors")
        self.assertEqual(len(processors), 2)
        self.assertEqual(processors[0]["name"], "text_cleaner")
        self.assertTrue(processors[0]["remove_urls"])
        self.assertFalse(processors[0]["lowercase"])
    
    def test_get_with_default(self):
        """
        Varsayılan değerli get işlemini test eder.
        """
        config_manager = self.ConfigManager(config_path=self.config_path)
        
        # Varolan bir değer al
        self.assertEqual(config_manager.get("tokenizer.vocab_size"), 30000)
        
        # Olmayan bir değer için varsayılanı al
        self.assertEqual(config_manager.get("tokenizer.min_frequency", 5), 5)
        
        # İç içe olmayan bir değer için varsayılanı al
        self.assertEqual(config_manager.get("nonexistent.key", "default"), "default")
    
    def test_set_and_save(self):
        """
        Yapılandırma değiştirme ve kaydetme işlemlerini test eder.
        """
        config_manager = self.ConfigManager(config_path=self.config_path)
        
        # Değer değiştir
        config_manager.set("tokenizer.vocab_size", 50000)
        config_manager.set("data.max_length", 512)
        
        # Değişiklikleri kontrol et
        self.assertEqual(config_manager.get("tokenizer.vocab_size"), 50000)
        self.assertEqual(config_manager.get("data.max_length"), 512)
        
        # Yeni bir dosyaya kaydet
        new_config_path = os.path.join(self.temp_dir.name, "new_config.yaml")
        config_manager.save_config(new_config_path)
        
        # Yeni dosyayı oku ve değerleri kontrol et
        new_config_manager = self.ConfigManager(config_path=new_config_path)
        
        self.assertEqual(new_config_manager.get("tokenizer.vocab_size"), 50000)
        self.assertEqual(new_config_manager.get("data.max_length"), 512)
    
    def test_merge_configs(self):
        """
        Yapılandırma birleştirme işlemini test eder.
        """
        config_manager = self.ConfigManager(config_path=self.config_path)
        
        # Yeni bir yapılandırma oluştur
        override_config = {
            "tokenizer": {
                "vocab_size": 40000,
                "new_option": True
            },
            "training": {
                "batch_size": 2000
            }
        }
        
        # Yapılandırmaları birleştir
        config_manager.update_config(override_config)
        
        # Değişiklikleri kontrol et
        self.assertEqual(config_manager.get("tokenizer.vocab_size"), 40000)
        self.assertTrue(config_manager.get("tokenizer.new_option"))
        self.assertEqual(config_manager.get("training.batch_size"), 2000)
        self.assertEqual(config_manager.get("training.epochs"), 2)  # Değişmedi
        
        # Özel tokenler korundu mu?
        self.assertIn("[PAD]", config_manager.get("tokenizer.special_tokens"))

    def test_validate_config(self):
        """
        Konfigürasyon doğrulama işlemini test eder.
        """
        # Geçerli bir yapılandırma
        config_manager = self.ConfigManager(config_path=self.config_path)
        
        # output.path alanı gerekliydi, SAMPLE_CONFIG'e eklendiğinden şimdi geçerli olmalı
        from aiquantr_tokenizer.config.config_manager import validate_config
        self.assertTrue(validate_config(config_manager.config))
        
        # Geçersiz bir yapılandırma oluştur (data alanı olmayan)
        invalid_config_path = os.path.join(self.temp_dir.name, "invalid_config.yaml")
        with open(invalid_config_path, "w") as f:
            f.write("""
            # Geçersiz yapılandırma
            tokenizer:
              name: invalid_tokenizer
            """)
        
        # Geçersiz yapılandırmayı yükle, ConfigManager en azından minimal yapıyı sağlamalı
        invalid_config_manager = self.ConfigManager(config_path=invalid_config_path)
        
        # Temel alanlar hâlâ mevcut olmalı
        self.assertIsNotNone(invalid_config_manager.get("tokenizer"))
        self.assertIsNotNone(invalid_config_manager.get("data"))


if __name__ == "__main__":
    unittest.main()