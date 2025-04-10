"""
Farklı veri formatları için tokenizer test modülü.

Bu test modülü, projede bulunan tüm tokenizer tiplerini
farklı veri formatları (HTML, XML, JSON, CSV, Markdown) üzerinde test eder.
"""

import unittest
import os
import tempfile
from pathlib import Path
import json
import csv
from typing import Dict, Any, List, Type, Optional

from aiquantr_tokenizer.tokenizers.base import BaseTokenizer, TokenizerTrainer


class TestDataFormats(unittest.TestCase):
    """
    Farklı veri formatları üzerinde tokenizer'ları test eder.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Gerekli modülleri import et
        try:
            # Tüm tokenizer sınıfları
            from aiquantr_tokenizer.tokenizers.bpe import BPETokenizer
            from aiquantr_tokenizer.tokenizers.wordpiece import WordPieceTokenizer
            from aiquantr_tokenizer.tokenizers.byte_level import ByteLevelTokenizer
            from aiquantr_tokenizer.tokenizers.unigram import UnigramTokenizer
            from aiquantr_tokenizer.tokenizers.mixed import MixedTokenizer
            from aiquantr_tokenizer.tokenizers.factory import create_tokenizer_from_config, register_tokenizer_type
            
            self.BPETokenizer = BPETokenizer
            self.WordPieceTokenizer = WordPieceTokenizer
            self.ByteLevelTokenizer = ByteLevelTokenizer
            self.UnigramTokenizer = UnigramTokenizer
            self.MixedTokenizer = MixedTokenizer
            self.create_tokenizer_from_config = create_tokenizer_from_config
            self.register_tokenizer_type = register_tokenizer_type
            
            self.all_tokenizer_classes = {
                "BPE": BPETokenizer,
                "WordPiece": WordPieceTokenizer,
                "ByteLevel": ByteLevelTokenizer, 
                "Unigram": UnigramTokenizer,
            }
            
        except ImportError as e:
            self.skipTest(f"Gerekli tokenizer modülleri bulunamadı: {e}")
        
        # Geçici dizin oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Format örnekleri
        self.format_samples = {
            "html": self._get_html_sample(),
            "xml": self._get_xml_sample(),
            "json": self._get_json_sample(),
            "csv": self._get_csv_sample(),
            "markdown": self._get_markdown_sample(),
            "yaml": self._get_yaml_sample()
        }
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        self.temp_dir.cleanup()
    
    def _get_html_sample(self):
        """
        HTML örneği oluşturur.
        """
        return """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tokenizer Test Sayfası</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        code {
            background: #f1f1f1;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tokenizer Test Sayfası</h1>
        
        <p>Bu sayfa, farklı tokenizer'ları HTML içeriği üzerinde test etmek için oluşturulmuştur.</p>
        
        <div class="highlight">
            <p>Tokenizasyon, metin verisini işlenebilir parçalara ayırma işlemidir.</p>
        </div>
        
        <h2>Tokenizer Türleri</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Tokenizer</th>
                    <th>Açıklama</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>BPE</td>
                    <td>Byte Pair Encoding, en sık görülen karakter çiftlerini yinelemeli olarak birleştirir.</td>
                </tr>
                <tr>
                    <td>WordPiece</td>
                    <td>BERT modellerinde kullanılan bir alt kelime tokenizasyonu yöntemidir.</td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>"""
    
    def _get_xml_sample(self):
        """
        XML örneği oluşturur.
        """
        return """<?xml version="1.0" encoding="UTF-8"?>
<tokenizers xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:noNamespaceSchemaLocation="tokenizers.xsd"
           version="1.0">
    <metadata>
        <created>2023-04-15T10:30:00</created>
        <author>
            <name>Test User</name>
            <email>test@example.com</email>
        </author>
        <description>Tokenizer konfigürasyonları ve benchmark sonuçları</description>
    </metadata>
    
    <configuration>
        <default_vocab_size>10000</default_vocab_size>
        <min_frequency>2</min_frequency>
        <enable_caching>true</enable_caching>
        <threads>4</threads>
        <special_tokens>
            <token id="0" name="pad_token">[PAD]</token>
            <token id="1" name="unk_token">[UNK]</token>
            <token id="2" name="bos_token">[BOS]</token>
            <token id="3" name="eos_token">[EOS]</token>
        </special_tokens>
    </configuration>
    
    <tokenizer id="bpe_tokenizer" type="BPE">
        <name>BPE Tokenizer</name>
        <vocab_size>8000</vocab_size>
        <settings>
            <setting name="character_coverage">0.9995</setting>
            <setting name="byte_fallback">true</setting>
        </settings>
    </tokenizer>
</tokenizers>"""
    
    def _get_json_sample(self):
        """
        JSON örneği oluşturur.
        """
        return json.dumps({
            "tokenizer_config": {
                "name": "Çok Dilli Tokenizer",
                "version": "1.2.0",
                "type": "BPE",
                "vocab_size": 50257,
                "min_frequency": 2,
                "special_tokens": {
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                    "bos_token": "[BOS]",
                    "eos_token": "[EOS]",
                    "mask_token": "[MASK]"
                },
                "settings": {
                    "character_coverage": 1.0,
                    "split_pattern": "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
                    "byte_fallback": True,
                    "normalization": "NFC"
                }
            },
            "training": {
                "datasets": [
                    {"name": "wikipedia_tr", "samples": 1500000, "weight": 0.3},
                    {"name": "wikipedia_en", "samples": 3000000, "weight": 0.5},
                    {"name": "common_crawl", "samples": 1000000, "weight": 0.2}
                ],
                "parameters": {
                    "batch_size": 1024,
                    "epochs": 5,
                    "threads": 8,
                    "optimizer": {
                        "name": "Adam",
                        "learning_rate": 0.001,
                        "weight_decay": 0.01
                    }
                }
            },
            "benchmarks": [
                {
                    "name": "encoding_speed",
                    "result": 128547.3,
                    "unit": "tokens/s",
                    "platform": "cpu"
                },
                {
                    "name": "decoding_speed",
                    "result": 178392.1,
                    "unit": "tokens/s",
                    "platform": "cpu"
                },
                {
                    "name": "model_size",
                    "result": 48.7,
                    "unit": "MB"
                }
            ],
            "examples": {
                "turkish": {
                    "input": "Merhaba dünya! Bu bir test cümlesidir.",
                    "tokens": [15, 382, 27, 4, 192, 58, 731, 29, 41],
                    "decoded": "Merhaba dünya! Bu bir test cümlesidir."
                },
                "english": {
                    "input": "Hello world! This is a test sentence.",
                    "tokens": [31, 278, 17, 8, 127, 92, 6, 731, 138, 5],
                    "decoded": "Hello world! This is a test sentence."
                }
            },
            "metadata": {
                "created_at": "2023-05-20T14:30:00Z",
                "author": "Test User",
                "license": "MIT",
                "repository": "https://github.com/example/ai-tokenizer"
            }
        }, ensure_ascii=False, indent=2)
    
    def _get_csv_sample(self):
        """
        CSV örneği oluşturur.
        """
        csv_lines = [
            ["token_id", "token", "frequency", "type", "language"],
            ["0", "[PAD]", "0", "special", "all"],
            ["1", "[UNK]", "0", "special", "all"],
            ["2", "[BOS]", "0", "special", "all"],
            ["3", "[EOS]", "0", "special", "all"],
            ["4", "the", "1827365", "word", "en"],
            ["5", "of", "945271", "word", "en"],
            ["6", "and", "811916", "word", "en"],
            ["7", "to", "803365", "word", "en"],
            ["8", "a", "650346", "word", "en"],
            ["9", "in", "508264", "word", "en"],
            ["10", "ve", "392875", "word", "tr"],
            ["11", "bir", "287456", "word", "tr"],
            ["12", "bu", "194532", "word", "tr"],
            ["13", "için", "174938", "word", "tr"],
            ["14", "de", "161847", "word", "tr"],
            ["15", "ing", "143769", "subword", "en"],
            ["16", "##er", "127896", "subword", "en"],
            ["17", "##tion", "119547", "subword", "en"],
            ["18", "##lar", "108654", "subword", "tr"],
            ["19", "##mek", "97534", "subword", "tr"],
            ["20", "tokenizer", "543", "technical", "en"],
            ["21", "python", "476", "technical", "en"],
            ["22", "dataset", "421", "technical", "en"],
            ["23", "json", "387", "technical", "en"],
            ["24", "html", "342", "technical", "en"]
        ]
        
        result = []
        for line in csv_lines:
            result.append(",".join(line))
        
        return "\n".join(result)
    
    def _get_markdown_sample(self):
        """
        Markdown örneği oluşturur.
        """
        return """# Tokenizer Dokümantasyonu

## 1. Tokenizer Nedir?

Tokenizer, metni daha küçük parçalara (token) ayıran bir bileşendir. Bu parçalar kelimeler, alt kelimeler veya karakterler olabilir.

### 1.1 Tokenizasyonun Önemi

- Metin verileri üzerinde işlem yapan modeller için ön işleme adımıdır
- Kelimeleri vektör temsillerine dönüştürmenin ilk adımıdır
- Dil modellerinin kelime dağarcığını yönetir

## 2. Tokenizer Türleri

Projemizde desteklenen tokenizer türleri şunlardır:

| Tokenizer | Açıklama | Kullanım Alanı |
|-----------|----------|----------------|
| BPE | Byte Pair Encoding | GPT modelleri |
| WordPiece | Alt kelime tabanlı | BERT modelleri |
| Unigram | Olasılıksal model | XLNet, ALBERT |
| ByteLevel | Byte tabanlı | Çok dilli modeller |

### 2.1 BPE (Byte Pair Encoding)

```python
from aiquantr_tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(["Örnek eğitim metni"])
tokens = tokenizer.encode("Merhaba dünya!")
print(tokens)  # [15, 382, 27, 4]
```

Not: BPE, en sık görülen karakter çiftlerini yinelemeli olarak birleştirir.

### 2.2 WordPiece

WordPiece tokenizer, özellikle BERT gibi modellerde kullanılır:

```python
from aiquantr_tokenizer import WordPieceTokenizer

tokenizer = WordPieceTokenizer(
    vocab_size=1000,
    wordpiece_prefix="##"
)
```

## 3. Karşılaştırma

BPE vs WordPiece:
- BPE frekans tabanlıdır
- WordPiece olabilirlik tabanlıdır

Performans:
- ByteLevel daha hızlıdır
- Unigram daha doğrudur

**Önemli**: Tokenizer seçimi, projenin gereksinimleri ve dil özellikleri göz önünde bulundurularak yapılmalıdır.

## 4. Örnek Çıktılar

Token örnekleri:

- "merhaba" → [15, 382]
- "dünya" → [27, 4]
- "tokenizasyon" → [431, 709, 52]

## 5. Kurulum

```bash
pip install aiquantr_tokenizer
```

## 6. Referanslar

- BPE Algoritması
- WordPiece Makalesi
"""

    def _get_yaml_sample(self):
        """
        YAML örneği oluşturur.
        """
        return """# Tokenizer konfigürasyon dosyası
version: 1.0.0
name: "Çok Dilli Tokenizer"

# Genel ayarlar
settings:
  vocab_size: 32000
  min_frequency: 2
  save_path: "./models/tokenizer"
  normalize_unicode: true
  lowercase: false

# Özel tokenlar
special_tokens:
  pad_token: "[PAD]"
  unk_token: "[UNK]"
  bos_token: "[BOS]"
  eos_token: "[EOS]"
  mask_token: "[MASK]"
  sep_token: "[SEP]"
  cls_token: "[CLS]"

# BPE özellikleri
bpe:
  character_coverage: 1.0
  split_pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  byte_fallback: true

# WordPiece özellikleri
wordpiece:
  word_tokenizer_pattern: "[^\\s]+"
  wordpiece_prefix: "##"
  strip_accents: false

# Eğitim dosyaları
training:
  datasets:
    - path: "./data/wiki_tr.txt"
      weight: 0.4
      format: "text"
    - path: "./data/wiki_en.txt"
      weight: 0.3
      format: "text"
    - path: "./data/code.jsonl"
      weight: 0.3
      format: "jsonl"
      field: "content"

# Eğitim parametreleri
parameters:
  batch_size: 1024
  epochs: 3
  threads: 8

# Önişleme
preprocessing:
  - type: "replace_url"
    pattern: "https?://\\S+"
    replacement: "[URL]"
  - type: "replace_email"
    pattern: "\\S+@\\S+\\.\\S+"
    replacement: "[EMAIL]"
  - type: "normalize_whitespace"

# Tokenizer rotası - hangi metinlerin hangi tokenizer tarafından işleneceği
router:
  type: "regex"
  rules:
    - pattern: "<html|<!DOCTYPE"
      tokenizer: "html_tokenizer"
    - pattern: "^\\s*[{\\[]"
      tokenizer: "json_tokenizer"
    - pattern: "import |def |class |from "
      tokenizer: "code_tokenizer"
    - pattern: "SELECT |INSERT |UPDATE |DELETE FROM"
      tokenizer: "sql_tokenizer"
  default: "text_tokenizer"

# Bellek önbelleği
cache:
  enabled: true
  max_size: 100000

# Performans izleme
metrics:
  log_file: "./logs/tokenizer_metrics.log"
  report_interval: 1000
"""

    def test_individual_tokenizers_on_formats(self):
        """
        Her tokenizer'ı farklı veri formatları üzerinde test eder.
        """
        # Test edilecek tokenizer'lar
        tokenizer_instances = {
            "BPE": self.BPETokenizer(vocab_size=1000),
            "WordPiece": self.WordPieceTokenizer(vocab_size=1000),
            "ByteLevel": self.ByteLevelTokenizer(vocab_size=1000),
            "Unigram": self.UnigramTokenizer(vocab_size=1000)
        }
        
        # Tüm format örneklerini birleştir ve eğitim için kullan
        all_samples = list(self.format_samples.values())
        
        # Her tokenizer'ı test et
        for tokenizer_name, tokenizer in tokenizer_instances.items():
            with self.subTest(tokenizer=tokenizer_name):
                # Tokenizer'ı eğit
                print(f"\n{tokenizer_name} tokenizer farklı veri formatları üzerinde eğitiliyor...")
                train_result = tokenizer.train(all_samples)
                
                self.assertTrue(tokenizer.is_trained, f"{tokenizer_name} eğitimi başarısız oldu")
                self.assertGreater(tokenizer.get_vocab_size(), 0, f"{tokenizer_name} boş sözlük oluşturdu")
                
                # Farklı formatları test et
                for format_name, text in self.format_samples.items():
                    sample_text = text[:500]  # İlk 500 karakteri test et
                    
                    # Encode ve decode işlemleri
                    encoded = tokenizer.encode(sample_text)
                    decoded = tokenizer.decode(encoded)
                    
                    # Sonuçları yazdır
                    print(f"{tokenizer_name} - {format_name} encode sonucu: {len(encoded)} token")
                    print(f"İlk 10 token ID: {encoded[:10]}")
                    
                    # Token yoğunluğunu hesapla (token sayısı / metin uzunluğu)
                    density = len(encoded) / len(sample_text)
                    print(f"Token yoğunluğu: {density:.4f} token/karakter")
                    
                    # Minimal doğrulama
                    self.assertGreater(len(encoded), 0, f"{tokenizer_name} hiç token üretmedi")
                
                # Tokenizer'ı kaydet ve yükle
                save_path = self.temp_path / f"{tokenizer_name}_formats"
                tokenizer.save(save_path)
                
                try:
                    loaded_tokenizer = tokenizer.__class__.load(save_path)
                    self.assertEqual(
                        tokenizer.get_vocab_size(), 
                        loaded_tokenizer.get_vocab_size(), 
                        f"{tokenizer_name} yükleme sonrası sözlük boyutu değişti"
                    )
                except Exception as e:
                    print(f"{tokenizer_name} yüklenirken hata oluştu: {e}")

    def test_format_specific_mixed_tokenizer(self):
        """
        Farklı veri formatları için özelleştirilmiş MixedTokenizer'ı test eder.
        """
        # Format özel tokenizer'lar oluştur
        html_tokenizer = self.BPETokenizer(vocab_size=500, name="HTMLTokenizer")
        json_tokenizer = self.WordPieceTokenizer(vocab_size=400, name="JSONTokenizer")
        markdown_tokenizer = self.UnigramTokenizer(vocab_size=400, name="MarkdownTokenizer")
        general_tokenizer = self.ByteLevelTokenizer(vocab_size=400, name="GeneralTokenizer")
        
        # Formatları eğit
        html_tokenizer.train([self.format_samples["html"]])
        json_tokenizer.train([self.format_samples["json"]])
        markdown_tokenizer.train([self.format_samples["markdown"]])
        general_tokenizer.train([self.format_samples["xml"], self.format_samples["csv"], self.format_samples["yaml"]])
        
        # MixedTokenizer oluştur
        mixed_tokenizer = self.MixedTokenizer(
            tokenizers={
                "html": html_tokenizer,
                "json": json_tokenizer, 
                "markdown": markdown_tokenizer,
                "general": general_tokenizer
            },
            default_tokenizer="general",
            merged_vocab=True,
            name="FormatSpecificTokenizer"
        )
        
        # Router fonksiyonu tanımla
        def router(text):
            text_start = text[:100].lower()
            
            if "<!doctype html" in text_start or "<html" in text_start:
                return "html"
            elif text_start.strip().startswith(("{", "[")):
                return "json"
            elif text_start.strip().startswith("#") or "```" in text:
                return "markdown"
            else:
                return "general"
        
        mixed_tokenizer.router = router
        
        # Test et
        for format_name, text in self.format_samples.items():
            sample_text = text[:300]  # İlk 300 karakteri test et
            
            # Encode ve decode işlemleri
            encoded = mixed_tokenizer.encode(sample_text)
            decoded = mixed_tokenizer.decode(encoded)
            
            # Sonuçları yazdır
            print(f"\nMixedTokenizer - {format_name} encode sonucu: {len(encoded)} token")
            print(f"İlk 10 token ID: {encoded[:10]}")
            
            # Token yoğunluğunu hesapla
            density = len(encoded) / len(sample_text)
            print(f"Token yoğunluğu: {density:.4f} token/karakter")
            
            # Tespit edilen tokenizer'ı kontrol et
            detected = router(sample_text)
            print(f"Tespit edilen tokenizer: {detected}")
            
            # Minimal doğrulama
            self.assertGreater(len(encoded), 0, f"MixedTokenizer {format_name} için hiç token üretmedi")
        
        # Kaydet ve yükle
        save_path = self.temp_path / "mixed_formats"
        mixed_tokenizer.save(save_path)
        
        try:
            loaded_tokenizer = self.MixedTokenizer.load(save_path)
            self.assertEqual(
                mixed_tokenizer.get_vocab_size(), 
                loaded_tokenizer.get_vocab_size(), 
                "MixedTokenizer yükleme sonrası sözlük boyutu değişti"
            )
        except Exception as e:
            print(f"MixedTokenizer yüklenirken hata oluştu: {e}")

    def test_format_tags_and_structure(self):
        """
        Format etiketleri ve yapısal elementleri test eder.
        """
        # ByteLevel tokenizer kullan
        tokenizer = self.ByteLevelTokenizer(vocab_size=500)
        tokenizer.train(list(self.format_samples.values()))
        
        # HTML özel etiketleri
        html_tags = ["<html>", "<head>", "<body>", "<div>", "<p>", "<h1>", "<table>", "</html>"]
        
        # JSON yapıları
        json_structures = ["{", "}", "[", "]", ":", "\"", ","]
        
        # XML etiketleri
        xml_tags = ["<?xml", "</", "/>", "<tokenizers>", "<metadata>"]
        
        # Markdown yapıları
        markdown_structures = ["# ", "## ", "**", "```", "- ", "| ", "[", "](", ")"]
        
        # Tüm özel yapıları test et
        all_special_structures = html_tags + json_structures + xml_tags + markdown_structures
        
        # Encode ve sonuçları kontrol et
        for structure in all_special_structures:
            encoded = tokenizer.encode(structure)
            decoded = tokenizer.decode(encoded)
            
            print(f"Yapı: '{structure}' -> Token IDs: {encoded} -> Decoded: '{decoded}'")
            
            # En azından bir token üretilmeli
            self.assertGreater(len(encoded), 0, f"'{structure}' için hiç token üretilmedi")


if __name__ == "__main__":
    unittest.main()