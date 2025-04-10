#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tüm tokenizer tiplerini test eden kapsamlı test modülü.

Bu test modülü, projede bulunan tüm tokenizer tiplerini ve özel tokenizer'ları
gerçek Python dosyaları üzerinde test eder. BaseTokenizerTester sınıfını kullanarak
detaylı performans ve doğruluk ölçümleri yapar.
"""

import unittest
import os
import tempfile
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re
import random

from aiquantr_tokenizer.core.base_tokenizer_tester import BaseTokenizerTester
from aiquantr_tokenizer.tokenizers.base import BaseTokenizer
from aiquantr_tokenizer.processors.code.python import PythonProcessor
from aiquantr_tokenizer.processors.code.general import CodeProcessor


class TestAllTokenizersPython(unittest.TestCase):
    """
    Tüm tokenizer tiplerini kapsamlı şekilde test eden sınıf.
    
    Bu test sınıfı:
    1. Tüm tokenizer tiplerini ayrı ayrı test eder
    2. Google tarzı roundtrip metriklerini hesaplar
    3. İşlemcilerle (processor) tokenizer entegrasyonunu test eder
    4. MixedTokenizer kullanımını test eder
    5. TokenizerFactory API ve özel tokenizer'ları test eder
    6. Karşılaştırmalı performans ölçümleri yapar
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar, gerekli sınıfları yükler ve test verilerini oluşturur.
        """
        # Loglama yapılandırması
        self.logger = logging.getLogger("TestAllTokenizersPython")
        self.logger.setLevel(logging.INFO)
        
        # Daha önce eklenmemişse handler ekle
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(console_handler)
            
        self.logger.info("Test ortamı hazırlanıyor...")
        
        # Gerekli tokenizer sınıflarını içe aktar
        try:
            from aiquantr_tokenizer.tokenizers.bpe import BPETokenizer
            from aiquantr_tokenizer.tokenizers.wordpiece import WordPieceTokenizer
            from aiquantr_tokenizer.tokenizers.byte_level import ByteLevelTokenizer
            from aiquantr_tokenizer.tokenizers.unigram import UnigramTokenizer
            from aiquantr_tokenizer.tokenizers.sentencepiece import SentencePieceTokenizer
            from aiquantr_tokenizer.tokenizers.mixed import MixedTokenizer
            from aiquantr_tokenizer.tokenizers.factory import create_tokenizer_from_config, register_tokenizer_type
            
            self.tokenizer_classes = {
                "BPE": BPETokenizer,
                "WordPiece": WordPieceTokenizer,
                "ByteLevel": ByteLevelTokenizer, 
                "Unigram": UnigramTokenizer,
                "SentencePiece": SentencePieceTokenizer,
                "Mixed": MixedTokenizer,
            }
            
            self.create_tokenizer_from_config = create_tokenizer_from_config
            self.register_tokenizer_type = register_tokenizer_type
            
        except ImportError as e:
            self.logger.error(f"Gerekli tokenizer modülleri bulunamadı: {e}")
            self.skipTest(f"Gerekli tokenizer modülleri bulunamadı: {e}")
            
        # Proje kök dizinini bul
        current_path = Path(os.path.abspath(__file__))
        self.project_root = current_path.parent.parent.parent
        self.logger.info(f"Proje kök dizini: {self.project_root}")
        
        # Test edilecek Python dosyalarını belirle
        self.python_files = self._find_test_files()
        if not self.python_files:
            self.skipTest("Test edilecek Python dosyası bulunamadı.")
        
        # Geçici test dizini oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Dosya içeriklerini yükle
        self.file_contents = self._load_file_contents()
        
        # BaseTokenizerTester'ı yapılandır ve test verilerini yükle
        self.tokenizer_tester = self._setup_tokenizer_tester()
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        # Logger handler'larını kapat
        if hasattr(self, 'logger') and self.logger:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()
        
        # TokenizerTester logger'ını kapat
        if hasattr(self, 'tokenizer_tester') and self.tokenizer_tester:
            if hasattr(self.tokenizer_tester, 'logger') and self.tokenizer_tester.logger:
                for handler in self.tokenizer_tester.logger.handlers[:]:
                    self.tokenizer_tester.logger.removeHandler(handler)
                    handler.close()
        
        # Geçici dizini sil (ignoring errors flag'i ile)
        if hasattr(self, 'temp_dir') and self.temp_dir:
            try:
                self.temp_dir.cleanup()
            except (PermissionError, OSError, NotADirectoryError) as e:
                print(f"Uyarı: Geçici dizin temizlenirken hata oluştu: {e}")
            
        print("Test ortamı temizlendi.")
    
    def _find_test_files(self) -> Dict[str, Path]:
        """
        Test edilecek Python dosyalarını bulur.
        
        Returns:
            Dict[str, Path]: Dosya adı -> dosya yolu eşleşmeleri
        """
        test_files = {}
        
        # Ana modül dosyalarını kontrol et
        key_modules = {
            "config_manager": self.project_root / "aiquantr_tokenizer" / "config" / "config_manager.py",
            #"python_processor": self.project_root / "aiquantr_tokenizer" / "processors" / "code" / "python.py",
            #"base_tokenizer": self.project_root / "aiquantr_tokenizer" / "tokenizers" / "base.py",
            #"bpe_tokenizer": self.project_root / "aiquantr_tokenizer" / "tokenizers" / "bpe.py",
        }
        
        # Bulunan dosyaları ekle
        for name, path in key_modules.items():
            if path.exists():
                test_files[name] = path
                self.logger.info(f"Dosya bulundu: {name} -> {path}")
            else:
                self.logger.warning(f"Dosya bulunamadı: {name} -> {path}")
        
        # Yeterli dosya bulunmamışsa, projedeki Python dosyalarını ara
        if len(test_files) < 1:
            self.logger.info("Yeterli sayıda belirtilen dosya bulunamadı, proje içinde Python dosyaları aranıyor...")
            
            tokenizer_dir = self.project_root / "aiquantr_tokenizer"
            if tokenizer_dir.exists():
                # En fazla 5 dosya ekle
                count = 0
                for file_path in tokenizer_dir.glob("**/*.py"):
                    if count >= 5:
                        break
                    
                    # Varsa dosyayı ekle
                    if file_path.exists() and file_path.is_file() and file_path.name.endswith(".py"):
                        file_name = f"file_{count}"
                        test_files[file_name] = file_path
                        self.logger.info(f"Ek dosya bulundu: {file_name} -> {file_path}")
                        count += 1
        
        return test_files
    
    def _load_file_contents(self) -> Dict[str, str]:
        """
        Test dosyalarının içeriklerini yükler.
        
        Returns:
            Dict[str, str]: Dosya adı -> içerik eşleşmeleri
        """
        file_contents = {}
        
        for name, path in self.python_files.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_contents[name] = content
                    self.logger.info(f"'{name}' dosyası yüklendi: {len(content)} karakter")
            except Exception as e:
                self.logger.warning(f"'{name}' dosyası yüklenirken hata: {e}")
        
        return file_contents
    
    def _setup_tokenizer_tester(self) -> BaseTokenizerTester:
        """
        BaseTokenizerTester'ı yapılandırır ve test verilerini yükler.
        
        Returns:
            BaseTokenizerTester: Yapılandırılmış test edici
        """
        # Test sonuçları için klasör oluştur
        results_dir = self.temp_path / "test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # BaseTokenizerTester oluştur
        tester = BaseTokenizerTester(
            test_name="python_tokenizers",
            results_dir=str(results_dir),
            logging_level=logging.INFO
        )
        
        # Dosya handler'ı yerine StringIO handler kullan (dosya kilitleme sorunlarını önlemek için)
        if tester.logger.handlers:
            for handler in tester.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    tester.logger.removeHandler(handler)
                    
                    # StringIO tabanlı handler ekle
                    string_handler = logging.StreamHandler()
                    string_handler.setFormatter(logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    ))
                    tester.logger.addHandler(string_handler)
        
        # Test verilerini hazırla
        test_data = {
            "python_code": list(self.file_contents.values()),
            "python_small": [content[:500] for content in self.file_contents.values()],
            "python_large": [content for content in self.file_contents.values() if len(content) > 1000]
        }
        
        # 5 küçük örnek daha ekle
        test_data["python_snippets"] = [
            "def hello_world():\n    print('Hello, World!')\n\nhello_world()",
            "class MyClass:\n    def __init__(self, value):\n        self.value = value",
            "import os\nimport sys\nfrom pathlib import Path",
            "try:\n    x = 1/0\nexcept ZeroDivisionError as e:\n    print(f'Error: {e}')",
            "with open('file.txt', 'r') as f:\n    content = f.read()"
        ]
        
        # Test verilerini yükle
        tester.test_data = test_data
        
        return tester
    
    def _create_custom_tokenizer(self, vocab_size: int = 5000) -> BaseTokenizer:
        """
        Özel bir tokenizer oluşturur ve kayıt eder.
        
        Args:
            vocab_size: Kelime dağarcığı boyutu
            
        Returns:
            BaseTokenizer: Oluşturulan özel tokenizer
        """
        from aiquantr_tokenizer.tokenizers.base import BaseTokenizer, TokenizerTrainer
        import re
        from typing import List, Dict, Any, Optional, Union
        from pathlib import Path
        import json
        import os
        
        # Özel tokenizer sınıfını tanımla
        class CustomTokenizer(BaseTokenizer):
            """
            Örnek özel tokenizer implementasyonu.
            Bu tokenizer, basit bir regex tabanlı token ayırma kullanır.
            """
            
            def __init__(self, vocab_size: int = 5000, name: str = "CustomTokenizer"):
                """
                CustomTokenizer başlatıcı.
                
                Args:
                    vocab_size: Maksimum kelime dağarcığı boyutu
                    name: Tokenizer adı
                """
                super().__init__(vocab_size=vocab_size, name=name)
                self.vocab = {}  # token -> id sözlüğü
                self.inverse_vocab = {}  # id -> token sözlüğü
                self.special_tokens = {
                    "<unk>": 0,
                    "<pad>": 1, 
                    "<bos>": 2,
                    "<eos>": 3
                }
                self._is_trained = False
                
                # Python kodu için özel token kalıbı
                self.token_pattern = re.compile(
                    r'([A-Za-z][A-Za-z0-9_]*)|(\d+)|(\s+)|([^\w\s])'
                )
                
                # Özel tokenleri ekle
                for token, token_id in self.special_tokens.items():
                    self.vocab[token] = token_id
                    self.inverse_vocab[token_id] = token
            
            def train(self, texts: List[str]) -> Dict[str, Any]:
                """
                Tokenizer'ı verilen metinler üzerinde eğitir.
                
                Args:
                    texts: Eğitim metinleri listesi
                    
                Returns:
                    Dict[str, Any]: Eğitim istatistikleri
                """
                # Token frekanslarını hesapla
                token_freqs = {}
                total_tokens = 0
                
                # Her metin için tokenları bul ve frekansını artır
                for text in texts:
                    matches = self.token_pattern.findall(text)
                    for groups in matches:
                        token = next((group for group in groups if group), "")
                        if token:
                            token_freqs[token] = token_freqs.get(token, 0) + 1
                            total_tokens += 1
                
                # Frekansa göre sıralayıp en sık görünenleri al
                sorted_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
                
                # Özel token sayısının dışında kalan slotları doldur
                available_slots = self.vocab_size - len(self.special_tokens)
                top_tokens = sorted_tokens[:available_slots]
                
                # Token ID'leri ata
                next_id = len(self.special_tokens)
                for token, freq in top_tokens:
                    if token not in self.vocab:
                        self.vocab[token] = next_id
                        self.inverse_vocab[next_id] = token
                        next_id += 1
                
                self._is_trained = True
                
                # Eğitim istatistiklerini döndür
                stats = {
                    "vocab_size": len(self.vocab),
                    "training_tokens_count": total_tokens,
                    "unique_tokens_count": len(token_freqs),
                    "coverage_percentage": (len(self.vocab) - len(self.special_tokens)) / 
                                          max(1, len(token_freqs)) * 100
                }
                
                return stats
            
            def encode(self, text: str) -> List[int]:
                """
                Metni token ID'lerine dönüştürür.
                
                Args:
                    text: Tokenize edilecek metin
                    
                Returns:
                    List[int]: Token ID'leri listesi
                """
                if not self._is_trained:
                    raise ValueError("Tokenizer henüz eğitilmemiş. Önce train() metodunu çağırın.")
                
                # Metni parçalara ayır
                matches = self.token_pattern.findall(text)
                token_ids = []
                
                # Her eşleşmeyi işle
                for groups in matches:
                    token = next((group for group in groups if group), "")
                    if token:
                        # Kelime dağarcığında yoksa <unk> kullan
                        token_id = self.vocab.get(token, self.vocab.get("<unk>"))
                        token_ids.append(token_id)
                    
                return token_ids
            
            def decode(self, token_ids: List[int]) -> str:
                """
                Token ID'lerini metne dönüştürür.
                
                Args:
                    token_ids: Token ID'leri listesi
                    
                Returns:
                    str: Oluşturulan metin
                """
                if not self._is_trained:
                    raise ValueError("Tokenizer henüz eğitilmemiş. Önce train() metodunu çağırın.")
                
                # Token ID'lerini tokenlere dönüştür
                tokens = []
                for token_id in token_ids:
                    token = self.inverse_vocab.get(token_id, " ")
                    tokens.append(token)
                
                # Boşluk karakterlerini özel işle
                result = ""
                for token in tokens:
                    # Boşluksa, karakteri koru; değilse birleştir
                    if token.isspace():
                        result += token
                    else:
                        result += token
                
                return result
            
            def get_vocab(self) -> Dict[str, int]:
                """
                Tokenizer kelime dağarcığını döndürür.
                
                Returns:
                    Dict[str, int]: Token -> ID sözlüğü
                """
                return self.vocab
            
            def get_vocab_size(self) -> int:
                """
                Tokenizer kelime dağarcığı boyutunu döndürür.
                
                Returns:
                    int: Kelime dağarcığı boyutu
                """
                return len(self.vocab)
            
            @property
            def is_trained(self) -> bool:
                """
                Tokenizer'ın eğitilip eğitilmediğini döndürür.
                
                Returns:
                    bool: Eğitim durumu
                """
                return self._is_trained
            
            def save(self, save_dir: Union[str, Path]) -> None:
                """
                Tokenizer'ı belirtilen dizine kaydeder.
                
                Args:
                    save_dir: Kayıt dizini
                """
                save_dir = Path(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                
                # Model verilerini hazırla
                model_data = {
                    "vocab": self.vocab,
                    "special_tokens": self.special_tokens,
                    "vocab_size": self.vocab_size,
                    "name": self.name,
                    "is_trained": self._is_trained
                }
                
                # Sözlüğü JSON olarak kaydet
                vocab_file = save_dir / "vocab.json"
                with open(vocab_file, "w", encoding="utf-8") as f:
                    json.dump(model_data, f, ensure_ascii=False, indent=2)
                
                # Konfigürasyon dosyasını kaydet
                config_file = save_dir / "config.json"
                config = {
                    "tokenizer_type": "CustomTokenizer",
                    "vocab_size": self.vocab_size,
                    "name": self.name,
                    "version": "1.0.0"
                }
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
            
            @classmethod
            def load(cls, load_dir: Union[str, Path]) -> 'CustomTokenizer':
                """
                Kaydedilmiş bir tokenizer'ı yükler.
                
                Args:
                    load_dir: Yükleme dizini
                    
                Returns:
                    CustomTokenizer: Yüklenen tokenizer
                """
                load_dir = Path(load_dir)
                
                # Konfigürasyon dosyasını oku
                config_file = load_dir / "config.json"
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # Kelime dağarcığını oku
                vocab_file = load_dir / "vocab.json"
                with open(vocab_file, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
                
                # Yeni tokenizer oluştur
                tokenizer = cls(vocab_size=config["vocab_size"], name=config["name"])
                
                # Model verilerini ayarla
                tokenizer.vocab = {str(k): int(v) for k, v in model_data["vocab"].items()}
                tokenizer.special_tokens = model_data["special_tokens"]
                tokenizer._is_trained = model_data["is_trained"]
                
                # inverse_vocab'ı oluştur
                tokenizer.inverse_vocab = {int(v): str(k) for k, v in tokenizer.vocab.items()}
                
                return tokenizer
        
        # Özel tokenizer'ı kaydet
        self.register_tokenizer_type("custom", CustomTokenizer)
        
        # Örnek oluştur ve döndür
        return CustomTokenizer(vocab_size=vocab_size, name="CustomTokenizer")
    
    def test_individual_tokenizers(self):
        """
        Her tokenizer türünü ayrı ayrı test eder.
        """
        # Python işlemcisi oluştur
        processor = PythonProcessor(remove_comments=True, remove_docstrings=True)
        
        # Test edilecek tokenizer'ları hazırla
        test_tokenizers = {}
        for name, cls in self.tokenizer_classes.items():
            # SentencePiece ve Mixed şimdilik dışar bırak
            if name not in ["SentencePiece", "Mixed"]:
                try:
                    test_tokenizers[name] = cls(vocab_size=1000)
                    self.logger.info(f"{name} tokenizer oluşturuldu")
                except Exception as e:
                    self.logger.warning(f"{name} tokenizer oluşturulamadı: {e}")
        
        # Özel tokenizer ekle
        test_tokenizers["Custom"] = self._create_custom_tokenizer(vocab_size=1000)
        
        # İşlenecek kodları hazırla
        processed_codes = []
        for name, content in self.file_contents.items():
            processed_code = processor.process(content)
            processed_codes.append(processed_code)
        
        # Sonuçları raporlamak için bir tablo oluştur
        results_table = {
            "Tokenizer": [],
            "Vocab Size": [],
            "Encode Time (s)": [],
            "Tokens/Second": [],
            "Decode Time (s)": [],
            "Roundtrip Similarity (%)": []
        }
        
        # Her tokenizer için test et
        for name, tokenizer in test_tokenizers.items():
            with self.subTest(tokenizer=name):
                self.logger.info(f"\n{name} tokenizer test ediliyor...")
                
                # Tokenizer'ı eğit
                start_time = time.time()
                train_stats = tokenizer.train(processed_codes)
                train_time = time.time() - start_time
                
                self.assertTrue(tokenizer.is_trained, f"{name} eğitimi başarısız oldu")
                self.assertGreater(tokenizer.get_vocab_size(), 0, f"{name} boş kelime dağarcığı oluşturdu")
                
                self.logger.info(f"{name} eğitim tamamlandı: {train_time:.2f} saniye")
                self.logger.info(f"Kelime dağarcığı boyutu: {tokenizer.get_vocab_size()}")
                
                # BaseTokenizerTester'a kaydet
                self.tokenizer_tester.register_tokenizer(name, tokenizer)
                
                # Test metinlerini hazırla
                test_texts = []
                for file_name, content in self.file_contents.items():
                    # Her dosyadan bir örnek al
                    processed_code = processor.process(content)
                    # İlk 1000 karakteri ekle
                    test_texts.append(processed_code[:1000])
                
                # Encode ve decode süresini ölç
                total_encoded_tokens = 0
                total_encode_time = 0
                total_decode_time = 0
                total_similarity = 0
                
                for text in test_texts:
                    # Encode
                    encode_start = time.time()
                    encoded = tokenizer.encode(text)
                    encode_time = time.time() - encode_start
                    
                    total_encoded_tokens += len(encoded)
                    total_encode_time += encode_time
                    
                    # Decode
                    decode_start = time.time()
                    decoded = tokenizer.decode(encoded)
                    decode_time = time.time() - decode_start
                    
                    total_decode_time += decode_time
                    
                    # Benzerlik hesapla
                    similarity = self.tokenizer_tester.calculate_similarity(text, decoded)
                    total_similarity += similarity
                
                # Ortalama değerleri hesapla
                avg_encode_time = total_encode_time / len(test_texts)
                avg_decode_time = total_decode_time / len(test_texts)
                avg_similarity = total_similarity / len(test_texts)
                tokens_per_second = total_encoded_tokens / total_encode_time if total_encode_time > 0 else 0
                
                # Sonuçları yazdır
                self.logger.info(f"Ortalama encode süresi: {avg_encode_time:.5f} saniye")
                self.logger.info(f"Tokenizasyon hızı: {tokens_per_second:.1f} token/saniye")
                self.logger.info(f"Ortalama decode süresi: {avg_decode_time:.5f} saniye")
                self.logger.info(f"Ortalama roundtrip benzerlik: {avg_similarity:.2f}%")
                
                # Detaylı bir roundtrip analizi yap
                sample_text = test_texts[0]  # İlk metni analiz et
                analysis = self.tokenizer_tester.analyze_roundtrip_differences(tokenizer, sample_text)
                if analysis.get("problem_token_count", 0) > 0:
                    self.logger.info(f"Sorunlu token sayısı: {analysis.get('problem_token_count')}")
                    self.logger.info(f"Token hata oranı: {analysis.get('token_error_rate', 0):.4f}")
                
                # Tokenizer'ı kaydet ve yükle
                try:
                    save_path = self.temp_path / name
                    tokenizer.save(save_path)
                    
                    loaded_tokenizer = tokenizer.__class__.load(save_path)
                    self.assertEqual(
                        tokenizer.get_vocab_size(), 
                        loaded_tokenizer.get_vocab_size(),
                        f"{name} yüklemeden sonra kelime dağarcığı boyutu değişti"
                    )
                    self.logger.info(f"{name} başarıyla kaydedilip yüklendi")
                except Exception as e:
                    self.logger.warning(f"{name} kaydedilirken veya yüklenirken hata: {e}")
                
                # Sonuçları tabloya ekle
                results_table["Tokenizer"].append(name)
                results_table["Vocab Size"].append(tokenizer.get_vocab_size())
                results_table["Encode Time (s)"].append(f"{avg_encode_time:.5f}")
                results_table["Tokens/Second"].append(f"{tokens_per_second:.1f}")
                results_table["Decode Time (s)"].append(f"{avg_decode_time:.5f}")
                results_table["Roundtrip Similarity (%)"].append(f"{avg_similarity:.2f}")
        
        # Sonuçların özetini yazdır
        self._print_table(results_table)
    
    def test_mixed_tokenizer(self):
        """
        MixedTokenizer'ı test eder ve alt tokenizer davranışlarını doğrular.
        """
        self.logger.info("\nMixedTokenizer test ediliyor...")
        
        # Alt tokenizer'ları oluştur
        try:
            bpe_tokenizer = self.tokenizer_classes["BPE"](vocab_size=500)
            byte_tokenizer = self.tokenizer_classes["ByteLevel"](vocab_size=500)
        except (KeyError, AttributeError) as e:
            self.skipTest(f"MixedTokenizer testi için gereken alt tokenizer'lar bulunamadı: {e}")
        
        # Alt tokenizer'ları eğit - Farklı veri türleri için
        # Python kodu için BPE
        bpe_texts = list(self.file_contents.values())
        bpe_tokenizer.train(bpe_texts)
        self.logger.info(f"BPE alt tokenizer eğitildi: {bpe_tokenizer.get_vocab_size()} token")
        
        # JSON/yapılandırma için ByteLevel
        json_texts = []
        for i, name in enumerate(self.file_contents.keys()):
            # Örnek JSON içeriği oluştur
            sample = json.dumps({
                "id": i,
                "name": name,
                "type": "python",
                "length": len(self.file_contents[name]),
                "first_line": self.file_contents[name].split("\n")[0][:30]
            }, indent=2)
            json_texts.append(sample)
        
        byte_tokenizer.train(json_texts)
        self.logger.info(f"ByteLevel alt tokenizer eğitildi: {byte_tokenizer.get_vocab_size()} token")
        
        # MixedTokenizer oluştur
        try:
            mixed_tokenizer = self.tokenizer_classes["Mixed"](
                tokenizers={
                    "python": bpe_tokenizer, 
                    "json": byte_tokenizer
                },
                default_tokenizer="python"
            )
            
            # İçerik türüne göre tokenizer seçen router fonksiyonu
            def content_router(text):
                # Kaba bir kontrol - JSON için süslü parantez ara
                if text.strip().startswith("{") and (":" in text):
                    return "json"
                return "python"
            
            mixed_tokenizer.router = content_router
            
        except Exception as e:
            self.skipTest(f"MixedTokenizer oluşturulamadı: {e}")
        
        # BaseTokenizerTester'a kaydet
        self.tokenizer_tester.register_tokenizer("Mixed", mixed_tokenizer)
        
        # Her dosya türü için test et
        results = {
            "python": {"tokens": [], "similarity": []},
            "json": {"tokens": [], "similarity": []}
        }
        
        # Python kodunu test et
        for name, content in self.file_contents.items():
            test_python = content[:500]  # İlk 500 karakter
            try:
                python_tokens = mixed_tokenizer.encode(test_python)
                python_decoded = mixed_tokenizer.decode(python_tokens)
                python_similarity = self.tokenizer_tester.calculate_similarity(test_python, python_decoded)
                
                # Sonuçları kaydet
                results["python"]["tokens"].append(len(python_tokens))
                results["python"]["similarity"].append(python_similarity)
                
                # İlk dosya için detaylı çıktı
                if name == list(self.file_contents.keys())[0]:
                    self.logger.info(f"Python örneği: {len(python_tokens)} token")
                    self.logger.info(f"Seçilen tokenizer: python (BPE)")
                    self.logger.info(f"Benzerlik: {python_similarity:.2f}%")
            except Exception as e:
                self.logger.warning(f"Python testi başarısız: {e}")
        
        # JSON verilerini test et
        for i, json_text in enumerate(json_texts[:3]):  # Sadece ilk 3 JSON örneği
            try:
                json_tokens = mixed_tokenizer.encode(json_text)
                json_decoded = mixed_tokenizer.decode(json_tokens)
                json_similarity = self.tokenizer_tester.calculate_similarity(json_text, json_decoded)
                
                # Sonuçları kaydet
                results["json"]["tokens"].append(len(json_tokens))
                results["json"]["similarity"].append(json_similarity)
                
                # İlk JSON için detaylı çıktı
                if i == 0:
                    self.logger.info(f"JSON örneği: {len(json_tokens)} token")
                    self.logger.info(f"Seçilen tokenizer: json (ByteLevel)")
                    self.logger.info(f"Benzerlik: {json_similarity:.2f}%")
            except Exception as e:
                self.logger.warning(f"JSON testi başarısız: {e}")
        
        # Sonuçları özetle
        avg_python_tokens = sum(results["python"]["tokens"]) / max(1, len(results["python"]["tokens"]))
        avg_python_similarity = sum(results["python"]["similarity"]) / max(1, len(results["python"]["similarity"]))
        
        avg_json_tokens = sum(results["json"]["tokens"]) / max(1, len(results["json"]["tokens"]))
        avg_json_similarity = sum(results["json"]["similarity"]) / max(1, len(results["json"]["similarity"]))
        
        self.logger.info("\nMixedTokenizer sonuçları:")
        self.logger.info(f"Python: Ort. {avg_python_tokens:.1f} token, Ort. benzerlik: {avg_python_similarity:.2f}%")
        self.logger.info(f"JSON: Ort. {avg_json_tokens:.1f} token, Ort. benzerlik: {avg_json_similarity:.2f}%")
        
        # Roundtrip doğruluğunu kontrol et
        self.assertGreater(avg_python_similarity, 50.0, "Python için roundtrip benzerliği çok düşük")
        self.assertGreater(avg_json_similarity, 10.0, "JSON için roundtrip benzerliği çok düşük")
        
        # MixedTokenizer'ı kaydet ve yükle
        try:
            save_path = self.temp_path / "mixed"
            mixed_tokenizer.save(save_path)
            
            loaded_tokenizer = self.tokenizer_classes["Mixed"].load(save_path)
            self.logger.info("MixedTokenizer başarıyla kaydedilip yüklendi")
            
            # Router fonksiyonunu yeniden atamak gerekebilir
            loaded_tokenizer.router = content_router
            
            # Yükleme sonrası temel doğrulama
            test_python = self.file_contents[list(self.file_contents.keys())[0]][:100]
            test_json = json_texts[0][:100]
            
            encoded_python = loaded_tokenizer.encode(test_python)
            encoded_json = loaded_tokenizer.encode(test_json)
            
            self.assertGreater(len(encoded_python), 0, "Yüklenen MixedTokenizer Python kodunu tokenize edemedi")
            self.assertGreater(len(encoded_json), 0, "Yüklenen MixedTokenizer JSON verilerini tokenize edemedi")
            
        except Exception as e:
            self.logger.warning(f"MixedTokenizer kayıt/yükleme testi başarısız: {e}")
    
    def test_google_roundtrip_metrics(self):
        """
        Google tarzı roundtrip metrikleri hesaplar ve tokenizer'ların
        performansını bu metriklerle karşılaştırır.
        """
        self.logger.info("\nGoogle tarzı roundtrip metrikleri hesaplanıyor...")
        
        # Test edilecek tokenizer'ları hazırla (en yaygın 3 tür)
        tokenizer_instances = {}
        try:
            tokenizer_instances["BPE"] = self.tokenizer_classes["BPE"](vocab_size=1000)
            tokenizer_instances["WordPiece"] = self.tokenizer_classes["WordPiece"](vocab_size=1000)
            tokenizer_instances["ByteLevel"] = self.tokenizer_classes["ByteLevel"](vocab_size=1000)
        except (KeyError, Exception) as e:
            self.skipTest(f"Google roundtrip testi için gerekli tokenizer'lar oluşturulamadı: {e}")
        
        # İşlenmiş kodları hazırla
        processor = PythonProcessor(remove_comments=True, remove_docstrings=True)
        processed_codes = []
        for name, content in self.file_contents.items():
            processed_code = processor.process(content)
            processed_codes.append(processed_code)
        
        # Eğit ve BaseTokenizerTester'a kaydet
        for name, tokenizer in tokenizer_instances.items():
            tokenizer.train(processed_codes)
            self.tokenizer_tester.register_tokenizer(name, tokenizer)
            self.logger.info(f"{name} eğitildi ve test ediciye kaydedildi")
        
        # Tüm tokenizer'ları değerlendir
        results = self.tokenizer_tester.evaluate_tokenizers()
        
        # Özet tablosu oluştur
        summary_table = {
            "Tokenizer": [],
            "Tam Eşleşme (%)": [],
            "Karakter Kapsamı (%)": [],
            "Karakter Hata Oranı": []
        }
        
        # Sonuçları işle
        for name, data in results["tokenizers"].items():
            if "google_roundtrip" in data and "accuracy" in data["google_roundtrip"]:
                metrics = data["google_roundtrip"]["accuracy"]
                
                summary_table["Tokenizer"].append(name)
                summary_table["Tam Eşleşme (%)"].append(f"{metrics.get('exact_match_percentage', 0):.2f}")
                summary_table["Karakter Kapsamı (%)"].append(f"{metrics.get('character_coverage_percentage', 0):.2f}")
                summary_table["Karakter Hata Oranı"].append(f"{metrics.get('character_error_rate', 0):.4f}")
                
                # Sorunlu örnekleri kontrol et
                problem_count = data["google_roundtrip"].get("problem_examples_count", 0)
                if problem_count > 0:
                    self.logger.info(f"{name} için {problem_count} sorunlu örnek bulundu")
                    
                    # İlk problemi detaylı incele
                    if data["google_roundtrip"].get("problem_examples"):
                        problem = data["google_roundtrip"]["problem_examples"][0]
                        self.logger.info(f"Örnek problem analizi: {problem.get('analysis', {})}")
        
        # Sonuçları yazdır
        self._print_table(summary_table)
        
        # Grafikler oluştur
        output_dir = self.temp_path / "test_results"
        os.makedirs(output_dir, exist_ok=True)
        try:
            graph_files = self.tokenizer_tester.plot_results(str(output_dir))
            self.logger.info(f"{len(graph_files)} grafik oluşturuldu")
        except Exception as e:
            self.logger.warning(f"Grafikler oluşturulurken hata: {e}")
        
        # Karşılaştırmalı özet
        self.tokenizer_tester.print_summary()
        
        # Temel doğrulamalar
        for name in tokenizer_instances.keys():
            with self.subTest(tokenizer=name):
                self.assertIn(name, results["tokenizers"], f"{name} sonuçlarda bulunamadı")
                self.assertIn("google_roundtrip", results["tokenizers"][name], 
                             f"{name} için Google roundtrip metrikleri hesaplanmadı")
                
                accuracy = results["tokenizers"][name]["google_roundtrip"]["accuracy"]
                self.assertIn("character_coverage_percentage", accuracy, 
                             f"{name} için karakter kapsamı hesaplanmadı")
                
                # Tokenizer'ın makul bir karakter kapsamına sahip olduğunu kontrol et
                coverage = accuracy.get("character_coverage_percentage", 0)
                self.assertGreater(coverage, 20.0, f"{name} çok düşük karakter kapsamına sahip")
    
    def test_processor_with_tokenizers(self):
        """
        Farklı kod işlemcilerinin (processor) tokenizer'larla olan
        entegrasyonunu ve performansını test eder.
        """
        self.logger.info("\nİşlemciler ve tokenizer'lar birlikte test ediliyor...")
        
        # İşlemcileri oluştur
        processors = {
            "RawText": None,  # İşlemsiz ham metin
            "PythonBasic": PythonProcessor(
                remove_comments=True,
                remove_docstrings=True
            ),
            "PythonAdvanced": PythonProcessor(
                remove_comments=True,
                remove_docstrings=True,
                remove_type_hints=True,
                normalize_whitespace=True,
                remove_empty_lines=True
            ),
            "GeneralCode": CodeProcessor()
        }
        
        # Test için BPE tokenizer'ı kullan
        try:
            tokenizer = self.tokenizer_classes["BPE"](vocab_size=1000)
        except (KeyError, Exception) as e:
            self.skipTest(f"İşlemci testi için BPE tokenizer oluşturulamadı: {e}")
        
        # Test sonuçları
        processor_results = {
            "Processor": [],
            "Tokens/Sample": [],
            "Chars/Token": [],
            "Benzerlik (%)": []
        }
        
        # Her işlemci için test et
        for proc_name, processor in processors.items():
            with self.subTest(processor=proc_name):
                # İşlenmiş kodları hazırla
                processed_texts = []
                for name, content in self.file_contents.items():
                    if processor:
                        processed_text = processor.process(content)
                    else:
                        processed_text = content  # Ham metin
                    processed_texts.append(processed_text[:1000])  # İlk 1000 karakter
                
                # Tokenizer'ı eğit
                tokenizer.train(processed_texts)
                
                # Test metrikleri
                total_tokens = 0
                total_chars = 0
                total_similarity = 0
                
                # Her metin için test et
                for text in processed_texts:
                    # Encode ve decode
                    try:
                        tokens = tokenizer.encode(text)
                        decoded = tokenizer.decode(tokens)
                        
                        # Metrikleri hesapla
                        total_tokens += len(tokens)
                        total_chars += len(text)
                        
                        # Benzerlik
                        similarity = self.tokenizer_tester.calculate_similarity(text, decoded)
                        total_similarity += similarity
                        
                    except Exception as e:
                        self.logger.warning(f"{proc_name} işlemci testi sırasında hata: {e}")
                
                # Ortalama değerleri hesapla
                avg_tokens = total_tokens / len(processed_texts) if processed_texts else 0
                avg_chars_per_token = total_chars / total_tokens if total_tokens else 0
                avg_similarity = total_similarity / len(processed_texts) if processed_texts else 0
                
                # Bilgileri ekle
                processor_results["Processor"].append(proc_name)
                processor_results["Tokens/Sample"].append(f"{avg_tokens:.1f}")
                processor_results["Chars/Token"].append(f"{avg_chars_per_token:.2f}")
                processor_results["Benzerlik (%)"].append(f"{avg_similarity:.2f}")
                
                self.logger.info(f"{proc_name} işlemcisi test edildi:")
                self.logger.info(f"  Ortalama token sayısı: {avg_tokens:.1f}")
                self.logger.info(f"  Karakter/Token: {avg_chars_per_token:.2f}")
                self.logger.info(f"  Roundtrip benzerliği: {avg_similarity:.2f}%")
                
                # İşlemcinin düzgün çalıştığını doğrula
                if processor and proc_name == "PythonAdvanced":
                    sample = processed_texts[0]
                    self.assertNotIn('"""', sample, "Docstring kaldırma başarısız")
                    self.assertNotIn('#', sample, "Yorum kaldırma başarısız") 
                    self.assertNotIn(': int', sample, "Tip işareti kaldırma başarısız")
        
        # Sonuçları yazdır
        self._print_table(processor_results)
        
        # İşlemciler arası karşılaştırma
        self.logger.info("\nİşlemci Karşılaştırması:")
        raw_index = processor_results["Processor"].index("RawText") if "RawText" in processor_results["Processor"] else -1
        advanced_index = processor_results["Processor"].index("PythonAdvanced") if "PythonAdvanced" in processor_results["Processor"] else -1
        
        if raw_index >= 0 and advanced_index >= 0:
            raw_tokens = float(processor_results["Tokens/Sample"][raw_index].replace(',', ''))
            adv_tokens = float(processor_results["Tokens/Sample"][advanced_index].replace(',', ''))
            
            token_reduction = ((raw_tokens - adv_tokens) / raw_tokens) * 100 if raw_tokens > 0 else 0
            self.logger.info(f"İşleme ile token azalma oranı: {token_reduction:.1f}%")
            
            # İşlemenin token ekonomisini artırdığını doğrula
            self.assertGreater(token_reduction, -10.0, 
                              "İşleme sonrası token sayısında beklenmedik artış var")
    
    def test_factory_custom_tokenizer(self):
        """
        TokenizerFactory API'sini ve özel tokenizer tiplerini test eder.
        """
        self.logger.info("\nTokenizerFactory API ve özel tokenizer test ediliyor...")
        
        # Özel tokenizer tipini kaydet ve örnek oluştur
        try:
            custom_tokenizer = self._create_custom_tokenizer(vocab_size=800)
            self.logger.info("Özel tokenizer oluşturuldu ve kaydedildi")
        except Exception as e:
            self.skipTest(f"Özel tokenizer oluşturulamadı: {e}")
        
        # Fabrika (factory) API'si ile başka bir örnek oluştur
        try:
            config = {
                "type": "custom",  # _create_custom_tokenizer'ın kaydettiği tip
                "vocab_size": 1000,
                "name": "FactoryCustomTokenizer"
            }
            
            factory_tokenizer = self.create_tokenizer_from_config(config)
            self.logger.info("Factory API ile ikinci özel tokenizer oluşturuldu")
        except Exception as e:
            self.skipTest(f"Factory API tokenizer oluşturulamadı: {e}")
        
        # İki tokenizer'ın farklı olduğunu doğrula
        self.assertIsNot(custom_tokenizer, factory_tokenizer, 
                       "Factory API aynı örneği döndürdü, yeni örnek bekleniyor")
        
        # Her iki tokenizer'ı da eğit
        sample_texts = list(self.file_contents.values())
        
        custom_tokenizer.train(sample_texts)
        factory_tokenizer.train(sample_texts)
        
        # BaseTokenizerTester'a kaydet
        self.tokenizer_tester.register_tokenizer("CustomDirect", custom_tokenizer)
        self.tokenizer_tester.register_tokenizer("CustomFactory", factory_tokenizer)
        
        # Karşılaştırmalı test
        sample = sample_texts[0][:500]  # İlk dosyanın ilk 500 karakteri
        
        # Her iki tokenizer için encode/decode performansını ölç
        for name, tokenizer in [("CustomDirect", custom_tokenizer), ("CustomFactory", factory_tokenizer)]:
            try:
                start_time = time.time()
                encoded = tokenizer.encode(sample)
                encode_time = time.time() - start_time
                
                start_time = time.time()
                decoded = tokenizer.decode(encoded)
                decode_time = time.time() - start_time
                
                # Benzerlik hesapla
                similarity = self.tokenizer_tester.calculate_similarity(sample, decoded)
                
                # Sonuçları yaz
                self.logger.info(f"{name} test sonuçları:")
                self.logger.info(f"  Token sayısı: {len(encoded)}")
                self.logger.info(f"  Encode süresi: {encode_time:.5f} saniye")
                self.logger.info(f"  Decode süresi: {decode_time:.5f} saniye")
                self.logger.info(f"  Benzerlik: {similarity:.2f}%")
                
                # Minimal doğrulama
                self.assertGreater(len(encoded), 0, f"{name} hiç token üretmedi")
                self.assertGreater(similarity, 30.0, f"{name} çok düşük benzerlik oranı")
                
            except Exception as e:
                self.logger.warning(f"{name} test sırasında hata: {e}")
        
        # Kaydetme ve yüklemeyi test et
        save_path = self.temp_path / "custom_factory"
        factory_tokenizer.save(save_path)
        
        try:
            # Doğrudan sınıf metodu ile yükle
            loaded_tokenizer = factory_tokenizer.__class__.load(save_path)
            
            # Temel doğrulama
            self.assertEqual(
                factory_tokenizer.get_vocab_size(), 
                loaded_tokenizer.get_vocab_size(),
                "Yükleme sonrası kelime dağarcığı boyutu değişti"
            )
            
            # Encode/decode işlevselliğini doğrula
            test_sample = "def test_function():\n    return 42"
            encoded = loaded_tokenizer.encode(test_sample)
            decoded = loaded_tokenizer.decode(encoded)
            
            similarity = self.tokenizer_tester.calculate_similarity(test_sample, decoded)
            self.logger.info(f"Yüklenen tokenizer'ın benzerlik oranı: {similarity:.2f}%")
            
            self.assertGreater(similarity, 30.0, "Yüklenen tokenizer düşük benzerlik oranına sahip")
            
        except Exception as e:
            self.logger.warning(f"Tokenizer yükleme testi başarısız: {e}")
    
    def test_tokenizer_performance_comparison(self):
        """
        Farklı tokenizer'ların performans karşılaştırmasını yapar.
        """
        self.logger.info("\nTokenizer performans karşılaştırması yapılıyor...")
        
        # Test için tokenizer'lar
        try:
            tokenizers = [
                ("BPE", self.tokenizer_classes["BPE"](vocab_size=1000)),
                ("WordPiece", self.tokenizer_classes["WordPiece"](vocab_size=1000)),
                ("ByteLevel", self.tokenizer_classes["ByteLevel"](vocab_size=1000)),
                ("Custom", self._create_custom_tokenizer(vocab_size=1000))
            ]
        except (KeyError, Exception) as e:
            self.skipTest(f"Performans testi için tokenizer'lar oluşturulamadı: {e}")
        
        # Test verisini hazırla 
        benchmark_texts = []
        
        # Her dosyadan farklı uzunluklarda metinler ekle
        for content in self.file_contents.values():
            # Küçük örnek (ilk 200 karakter)
            benchmark_texts.append(content[:200])
            
            # Orta boy örnek (1000 karakter)
            if len(content) > 1000:
                benchmark_texts.append(content[:1000])
            
            # Büyük örnek (tam dosya, en fazla 10.000 karakter)
            benchmark_texts.append(content[:10000])
        
        # Eğer örnek sayısı çok fazlaysa sınırla
        if len(benchmark_texts) > 15:
            benchmark_texts = random.sample(benchmark_texts, 15)
        
        # Karşılaştırma sonuçları
        performance_results = {
            "Tokenizer": [],
            "Ort. Encode Süresi (s)": [],
            "Token/Saniye": [],
            "Ort. Karakter/Token": [],
            "Bellek Kullanımı (KB)": [],
            "Benzerlik (%)": []
        }
        
        # Her tokenizer için
        for name, tokenizer in tokenizers:
            self.logger.info(f"{name} tokenizer'ı test ediliyor...")
            
            # BaseTokenizerTester'a kaydet
            self.tokenizer_tester.register_tokenizer(name, tokenizer)
            
            # Tokenizer'ı eğit
            tokenizer.train(benchmark_texts)
            
            # Performans testi yap
            total_tokens = 0
            total_chars = 0
            total_encode_time = 0
            total_similarity = 0
            
            for text in benchmark_texts:
                # Encode performansı
                start_time = time.time()
                tokens = tokenizer.encode(text)
                encode_time = time.time() - start_time
                
                # Metrikleri güncelle
                total_tokens += len(tokens)
                total_chars += len(text)
                total_encode_time += encode_time
                
                # Decode ve benzerlik
                decoded = tokenizer.decode(tokens)
                similarity = self.tokenizer_tester.calculate_similarity(text, decoded)
                total_similarity += similarity
            
            # Ortalama değerleri hesapla
            avg_encode_time = total_encode_time / len(benchmark_texts)
            tokens_per_second = total_tokens / total_encode_time if total_encode_time > 0 else 0
            avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
            avg_similarity = total_similarity / len(benchmark_texts)
            
            # Bellek kullanımını tahmin et (kelime dağarcığı boyutuna göre)
            vocab_size = tokenizer.get_vocab_size()
            avg_token_length = 5  # Ortalama token uzunluğu tahmini
            memory_usage = (vocab_size * avg_token_length * 2) / 1024  # KB cinsinden
            
            # Sonuçları ekle
            performance_results["Tokenizer"].append(name)
            performance_results["Ort. Encode Süresi (s)"].append(f"{avg_encode_time:.6f}")
            performance_results["Token/Saniye"].append(f"{tokens_per_second:.1f}")
            performance_results["Ort. Karakter/Token"].append(f"{avg_chars_per_token:.2f}")
            performance_results["Bellek Kullanımı (KB)"].append(f"{memory_usage:.1f}")
            performance_results["Benzerlik (%)"].append(f"{avg_similarity:.2f}")
            
            self.logger.info(f"{name} sonuçları:")
            self.logger.info(f"  Ortalama encode süresi: {avg_encode_time:.6f} saniye")
            self.logger.info(f"  Tokenizasyon hızı: {tokens_per_second:.1f} token/saniye")
            self.logger.info(f"  Karakter/Token: {avg_chars_per_token:.2f}")
            self.logger.info(f"  Benzerlik: {avg_similarity:.2f}%")
            
            # Temel doğrulama
            self.assertGreater(tokens_per_second, 0, f"{name} tokenizasyon hızı ölçülemedi")
        
        # Sonuçları yazdır
        self._print_table(performance_results)
        
        # BaseTokenizerTester ile karşılaştırmalı değerlendirme
        results = self.tokenizer_tester.evaluate_tokenizers()
        
        # En iyi/en kötü performans gösteren tokenizer'ları belirle
        if "comparative" in results:
            self.logger.info("\nKarşılaştırmalı Sonuçlar:")
            
            for metric_name, metric_data in results["comparative"].items():
                if isinstance(metric_data, dict) and "best" in metric_data and "worst" in metric_data:
                    best_name = metric_data["best"]
                    worst_name = metric_data["worst"]
                    best_value = metric_data["values"][best_name]
                    worst_value = metric_data["values"][worst_name]
                    
                    # Metriğe göre açıklama
                    if metric_name == "chars_per_token":
                        metric_desc = "Token Ekonomisi (karakter/token)"
                        compare = "düşük" if best_value < worst_value else "yüksek"
                    elif metric_name == "tokens_per_second":
                        metric_desc = "Tokenizasyon Hızı"
                        compare = "yüksek" if best_value > worst_value else "düşük"
                    elif metric_name == "roundtrip_accuracy" or metric_name == "character_coverage":
                        metric_desc = "Roundtrip Doğruluğu"
                        compare = "yüksek" if best_value > worst_value else "düşük"
                    else:
                        metric_desc = metric_name
                        compare = "optimal" 
                    
                    self.logger.info(f"{metric_desc}: En iyi {best_name} ({best_value:.2f}), "
                                    f"En kötü {worst_name} ({worst_value:.2f})")
                    
    def _print_table(self, table_data: Dict[str, List[str]], max_width: int = 15) -> None:
        """
        Sözlük tabanlı verileri formatlı bir tablo olarak yazdırır.
        
        Args:
            table_data: Sütun adları ve değerleri içeren sözlük
            max_width: Maksimum sütun genişliği
        """
        if not table_data or not all(len(column) == len(list(table_data.values())[0]) 
                                    for column in table_data.values()):
            self.logger.warning("Geçersiz tablo verisi")
            return
        
        # Sütun genişliklerini belirle
        col_names = list(table_data.keys())
        widths = {}
        
        for col_name in col_names:
            col_values = table_data[col_name]
            max_len = max(len(str(col_name)), max(len(str(val)) for val in col_values))
            widths[col_name] = min(max_len + 2, max_width)
        
        # Başlık satırını yazdır
        header = "| "
        for col_name in col_names:
            header += f"{col_name:{widths[col_name]}} | "
        print("\n" + header)
        
        # Ayırıcı çizgiyi yazdır
        separator = "+-"
        for col_name in col_names:
            separator += "-" * widths[col_name] + "-+-"
        print(separator)
        
        # Veri satırlarını yazdır
        row_count = len(list(table_data.values())[0])
        for i in range(row_count):
            row = "| "
            for col_name in col_names:
                value = table_data[col_name][i]
                row += f"{value:{widths[col_name]}} | "
            print(row)
        
        # Alt çizgiyi yazdır
        print(separator + "\n")


if __name__ == "__main__":
    # Log seviyesini ayarla
    logging.basicConfig(level=logging.INFO, 
                        format='%(levelname)s: %(message)s')
    
    unittest.main()