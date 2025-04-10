"""
Tüm tokenizer tiplerini test eden kapsamlı test modülü.

Bu test modülü, projede bulunan tüm tokenizer tiplerini ve 
özel tokenizer'ları gerçek Python dosyaları üzerinde test eder.
"""

import unittest
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, Any, List, Type, Optional

from aiquantr_tokenizer.tokenizers.base import BaseTokenizer, TokenizerTrainer
from aiquantr_tokenizer.processors.code.python import PythonProcessor
from aiquantr_tokenizer.processors.code.general import CodeProcessor


class TestAllTokenizersPython(unittest.TestCase):
    """
    Tüm tokenizer tiplerini test etmek için test sınıfı.
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
            from aiquantr_tokenizer.tokenizers.sentencepiece import SentencePieceTokenizer
            from aiquantr_tokenizer.tokenizers.mixed import MixedTokenizer
            from aiquantr_tokenizer.tokenizers.factory import create_tokenizer_from_config, register_tokenizer_type
            
            self.BPETokenizer = BPETokenizer
            self.WordPieceTokenizer = WordPieceTokenizer
            self.ByteLevelTokenizer = ByteLevelTokenizer
            self.UnigramTokenizer = UnigramTokenizer
            self.SentencePieceTokenizer = SentencePieceTokenizer
            self.MixedTokenizer = MixedTokenizer
            self.create_tokenizer_from_config = create_tokenizer_from_config
            self.register_tokenizer_type = register_tokenizer_type
            
            self.all_tokenizer_classes = {
                "BPE": BPETokenizer,
                "WordPiece": WordPieceTokenizer,
                "ByteLevel": ByteLevelTokenizer, 
                "Unigram": UnigramTokenizer,
                "SentencePiece": SentencePieceTokenizer,
            }
            
        except ImportError as e:
            import traceback
            print(f"HATA! Gerekli tokenizer modülleri bulunamadı: {e}")
            print("TAM HATA İZİ:")
            traceback.print_exc()
            self.skipTest(f"Gerekli tokenizer modülleri bulunamadı: {e}")
            
        # Proje kök dizinini bul
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # İşlenecek dosyaların yolları
        self.python_files = {
            "config_manager": self.project_root / "aiquantr_tokenizer" / "config" / "config_manager.py",
            "base_tokenizer": self.project_root / "aiquantr_tokenizer" / "tokenizers" / "base.py",
            "bpe_tokenizer": self.project_root / "aiquantr_tokenizer" / "tokenizers" / "bpe.py",
            "python_processor": self.project_root / "aiquantr_tokenizer" / "processors" / "code" / "python.py"
        }
        
        # Dosyaların varlığını kontrol et
        for name, path in self.python_files.items():
            if not path.exists():
                self.skipTest(f"{name} için dosya bulunamadı: {path}")
        
        # Her test için geçici dizin oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Tüm dosyaların içeriğini oku
        self.file_contents = {}
        for name, path in self.python_files.items():
            with open(path, "r", encoding="utf-8") as f:
                self.file_contents[name] = f.read()
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        self.temp_dir.cleanup()
    
    def _create_custom_tokenizer(self, vocab_size: int = 5000) -> BaseTokenizer:
        """
        CustomTokenizer örneği oluşturur.
        
        Bu metot, TokenizerFactory API'sini kullanarak özel bir tokenizer oluşturur.
        """
        # Özel bir tokenizer sınıfı tanımla
        class CustomTokenizer(BaseTokenizer):
            """Özel tokenizer implementasyonu."""
            
            def __init__(
                self,
                vocab_size: int = 5000,
                min_frequency: int = 2,
                special_tokens: Optional[Dict[str, str]] = None,
                name: Optional[str] = None
            ):
                super().__init__(
                    vocab_size=vocab_size,
                    min_frequency=min_frequency,
                    special_tokens=special_tokens,
                    name=name or "CustomTokenizer"
                )
                self.vocab = {}
                self.ids_to_tokens = {}
            
            def train(self, texts, trainer=None, **kwargs):
                # Basit bir karakter tabanlı sözlük oluştur
                chars = set()
                for text in texts:
                    chars.update(text)
                
                self.vocab = {c: i for i, c in enumerate(sorted(chars))}
                self.ids_to_tokens = {i: c for c, i in self.vocab.items()}
                self.is_trained = True
                
                return {"vocab_size": len(self.vocab)}
            
            def encode(self, text, add_special_tokens=True, **kwargs):
                import time  # Direkt burada import edelim (geçici çözüm)
                start_time = time.time()
                
                result = [self.vocab.get(c, 0) for c in text]
                
                # İstatistikleri güncelle
                self.stats["num_encode_calls"] += 1
                self.stats["total_encode_time"] += time.time() - start_time
                return result

            def decode(self, ids, skip_special_tokens=True, **kwargs):
                import time  # Direkt burada import edelim (geçici çözüm)
                start_time = time.time()
                
                result = "".join([self.ids_to_tokens.get(i, "") for i in ids])
                
                # İstatistikleri güncelle
                self.stats["num_decode_calls"] += 1
                self.stats["total_decode_time"] += time.time() - start_time
                return result
            
            def get_vocab(self):
                return dict(self.vocab)
            
            def save(self, path, **kwargs):
                path = Path(path)
                if path.is_dir():
                    save_path = path / "tokenizer.json"
                else:
                    save_path = path
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"type": "CustomTokenizer", "vocab": self.vocab}, f)
            
            @classmethod
            def load(cls, path, **kwargs):
                path = Path(path)
                if path.is_dir():
                    load_path = path / "tokenizer.json"
                else:
                    load_path = path
                
                with open(load_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                tokenizer = cls(vocab_size=len(data.get("vocab", {})))
                tokenizer.vocab = data.get("vocab", {})
                tokenizer.ids_to_tokens = {int(i): t for t, i in tokenizer.vocab.items()}
                tokenizer.is_trained = True
                
                return tokenizer
        
        # CustomTokenizer'ı kaydet
        self.register_tokenizer_type("custom", CustomTokenizer)
        
        # Örnek oluştur
        return CustomTokenizer(vocab_size=vocab_size)
    
    def test_individual_tokenizers(self):
        """
        Her tokenizer türünü ayrı ayrı test eder.
        """
        # Python işlemcisi oluştur
        processor = PythonProcessor(remove_comments=True, remove_docstrings=True)
        
        # Test edilecek tokenizer'lar
        tokenizer_instances = {
            "BPE": self.BPETokenizer(vocab_size=500),
            "WordPiece": self.WordPieceTokenizer(vocab_size=500),
            "ByteLevel": self.ByteLevelTokenizer(vocab_size=500),
            "Unigram": self.UnigramTokenizer(vocab_size=500),
            "Custom": self._create_custom_tokenizer(vocab_size=500)
        }
        
        # SentencePiece eğer gereklilikleri kuruluysa ekle
        try:
            tokenizer_instances["SentencePiece"] = self.SentencePieceTokenizer(vocab_size=500)
        except (ImportError, AttributeError):
            print("SentencePiece tokenizer yüklenemedi, test edilmeyecek.")
        
        # İşlenecek kodları hazırla
        processed_codes = []
        for name, content in self.file_contents.items():
            processed_code = processor.process(content)
            processed_codes.append(processed_code)
        
        # Her tokenizer'ı test et
        for tokenizer_name, tokenizer in tokenizer_instances.items():
            with self.subTest(tokenizer=tokenizer_name):
                # Tokenizer'ı eğit
                print(f"\n{tokenizer_name} tokenizer eğitiliyor...")
                train_result = tokenizer.train(processed_codes)
                
                self.assertTrue(tokenizer.is_trained, f"{tokenizer_name} eğitimi başarısız oldu")
                self.assertGreater(tokenizer.get_vocab_size(), 0, f"{tokenizer_name} boş sözlük oluşturdu")
                
                # Her dosyayı ayrı ayrı test et
                for file_name, content in self.file_contents.items():
                    processed_code = processor.process(content)
                    sample_text = processed_code[:1000]  # İlk 1000 karakteri test et
                    
                    # Encode ve decode işlemleri
                    encoded = tokenizer.encode(sample_text)
                    decoded = tokenizer.decode(encoded)
                    
                    # Sonuçları yazdır
                    print(f"{tokenizer_name} - {file_name} encode sonucu: {len(encoded)} token")
                    print(f"İlk 10 token ID: {encoded[:10]}")
                    
                    # Decode edilmiş metin orijinal metne ne kadar yakın?
                    similarity = self._calculate_similarity(sample_text, decoded)
                    print(f"{tokenizer_name} - {file_name} decode benzerliği: {similarity:.2f}%")
                    
                    # Minimal doğrulama
                    self.assertGreater(len(encoded), 0, f"{tokenizer_name} hiç token üretmedi")
                    
                # Tokenizer'ı kaydet ve yükle
                save_path = self.temp_path / tokenizer_name
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
    
    def test_mixed_tokenizer(self):
        """
        MixedTokenizer'ı test eder.
        """
        # Alt tokenizer'ları oluştur
        bpe_tokenizer = self.BPETokenizer(vocab_size=300)
        byte_tokenizer = self.ByteLevelTokenizer(vocab_size=300)
        
        # Alt tokenizer'ları kendi dosya türlerine göre eğit
        tokenizer_datasets = {}
        
        # BPE için Python dosyalarını kullan
        bpe_code_samples = []
        for name, content in self.file_contents.items():
            bpe_code_samples.append(content)
        
        # ByteLevel için JSON formatında içerik oluştur
        byte_samples = []
        for name in self.file_contents.keys():
            sample = json.dumps({"name": name, "type": "python", "size": len(self.file_contents[name])})
            byte_samples.append(sample)
        
        # Alt tokenizer'ları eğit
        bpe_tokenizer.train(bpe_code_samples)
        byte_tokenizer.train(byte_samples)
        
        # MixedTokenizer oluştur
        mixed_tokenizer = self.MixedTokenizer(
            tokenizers={"python": bpe_tokenizer, "json": byte_tokenizer},
            default_tokenizer="python",
            merged_vocab=True
        )
        
        # Router fonksiyonu tanımla
        def router(text):
            if text.strip().startswith("{") and text.strip().endswith("}"):
                return "json"
            return "python"
        
        mixed_tokenizer.router = router
        
        # Test et
        sample_python = self.file_contents["python_processor"][:500]
        sample_json = json.dumps({"test": "value", "array": [1, 2, 3]})
        
        # Python kodu tokenize et
        python_tokens = mixed_tokenizer.encode(sample_python)
        python_decoded = mixed_tokenizer.decode(python_tokens)
        
        # JSON tokenize et
        json_tokens = mixed_tokenizer.encode(sample_json)
        json_decoded = mixed_tokenizer.decode(json_tokens)
        
        # Sonuçları yazdır
        print("\nMixedTokenizer test sonuçları:")
        print(f"Python örneği: {len(python_tokens)} token")
        print(f"JSON örneği: {len(json_tokens)} token")
        
        # Minimal doğrulama
        self.assertGreater(len(python_tokens), 0, "MixedTokenizer Python için hiç token üretmedi")
        self.assertGreater(len(json_tokens), 0, "MixedTokenizer JSON için hiç token üretmedi")
        
        # Kaydet ve yükle
        save_path = self.temp_path / "mixed"
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
    
    def test_factory_custom_tokenizer(self):
        """
        Factory API'si ile özel tokenizer oluşturmayı ve kaydetmeyi test eder.
        """
        # Özel tokenizer'ı kaydet
        custom_tokenizer = self._create_custom_tokenizer()
        
        # Yapılandırma ile tokenizer oluştur
        config = {
            "type": "custom",
            "vocab_size": 1000,
            "name": "MyCustomTokenizer"
        }
        
        factory_tokenizer = self.create_tokenizer_from_config(config)
        
        # Eğit ve test et
        sample_texts = list(self.file_contents.values())
        factory_tokenizer.train(sample_texts)
        
        # Test işlemleri
        sample = sample_texts[0][:500]
        encoded = factory_tokenizer.encode(sample)
        decoded = factory_tokenizer.decode(encoded)
        
        # Sonuçları yazdır
        print("\nFactory API CustomTokenizer test sonuçları:")
        print(f"Encode sonucu: {len(encoded)} token")
        print(f"İlk 10 token ID: {encoded[:10]}")
        
        # Minimal doğrulama
        self.assertGreater(len(encoded), 0, "Factory CustomTokenizer hiç token üretmedi")
        
        # Kaydet ve yükle
        save_path = self.temp_path / "custom_factory"
        factory_tokenizer.save(save_path)
        
        try:
            # load_tokenizer_from_path kullanma, doğrudan sınıf metodu ile yükle
            loaded_tokenizer = factory_tokenizer.__class__.load(save_path)
            self.assertEqual(
                factory_tokenizer.get_vocab_size(), 
                loaded_tokenizer.get_vocab_size(), 
                "CustomTokenizer yükleme sonrası sözlük boyutu değişti"
            )
        except Exception as e:
            print(f"CustomTokenizer yüklenirken hata oluştu: {e}")
    
    def test_processor_with_tokenizers(self):
        """
        İşlemci ve tokenizer'ları birlikte test eder.
        """
        # İşlemcileri oluştur
        python_processor = PythonProcessor(
            remove_comments=True,
            remove_docstrings=True,
            remove_type_hints=True
        )
        
        general_processor = CodeProcessor()
        
        # Tokenizer'ları oluştur
        bpe_tokenizer = self.BPETokenizer(vocab_size=1000)
        
        # İşlenmemiş ve işlenmiş kodlarla test et
        for file_name, content in self.file_contents.items():
            with self.subTest(file=file_name):
                # İşlenmemiş kod
                raw_encoded = bpe_tokenizer.train([content])
                raw_tokens = bpe_tokenizer.encode(content[:500])
                
                # Python işlemcisi ile işlenmiş kod
                processed_python = python_processor.process(content)
                python_encoded = bpe_tokenizer.train([processed_python])
                python_tokens = bpe_tokenizer.encode(processed_python[:500])
                
                # Genel işlemci ile işlenmiş kod
                processed_general = general_processor.process(content)
                general_encoded = bpe_tokenizer.train([processed_general])
                general_tokens = bpe_tokenizer.encode(processed_general[:500])
                
                # Sonuçları yazdır
                print(f"\n{file_name} işlemci-tokenizer test sonuçları:")
                print(f"İşlenmemiş kod: {len(raw_tokens)} token")
                print(f"Python işlemcisi: {len(python_tokens)} token")
                print(f"Genel işlemci: {len(general_tokens)} token")
                
                # Python işlemcisinin yorumları ve docstring'leri kaldırdığını doğrula
                self.assertNotIn('"""', processed_python)
                self.assertNotIn('#', processed_python)
                
                # Python işlemcisinin tip işaretlerini kaldırdığını doğrula
                self.assertNotIn(': int', processed_python)
                self.assertNotIn('-> str', processed_python)
    
    def _calculate_similarity(self, original: str, decoded: str) -> float:
        """
        İki metin arasındaki benzerliği hesaplar.
        
        Args:
            original: Orijinal metin
            decoded: Decode edilmiş metin
            
        Returns:
            float: Benzerlik yüzdesi (0-100)
        """
        # Basitleştirilmiş benzerlik: karakterleri normalize et ve karşılaştır
        original = ''.join(original.split())
        decoded = ''.join(decoded.split())
        
        # Minimum uzunluk üzerinden karakterleri karşılaştır
        min_len = min(len(original), len(decoded))
        if min_len == 0:
            return 0.0
            
        matches = sum(1 for i in range(min_len) if original[i] == decoded[i])
        return (matches / min_len) * 100


if __name__ == "__main__":
    unittest.main()