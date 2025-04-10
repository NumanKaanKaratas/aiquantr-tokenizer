# aiquantr_tokenizer/tokenizers/byte_level.py
"""
Byte seviyesi tokenizer uygulaması.

Bu modül, metni byte seviyesinde tokenize eden bir
tokenizer modelinin implementasyonunu sağlar.
"""

import os
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Iterator, Callable

from .base import BaseTokenizer, TokenizerTrainer

# Logger oluştur
logger = logging.getLogger(__name__)


class ByteLevelTokenizer(BaseTokenizer):
    """
    Byte seviyesi tokenizer implementasyonu.
    
    Bu sınıf, metni doğrudan byte seviyesinde tokenize eden
    basit ama güçlü bir tokenizer modelini sağlar.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        min_frequency: int = 1,
        special_tokens: Optional[Dict[str, str]] = None,
        add_prefix_space: bool = False,
        name: Optional[str] = None
    ):
        """
        ByteLevelTokenizer sınıfı başlatıcısı.
        
        Args:
            vocab_size: Sözlük boyutu (varsayılan: 256)
            min_frequency: Minimum token frekansı (varsayılan: 1)
            special_tokens: Özel token eşlemeleri (varsayılan: None)
            add_prefix_space: Kodlama sırasında önek boşluğu ekle (varsayılan: False)
            name: Tokenizer adı (varsayılan: None)
            
        Note:
            Byte-level tokenizer, varsayılan olarak 256 byte değerini içerir.
            Vocab_size parametresi, özel tokenlar için ek alan ekler.
        """
        # vocab_size ≥ 256 olmalı
        if vocab_size < 256:
            logger.warning(f"vocab_size {vocab_size} olarak belirtildi, ancak en az 256 olmalı. 256'ya ayarlanıyor.")
            vocab_size = 256
            
        super().__init__(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            name=name or "ByteLevelTokenizer"
        )
        
        self.add_prefix_space = add_prefix_space
        
        # Sözlük ve byte eşleştirme
        self.byte_encoder = {}  # Byte -> Token ID
        self.byte_decoder = {}  # Token ID -> Byte
        
        # Özel tokenlar için ID aralığı
        self.special_token_ids = {}
        
        # Temel byte sözlüğünü oluştur
        self._init_byte_vocab()
    
    def _init_byte_vocab(self):
        """Temel byte sözlüğünü oluşturur."""
        # 0-255 aralığındaki her byte için ID ata
        for i in range(256):
            self.byte_encoder[i] = i
            self.byte_decoder[i] = i
            
        # Özel tokenlar için ID'ler ata
        next_id = 256
        for token_type, token in self.special_tokens.items():
            if next_id < self.vocab_size:
                self.special_token_ids[token_type] = next_id
                next_id += 1
            else:
                logger.warning(f"Vocab_size sınırı aşıldı. '{token_type}' için ID atanamadı.")
    
    def _text_to_bytes(self, text: str) -> List[int]:
        """
        Metni byte dizisine dönüştürür.
        
        Args:
            text: Dönüştürülecek metin
            
        Returns:
            List[int]: Byte değerleri listesi
        """
        # Önce UTF-8'e kodla
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
            
        return list(text.encode("utf-8"))
    
    def _bytes_to_text(self, bytes_list: List[int]) -> str:
        """
        Byte dizisini metne dönüştürür.
        
        Args:
            bytes_list: Dönüştürülecek byte dizisi
            
        Returns:
            str: Elde edilen metin
        """
        try:
            text = bytes(bytes_list).decode("utf-8", errors="replace")
            return text
        except Exception as e:
            logger.warning(f"Byte dizisi metne dönüştürülürken hata: {e}")
            # En iyi çabayı göster
            return "".join([chr(b) if b < 128 else "�" for b in bytes_list])
    
    def train(
        self,
        texts: List[str],
        trainer: Optional[TokenizerTrainer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokenizer modelini verilen metinler üzerinde eğitir.
        
        Args:
            texts: Eğitim metinleri
            trainer: Eğitimi yönetecek TokenizerTrainer nesnesi (varsayılan: None)
            **kwargs: Eğitim için ek parametreler
            
        Returns:
            Dict[str, Any]: Eğitim istatistikleri
            
        Note:
            Byte-level tokenizer için eğitim aslında sözlük oluşturmakla sınırlıdır.
            Özel tokenları ekler ve karakterlerin frekanslarını hesaplar.
        """
        # Eğitim opsiyonları
        compute_frequencies = kwargs.get("compute_frequencies", True)
        
        # Eğitim yöneticisini yapılandır
        if trainer is None:
            trainer = TokenizerTrainer(
                batch_size=kwargs.get("batch_size", 1000),
                num_iterations=1,  # Tek adımda işlem tamamlanır
                show_progress=kwargs.get("show_progress", True)
            )
            
        # Eğitim başlangıcı
        trainer.on_training_begin(self, texts)
        trainer.on_iteration_begin(self, 0)
        
        # Karakter frekanslarını hesapla (isteğe bağlı)
        byte_freqs = Counter()
        num_bytes = 0
        
        if compute_frequencies:
            for text in texts:
                byte_values = self._text_to_bytes(text)
                byte_freqs.update(byte_values)
                num_bytes += len(byte_values)
        
        # Eğitim meta verilerini güncelle
        self.metadata.update({
            "training_size": len(texts),
            "total_bytes": num_bytes,
            "byte_frequencies": {b: freq for b, freq in byte_freqs.most_common(30)}
        })
        
        # İlerleme metrikleri
        metrics = {
            "vocab_size": len(self.byte_encoder) + len(self.special_token_ids),
            "unique_bytes": len(byte_freqs),
            "total_bytes": num_bytes
        }
        
        trainer.on_iteration_end(self, 0, metrics)
        
        # Eğitimi tamamla
        self._is_trained = True
        final_metrics = metrics.copy()
        
        trainer.on_training_end(self, final_metrics)
        return final_metrics
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Metni token ID'lerine dönüştürür.
        
        Args:
            text: Encode edilecek metin
            add_special_tokens: Başlangıç/bitiş tokenlarını ekle (varsayılan: True)
            **kwargs: Encode için ek parametreler
            
        Returns:
            List[int]: Token ID'leri
        """
        start_time = time.time()
        
        if not text:
            result = []
            
            if add_special_tokens:
                bos_id = self.special_token_ids.get("bos_token")
                eos_id = self.special_token_ids.get("eos_token")
                
                if bos_id is not None:
                    result.append(bos_id)
                if eos_id is not None:
                    result.append(eos_id)
            
            self.stats["num_encode_calls"] += 1
            self.stats["total_encode_time"] += time.time() - start_time
            return result
        
        # Metni byte listesine dönüştür
        bytes_list = self._text_to_bytes(text)
        ids = []
        
        # Başlangıç tokeni ekle
        if add_special_tokens and "bos_token" in self.special_token_ids:
            ids.append(self.special_token_ids["bos_token"])
            
        # Her byte için ilgili ID'yi ekle
        for b in bytes_list:
            ids.append(self.byte_encoder.get(b, self.special_token_ids.get("unk_token", 0)))
            
        # Bitiş tokeni ekle
        if add_special_tokens and "eos_token" in self.special_token_ids:
            ids.append(self.special_token_ids["eos_token"])
        
        # İstatistikleri güncelle
        self.stats["num_encode_calls"] += 1
        self.stats["total_encode_time"] += time.time() - start_time
        
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Token ID'lerini metne dönüştürür.
        
        Args:
            ids: Decode edilecek token ID'leri
            skip_special_tokens: Özel tokenları atla (varsayılan: True)
            **kwargs: Decode için ek parametreler
            
        Returns:
            str: Elde edilen metin
        """
        start_time = time.time()
        
        if not ids:
            self.stats["num_decode_calls"] += 1
            self.stats["total_decode_time"] += time.time() - start_time
            return ""
            
        # Özel token ID'lerini belirle
        special_ids = set(self.special_token_ids.values()) if skip_special_tokens else set()
        
        # Token ID'lerini byte değerlerine dönüştür
        byte_values = []
        for token_id in ids:
            if token_id in special_ids:
                continue
                
            if token_id in self.byte_decoder:
                byte_values.append(self.byte_decoder[token_id])
                
        # Byte dizisini metne dönüştür
        text = self._bytes_to_text(byte_values)
        
        # İstatistikleri güncelle
        self.stats["num_decode_calls"] += 1
        self.stats["total_decode_time"] += time.time() - start_time
        
        return text
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Tokenizer'ın sözlüğünü döndürür.
        
        Returns:
            Dict[str, int]: Token - ID eşleşmelerini içeren sözlük
        """
        vocab = {}
        
        # Byte karakter sözlüğü
        for byte_val, token_id in self.byte_encoder.items():
            token_char = chr(byte_val) if 32 <= byte_val <= 126 else f"<byte_{byte_val}>"
            vocab[token_char] = token_id
            
        # Özel tokenları ekle
        for token_type, token in self.special_tokens.items():
            if token_type in self.special_token_ids:
                vocab[token] = self.special_token_ids[token_type]
                
        return vocab
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Tokenizer modelini kaydeder.
        
        Args:
            path: Kaydetme yolu
            **kwargs: Kaydetme için ek parametreler
        """
        path = Path(path)
        
        # Ana dizini oluştur
        if path.is_dir():
            save_path = path / "tokenizer.json"
        else:
            save_path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # Model verilerini kaydet
        data = {
            "type": "ByteLevelTokenizer",
            "byte_encoder": {str(b): token_id for b, token_id in self.byte_encoder.items()},
            "special_token_ids": self.special_token_ids,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "add_prefix_space": self.add_prefix_space,
            "metadata": self.metadata
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Tokenizer {save_path} konumuna kaydedildi")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "ByteLevelTokenizer":
        """
        Tokenizer modelini yükler.
        
        Args:
            path: Yükleme yolu
            **kwargs: Yükleme için ek parametreler
            
        Returns:
            ByteLevelTokenizer: Yüklenen tokenizer modeli
            
        Raises:
            ValueError: Model yüklenemezse
        """
        path = Path(path)
        
        # Dosya yolu veya dizin yolu kontrolü
        if path.is_dir():
            load_path = path / "tokenizer.json"
        else:
            load_path = path
            
        if not load_path.exists():
            raise ValueError(f"Tokenizer dosyası bulunamadı: {load_path}")
            
        # Model verilerini yükle
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Model türünü doğrula
        if data.get("type") != "ByteLevelTokenizer":
            logger.warning(f"Yüklenen tokenizer türü uyumsuz. Beklenen: ByteLevelTokenizer, Alınan: {data.get('type')}")
            
        # Tokenizer'ı yapılandır
        tokenizer = cls(
            vocab_size=data.get("vocab_size", 256),
            min_frequency=data.get("min_frequency", 1),
            special_tokens=data.get("special_tokens"),
            add_prefix_space=data.get("add_prefix_space", False)
        )
        
        # Byte eşlemelerini yükle
        tokenizer.byte_encoder = {int(b): token_id for b, token_id in data.get("byte_encoder", {}).items()}
        tokenizer.byte_decoder = {token_id: int(b) for b, token_id in data.get("byte_encoder", {}).items()}
        
        # Özel token ID'lerini yükle
        tokenizer.special_token_ids = data.get("special_token_ids", {})
        
        # Meta verileri yükle
        if "metadata" in data:
            tokenizer.metadata.update(data["metadata"])
            
        # Eğitilmiş olarak işaretle
        tokenizer.is_trained = True
        
        logger.info(f"Tokenizer {load_path} konumundan yüklendi")
        return tokenizer