# aiquantr_tokenizer/tokenizers/base.py
"""
Tokenizer modellerinin temel sınıfları.

Bu modül, tüm tokenizer uygulamaları için temel sınıfları
ve ortak işlevselliği tanımlar.
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Iterator, Callable

import numpy as np

# Logger oluştur
logger = logging.getLogger(__name__)


class BaseTokenizer(ABC):
    """
    Tüm tokenizer modelleri için temel sınıf.
    
    Bu soyut temel sınıf, tokenizer modelleri için
    gereken minimum arayüzü tanımlar.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None,
        name: Optional[str] = None
    ):
        """
        BaseTokenizer sınıfı başlatıcısı.
        
        Args:
            vocab_size: Sözlük boyutu (varsayılan: 30000)
            min_frequency: Minimum token frekansı (varsayılan: 2)
            special_tokens: Özel token eşlemeleri (varsayılan: None)
            name: Tokenizer adı (varsayılan: None - sınıf adı kullanılır)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.name = name or self.__class__.__name__
        
        # Varsayılan özel token tanımları
        default_special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "mask_token": "[MASK]"
        }
        
        # Sağlanan özel tokenları varsayılanlarla birleştir
        self.special_tokens = default_special_tokens.copy()
        if special_tokens:
            self.special_tokens.update(special_tokens)
            
        # Temel çağrı istatistikleri
        self.stats = {
            "num_encode_calls": 0,
            "num_decode_calls": 0,
            "total_encode_time": 0.0,
            "total_decode_time": 0.0,
        }
        
        # Eğitim ile ilgili meta veriler
        self.metadata = {}
        self.is_trained = False
    
    @abstractmethod
    def train(
        self,
        texts: List[str],
        trainer: Optional["TokenizerTrainer"] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokenizer modelini verilen metinler üzerinde eğitir.
        
        Args:
            texts: Eğitim metinleri
            trainer: Eğitimi yönetecek TokenizerTrainer nesnesi (varsayılan: None)
            **kwargs: Eğitim için ek parametreler
            
        Returns:
            Dict[str, Any]: Eğitim istatistikleri ve sonuçları
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """
        Tokenizer'ın sözlüğünü döndürür.
        
        Returns:
            Dict[str, int]: Token - ID eşleşmelerini içeren sözlük
        """
        pass
    
    def get_vocab_size(self) -> int:
        """
        Sözlük boyutunu döndürür.
        
        Returns:
            int: Sözlük boyutu
        """
        return len(self.get_vocab())
    
    @abstractmethod
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Tokenizer modelini kaydeder.
        
        Args:
            path: Kaydetme yolu
            **kwargs: Kaydetme için ek parametreler
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "BaseTokenizer":
        """
        Tokenizer modelini yükler.
        
        Args:
            path: Yükleme yolu
            **kwargs: Yükleme için ek parametreler
            
        Returns:
            BaseTokenizer: Yüklenen tokenizer modeli
        """
        pass
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Metni token dizilerine dönüştürür (ID'lere kodlamadan).
        
        Args:
            text: Tokenize edilecek metin
            **kwargs: Tokenize için ek parametreler
            
        Returns:
            List[str]: Token dizileri
        """
        # Bu metot alt sınıflar tarafından verimli bir şekilde uygulanabilir
        ids = self.encode(text, **kwargs)
        vocab = {v: k for k, v in self.get_vocab().items()}
        return [vocab.get(token_id, self.special_tokens["unk_token"]) for token_id in ids]
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        **kwargs
    ) -> List[List[int]]:
        """
        Metin listesini toplu olarak kodlar.
        
        Args:
            texts: Encode edilecek metinler listesi
            add_special_tokens: Başlangıç/bitiş tokenlarını ekle (varsayılan: True)
            **kwargs: Encode için ek parametreler
            
        Returns:
            List[List[int]]: Her metin için token ID'leri listesi
        """
        return [self.encode(text, add_special_tokens=add_special_tokens, **kwargs) for text in texts]
    
    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Token ID'leri listesi listesini metinlere dönüştürür.
        
        Args:
            batch_ids: Decode edilecek token ID'leri listelerinin listesi
            skip_special_tokens: Özel tokenları atla (varsayılan: True)
            **kwargs: Decode için ek parametreler
            
        Returns:
            List[str]: Elde edilen metinler
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs) for ids in batch_ids]
    
    def token_to_id(self, token: str) -> int:
        """
        Token dizisini ID'ye dönüştürür.
        
        Args:
            token: Dönüştürülecek token
            
        Returns:
            int: Token ID'si veya bilinmiyorsa unk_token ID'si
        """
        vocab = self.get_vocab()
        return vocab.get(token, vocab.get(self.special_tokens["unk_token"], 0))
    
    def id_to_token(self, token_id: int) -> str:
        """
        Token ID'sini token dizisine dönüştürür.
        
        Args:
            token_id: Dönüştürülecek token ID'si
            
        Returns:
            str: Token dizisi veya sınırlar dışındaysa unk_token
        """
        vocab = {v: k for k, v in self.get_vocab().items()}
        return vocab.get(token_id, self.special_tokens["unk_token"])
    
    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Tokenizer'ı transformers uyumlu biçimde kaydeder.
        
        Args:
            path: Kaydetme dizini
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Tokenizer modelini kaydet
        self.save(path / "tokenizer.json")
        
        # Yapılandırma dosyasını kaydet
        config = {
            "model_type": self.__class__.__name__,
            "name": self.name,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "min_frequency": self.min_frequency,
            "metadata": self.metadata
        }
        
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Tokenizer {path} konumuna kaydedildi")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Tokenizer istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistik sözlüğü
        """
        vocab = self.get_vocab()
        special_token_ids = {self.token_to_id(token): token for token in self.special_tokens.values()}
        
        stats = dict(self.stats)
        stats.update({
            "vocab_size": len(vocab),
            "special_tokens_count": len(special_token_ids),
            "is_trained": self.is_trained
        })
        
        # Ortalama kodlama/kod çözme sürelerini hesapla
        if stats["num_encode_calls"] > 0:
            stats["avg_encode_time"] = stats["total_encode_time"] / stats["num_encode_calls"]
        if stats["num_decode_calls"] > 0:
            stats["avg_decode_time"] = stats["total_decode_time"] / stats["num_decode_calls"]
        
        # Meta verileri ekle
        stats["metadata"] = dict(self.metadata)
        
        return stats


class TokenizerTrainer:
    """
    Tokenizer eğitimini yöneten sınıf.
    
    Bu sınıf, tokenizer eğitimini yapılandırmak,
    izlemek ve optimize etmek için kullanılır.
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        num_iterations: Optional[int] = None,
        show_progress: bool = True,
        progress_interval: int = 10,
        callbacks: Optional[List[Callable]] = None,
        seed: Optional[int] = None
    ):
        """
        TokenizerTrainer sınıfı başlatıcısı.
        
        Args:
            batch_size: Eğitim toplu iş boyutu (varsayılan: 1000)
            num_iterations: Maksimum yineleme sayısı (varsayılan: None - tokenizer'a bağlı)
            show_progress: İlerleme durumunu göster (varsayılan: True)
            progress_interval: İlerleme raporlama aralığı (varsayılan: 10)
            callbacks: Eğitim geri çağrıları (varsayılan: None)
            seed: Rastgele başlangıç değeri (varsayılan: None)
        """
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.show_progress = show_progress
        self.progress_interval = progress_interval
        self.callbacks = callbacks or []
        self.seed = seed
        
        # Eğitim durumunu izleme
        self.history = {
            "iterations": [],
            "metrics": {},
            "timings": {}
        }
        
        # Daha fazla izleme alanları alt sınıflar tarafından eklenebilir
        self.current_iteration = 0
        self.best_score = None
        self.start_time = None
    
    def on_training_begin(
        self,
        tokenizer: BaseTokenizer,
        texts: List[str]
    ) -> None:
        """
        Eğitim başlamadan önce çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            texts: Eğitim metinleri
        """
        self.start_time = time.time()
        self.current_iteration = 0
        
        # Geri çağrıları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_training_begin"):
                callback.on_training_begin(tokenizer=tokenizer, texts=texts, trainer=self)
        
        if self.show_progress:
            logger.info(f"Eğitim başladı: {len(texts)} örnek, {self.batch_size} toplu iş boyutu")
    
    def on_iteration_begin(
        self,
        tokenizer: BaseTokenizer,
        iteration: int
    ) -> None:
        """
        Her yineleme başlamadan önce çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            iteration: Mevcut yineleme indeksi
        """
        self.current_iteration = iteration
        
        # Geri çağrıları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_iteration_begin"):
                callback.on_iteration_begin(tokenizer=tokenizer, iteration=iteration, trainer=self)
    
    def on_iteration_end(
        self,
        tokenizer: BaseTokenizer,
        iteration: int,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Her yineleme bittikten sonra çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            iteration: Tamamlanan yineleme indeksi
            metrics: Yineleme metrikleri
        """
        # İlerleme geçmişini güncelle
        self.history["iterations"].append(iteration)
        
        for k, v in metrics.items():
            if k not in self.history["metrics"]:
                self.history["metrics"][k] = []
            self.history["metrics"][k].append(v)
        
        elapsed_time = time.time() - self.start_time
        self.history["timings"][iteration] = elapsed_time
        
        # İlerleme durumunu göster
        if self.show_progress and (
            iteration % self.progress_interval == 0 or
            iteration == 0 or
            iteration == self.num_iterations - 1
        ):
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(
                f"Yineleme {iteration}/{self.num_iterations}: {metrics_str} "
                f"(Süre: {elapsed_time:.2f}s)"
            )
        
        # Geri çağrıları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_iteration_end"):
                callback.on_iteration_end(
                    tokenizer=tokenizer,
                    iteration=iteration,
                    metrics=metrics,
                    trainer=self
                )
    
    def on_training_end(
        self,
        tokenizer: BaseTokenizer,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Eğitim tamamlandıktan sonra çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            metrics: Son metrikler
        """
        elapsed_time = time.time() - self.start_time
        
        # Meta verileri güncelle
        tokenizer.metadata.update({
            "training_time": elapsed_time,
            "iterations": self.current_iteration + 1,
            "final_metrics": metrics
        })
        
        # İlerleme durumunu göster
        if self.show_progress:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(
                f"Eğitim tamamlandı: {metrics_str} "
                f"(Toplam süre: {elapsed_time:.2f}s)"
            )
        
        # Geri çağrıları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_training_end"):
                callback.on_training_end(tokenizer=tokenizer, metrics=metrics, trainer=self)
    
    def should_stop_early(
        self,
        tokenizer: BaseTokenizer,
        iteration: int,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Eğitimin erken durdurulması gerekip gerekmediğini kontrol eder.
        
        Args:
            tokenizer: Eğitilen tokenizer
            iteration: Mevcut yineleme indeksi
            metrics: Mevcut metrikler
            
        Returns:
            bool: Eğitim erken durdurulmalı mı
        """
        # En büyük yineleme sayısını kontrol et
        if self.num_iterations is not None and iteration >= self.num_iterations - 1:
            return True
            
        # Herhangi bir geri çağrı erken durdurma istedi mi
        for callback in self.callbacks:
            if hasattr(callback, "should_stop_early") and callback.should_stop_early(
                tokenizer=tokenizer,
                iteration=iteration,
                metrics=metrics,
                trainer=self
            ):
                return True
                
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Eğitim özeti döndürür.
        
        Returns:
            Dict[str, Any]: Eğitim istatistikleri ve geçmişi
        """
        elapsed_time = 0.0
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            
        return {
            "total_iterations": len(self.history["iterations"]),
            "elapsed_time": elapsed_time,
            "history": self.history,
            "final_metrics": {
                k: v[-1] if v else None
                for k, v in self.history["metrics"].items()
            }
        }