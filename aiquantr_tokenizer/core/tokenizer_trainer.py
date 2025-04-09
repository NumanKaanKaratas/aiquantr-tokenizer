# aiquantr_tokenizer/core/tokenizer_trainer.py
"""
Tokenizer eğitim süreç kontrolü.

Bu modül, tokenizer eğitimi için gerekli süreç
kontrolü ve ilerleme izleme işlevselliğini içerir.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union, Callable

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TokenizerTrainer:
    """
    Tokenizer eğitimini kontrol eden sınıf.
    
    Bu sınıf, tokenizer eğitimini yöneten, 
    ilerlemeyi izleyen ve raporlayan bir ara katmandır.
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        num_iterations: Optional[int] = None,
        show_progress: bool = True,
        validation_split: float = 0.1,
        seed: int = 42,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        TokenizerTrainer sınıfı başlatıcısı.
        
        Args:
            batch_size: Eğitim toplu iş boyutu (varsayılan: 1000)
            num_iterations: Maksimum iterasyon sayısı (varsayılan: None)
            show_progress: İlerlemeyi göster (varsayılan: True)
            validation_split: Doğrulama kümesi oranı (varsayılan: 0.1)
            seed: Rastgele başlangıç değeri (varsayılan: 42)
            callbacks: Eğitim geri çağırma fonksiyonları (varsayılan: None)
        """
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.show_progress = show_progress
        self.validation_split = validation_split
        self.seed = seed
        self.callbacks = callbacks or []
        
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "total_duration": None,
            "iterations": [],
            "validation_results": {}
        }
        
        self.pbar = None
        
    def on_training_begin(self, tokenizer, texts: List[str]):
        """
        Eğitim başlangıcında çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            texts: Eğitim metinleri
        """
        self.training_stats["start_time"] = time.time()
        logger.info(f"Eğitim başlıyor: {len(texts)} metin, vocab_size={tokenizer.vocab_size}")
        
        # Geri çağırmaları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_training_begin"):
                callback.on_training_begin(tokenizer, texts)
                
    def on_iteration_begin(self, tokenizer, iteration: int):
        """
        Her iterasyonun başlangıcında çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            iteration: İterasyon numarası
        """
        # İlerleme çubuğunu başlat
        if self.show_progress and self.num_iterations and not self.pbar:
            self.pbar = tqdm(total=self.num_iterations, desc="Eğitim")
            
        # Geri çağırmaları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_iteration_begin"):
                callback.on_iteration_begin(tokenizer, iteration)
    
    def on_iteration_end(self, tokenizer, iteration: int, metrics: Dict[str, Any]):
        """
        Her iterasyonun sonunda çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            iteration: İterasyon numarası
            metrics: İterasyon metrikleri
        """
        # İlerleme çubuğunu güncelle
        if self.show_progress and self.pbar:
            self.pbar.update(1)
            
            # İlerleme çubuğuna metrikleri ekle
            postfix = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    postfix[key] = value
                    
            if postfix:
                self.pbar.set_postfix(postfix)
                
        # İterasyon istatistiklerini kaydet
        self.training_stats["iterations"].append({
            "iteration": iteration,
            "metrics": metrics,
            "time": time.time()
        })
        
        # Geri çağırmaları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_iteration_end"):
                callback.on_iteration_end(tokenizer, iteration, metrics)
    
    def on_training_end(self, tokenizer, final_metrics: Dict[str, Any]):
        """
        Eğitim sonunda çağrılır.
        
        Args:
            tokenizer: Eğitilen tokenizer
            final_metrics: Final eğitim metrikleri
        """
        # İlerleme çubuğunu kapat
        if self.pbar:
            self.pbar.close()
            self.pbar = None
            
        # Eğitim istatistiklerini güncelle
        self.training_stats["end_time"] = time.time()
        self.training_stats["total_duration"] = self.training_stats["end_time"] - self.training_stats["start_time"]
        self.training_stats["final_metrics"] = final_metrics
        
        logger.info(f"Eğitim tamamlandı: {self.training_stats['total_duration']:.2f}s")
        
        # Geri çağırmaları çalıştır
        for callback in self.callbacks:
            if hasattr(callback, "on_training_end"):
                callback.on_training_end(tokenizer, final_metrics)
                
    def validate(self, tokenizer, validation_texts: List[str]) -> Dict[str, Any]:
        """
        Doğrulama kümesi üzerinde tokenizer'ı değerlendirir.
        
        Args:
            tokenizer: Değerlendirilecek tokenizer
            validation_texts: Doğrulama metinleri
            
        Returns:
            Dict[str, Any]: Doğrulama metrikleri
        """
        from ..metrics.quality_metrics import calculate_reconstruction_accuracy, calculate_token_frequency
        
        # Doğrulama metriklerini hesapla
        results = {}
        
        # Yeniden yapılandırma doğruluğu
        accuracy = calculate_reconstruction_accuracy(tokenizer, validation_texts[:100])
        results["reconstruction"] = accuracy
        
        # Token frekansları
        tokenized_texts = [tokenizer.tokenize(text) for text in validation_texts[:1000]]
        token_stats = calculate_token_frequency(tokenized_texts)
        results["token_stats"] = token_stats
        
        # İstatistikleri kaydet
        self.training_stats["validation_results"] = results
        
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Eğitim istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: Eğitim istatistikleri
        """
        return self.training_stats