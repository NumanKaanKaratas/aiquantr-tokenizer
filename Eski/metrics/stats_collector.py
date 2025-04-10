"""
İstatistik toplama modülü.

Bu modül, tokenizer eğitimi sürecinde istatistikleri
toplayan ve raporlayan sınıfları ve fonksiyonları içerir.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Union

# Logger oluştur
logger = logging.getLogger(__name__)


class StatsCollector:
    """
    Genel istatistik toplama sınıfı.
    
    Bu sınıf, çeşitli işlem ve metriklere ait istatistikleri toplar,
    izler ve raporlar.
    """
    
    def __init__(self, name: str = "stats"):
        """
        StatsCollector başlatıcısı.
        
        Args:
            name: İstatistik toplayıcı adı
        """
        self.name = name
        self.stats = {}
        self.timestamps = {}
        
        # Başlangıç zamanı
        self._start_time = time.time()
    
    def add_stat(self, key: str, value: Any) -> None:
        """
        Yeni bir istatistik ekler veya varolan bir istatistiği günceller.
        
        Args:
            key: İstatistik anahtarı
            value: İstatistik değeri
        """
        self.stats[key] = value
    
    def increment_stat(self, key: str, increment: Union[int, float] = 1) -> None:
        """
        Sayısal bir istatistiği artırır.
        
        Args:
            key: İstatistik anahtarı
            increment: Artış miktarı (varsayılan: 1)
        """
        if key not in self.stats:
            self.stats[key] = increment
        elif isinstance(self.stats[key], (int, float)):
            self.stats[key] += increment
    
    def update_stats(self, stats_dict: Dict[str, Any]) -> None:
        """
        Birden çok istatistiği toplu olarak günceller.
        
        Args:
            stats_dict: İstatistik sözlüğü
        """
        self.stats.update(stats_dict)
    
    def get_stat(self, key: str) -> Any:
        """
        Bir istatistiği döndürür.
        
        Args:
            key: İstatistik anahtarı
            
        Returns:
            Any: İstatistik değeri
        """
        return self.stats.get(key)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Tüm istatistikleri döndürür.
        
        Returns:
            Dict[str, Any]: İstatistik sözlüğü
        """
        # Zamanı da ekle
        elapsed = time.time() - self._start_time
        stats_with_time = dict(self.stats)
        stats_with_time["elapsed_seconds"] = elapsed
        
        return stats_with_time
    
    def mark_time(self, marker: str) -> float:
        """
        Bir zaman işareti ekler ve geçen süreyi döndürür.
        
        Args:
            marker: Zaman işareti adı
            
        Returns:
            float: Başlangıçtan bu yana geçen süre (saniye)
        """
        current_time = time.time()
        elapsed = current_time - self._start_time
        self.timestamps[marker] = elapsed
        return elapsed
    
    def elapsed_since(self, marker: str) -> float:
        """
        Belirli bir zaman işaretinden bu yana geçen süreyi hesaplar.
        
        Args:
            marker: Zaman işareti adı
            
        Returns:
            float: İşaretten bu yana geçen süre (saniye)
        """
        if marker not in self.timestamps:
            return 0.0
            
        current_time = time.time()
        marker_time = self._start_time + self.timestamps[marker]
        return current_time - marker_time
    
    def elapsed_total(self) -> float:
        """
        Başlangıçtan bu yana geçen toplam süreyi hesaplar.
        
        Returns:
            float: Geçen toplam süre (saniye)
        """
        return time.time() - self._start_time
    
    def reset(self) -> None:
        """
        İstatistikleri ve zamanı sıfırlar.
        """
        self.stats = {}
        self.timestamps = {}
        self._start_time = time.time()
    
    def save(self, file_path: Union[str, Path], pretty: bool = True) -> None:
        """
        İstatistikleri bir dosyaya kaydeder.
        
        Args:
            file_path: Dosya yolu
            pretty: JSON formatını güzelleştir (varsayılan: True)
        """
        stats = self.get_all_stats()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(stats, f, indent=2 if pretty else None)
                
            logger.info(f"İstatistikler {file_path} dosyasına kaydedildi.")
            
        except Exception as e:
            logger.error(f"İstatistikler kaydedilemedi: {str(e)}")
    
    def load(self, file_path: Union[str, Path]) -> bool:
        """
        İstatistikleri bir dosyadan yükler.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            bool: Yükleme başarılı ise True
        """
        try:
            with open(file_path, 'r') as f:
                loaded_stats = json.load(f)
                
            self.stats = loaded_stats
            logger.info(f"İstatistikler {file_path} dosyasından yüklendi.")
            return True
            
        except Exception as e:
            logger.error(f"İstatistikler yüklenemedi: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """
        İstatistik özetini döndürür.
        
        Returns:
            str: İstatistik özeti
        """
        stats = self.get_all_stats()
        output = [f"=== {self.name} Statistics ==="]
        
        for key, value in stats.items():
            if key == "elapsed_seconds":
                minutes, seconds = divmod(value, 60)
                output.append(f"Elapsed Time: {int(minutes):02d}m {seconds:.2f}s")
            elif isinstance(value, (int, float)) and key != "elapsed_seconds":
                output.append(f"{key}: {value:,}")
            else:
                output.append(f"{key}: {value}")
                
        return "\n".join(output)


class ProcessingStats(StatsCollector):
    """
    Veri işleme istatistiklerini toplayan özel sınıf.
    """
    
    def __init__(self):
        """
        ProcessingStats başlatıcısı.
        """
        super().__init__(name="Processing")
        
        # İşleme istatistikleri
        self.stats.update({
            "files_processed": 0,
            "samples_processed": 0,
            "total_chars_in": 0,
            "total_chars_out": 0,
            "skipped_samples": 0,
            "filtered_samples": 0,
            "processing_throughput": 0.0  # chars/sec
        })
    
    def update_processing_stats(
        self,
        files_processed: int = 0,
        samples_processed: int = 0,
        chars_in: int = 0,
        chars_out: int = 0,
        skipped: int = 0,
        filtered: int = 0
    ) -> None:
        """
        İşleme istatistiklerini günceller.
        
        Args:
            files_processed: İşlenen dosya sayısı
            samples_processed: İşlenen örnek sayısı
            chars_in: Giriş karakter sayısı
            chars_out: Çıkış karakter sayısı
            skipped: Atlanan örnek sayısı
            filtered: Filtrelenen örnek sayısı
        """
        self.stats["files_processed"] += files_processed
        self.stats["samples_processed"] += samples_processed
        self.stats["total_chars_in"] += chars_in
        self.stats["total_chars_out"] += chars_out
        self.stats["skipped_samples"] += skipped
        self.stats["filtered_samples"] += filtered
        
        # Throughput hesapla
        elapsed = self.elapsed_total()
        if elapsed > 0:
            self.stats["processing_throughput"] = self.stats["total_chars_in"] / elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """
        İşleme istatistiklerinin özetini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistik özeti
        """
        stats = self.get_all_stats()
        
        if stats["samples_processed"] > 0:
            avg_chars_in = stats["total_chars_in"] / stats["samples_processed"]
            avg_chars_out = stats["total_chars_out"] / stats["samples_processed"]
        else:
            avg_chars_in = 0
            avg_chars_out = 0
        
        if stats["total_chars_in"] > 0:
            compression_ratio = stats["total_chars_out"] / stats["total_chars_in"]
        else:
            compression_ratio = 0
            
        summary = {
            "files_processed": stats["files_processed"],
            "samples_processed": stats["samples_processed"],
            "total_chars_in": stats["total_chars_in"],
            "total_chars_out": stats["total_chars_out"],
            "avg_chars_in": avg_chars_in,
            "avg_chars_out": avg_chars_out,
            "compression_ratio": compression_ratio,
            "skipped_samples": stats["skipped_samples"],
            "filtered_samples": stats["filtered_samples"],
            "processing_time": stats["elapsed_seconds"],
            "throughput": stats["processing_throughput"]
        }
        
        return summary


class TrainingStats(StatsCollector):
    """
    Tokenizer eğitim istatistiklerini toplayan özel sınıf.
    """
    
    def __init__(self):
        """
        TrainingStats başlatıcısı.
        """
        super().__init__(name="Training")
        
        # Eğitim istatistikleri
        self.stats.update({
            "epoch": 0,
            "samples_trained": 0,
            "vocab_size": 0,
            "loss": 0.0,
            "learning_rate": 0.0,
            "batch_size": 0,
            "training_throughput": 0.0  # samples/sec
        })
        
        # Epoch süreleri
        self.epoch_times = []
        
        # Kayıp değerleri
        self.losses = []
    
    def update_epoch(
        self,
        epoch: int,
        loss: float,
        samples: int,
        learning_rate: float
    ) -> None:
        """
        Epoch sonunda istatistikleri günceller.
        
        Args:
            epoch: Güncel epoch
            loss: Epoch kaybı
            samples: Eğitilen örnek sayısı
            learning_rate: Öğrenme oranı
        """
        self.stats["epoch"] = epoch
        self.stats["loss"] = loss
        self.stats["samples_trained"] += samples
        self.stats["learning_rate"] = learning_rate
        
        # Kayıpları kaydeder
        self.losses.append((epoch, loss))
        
        # Epoch süresini kaydeder
        epoch_time = self.mark_time(f"epoch_{epoch}")
        if epoch > 0:
            prev_time = self.timestamps.get(f"epoch_{epoch-1}", 0)
            self.epoch_times.append(epoch_time - prev_time)
            
            # Throughput hesapla
            if samples > 0 and epoch_time - prev_time > 0:
                self.stats["training_throughput"] = samples / (epoch_time - prev_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Eğitim istatistiklerinin özetini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistik özeti
        """
        stats = self.get_all_stats()
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        summary = {
            "total_epochs": stats["epoch"],
            "samples_trained": stats["samples_trained"],
            "vocab_size": stats["vocab_size"],
            "final_loss": stats["loss"],
            "learning_rate": stats["learning_rate"],
            "batch_size": stats["batch_size"],
            "avg_epoch_time": avg_epoch_time,
            "training_time": stats["elapsed_seconds"],
            "throughput": stats["training_throughput"],
            "loss_history": self.losses
        }
        
        return summary
    
    def save_loss_plot(self, file_path: Union[str, Path]) -> bool:
        """
        Kayıp grafiğini kaydeder.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            import matplotlib.pyplot as plt
            
            epochs, losses = zip(*self.losses)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(file_path, dpi=300)
            plt.close()
            
            logger.info(f"Kayıp grafiği {file_path} dosyasına kaydedildi.")
            return True
            
        except ImportError:
            logger.error("matplotlib yüklü değil. Grafik kaydedilemedi.")
            return False
        except Exception as e:
            logger.error(f"Grafik kaydedilemedi: {str(e)}")
            return False