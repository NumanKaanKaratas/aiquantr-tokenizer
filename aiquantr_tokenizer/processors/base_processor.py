"""
Temel işlemci sınıfları.

Bu modül, veri işleme işlemleri için temel sınıfları içerir.
Tüm özel işlemciler bu temel sınıfları miras almalıdır.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable, Union

# Logger oluştur
logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Tüm işlemciler için temel sınıf.
    
    Bu soyut temel sınıf, tüm veri işlemcileri için ortak
    arayüzü ve işlevselliği tanımlar.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        BaseProcessor sınıfı başlatıcısı.
        
        Args:
            name: İşlemci adı (varsayılan: None - sınıf adı kullanılır)
        """
        self.name = name or self.__class__.__name__
        self.stats = {
            "processed_count": 0,
            "total_chars_in": 0,
            "total_chars_out": 0
        }
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Veri üzerinde işleme yapar.
        
        Args:
            data: İşlenecek veri
            
        Returns:
            Any: İşlenmiş veri
        """
        pass
    
    def __call__(self, data: Any) -> Any:
        """
        Processor'u doğrudan bir fonksiyon gibi çağırılabilir yapar.
        
        Args:
            data: İşlenecek veri
            
        Returns:
            Any: İşlenmiş veri
        """
        # İstatistikleri güncelle
        self.stats["processed_count"] += 1
        
        # Eğer string ise, karakter sayılarını kaydet
        if isinstance(data, str):
            self.stats["total_chars_in"] += len(data)
        
        # İşleme yap
        result = self.process(data)
        
        # Eğer sonuç string ise, karakter sayılarını kaydet
        if isinstance(result, str):
            self.stats["total_chars_out"] += len(result)
        
        return result
    
    def reset_stats(self) -> None:
        """İstatistikleri sıfırlar."""
        self.stats = {
            "processed_count": 0,
            "total_chars_in": 0,
            "total_chars_out": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        İşlemci istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistikler sözlüğü
        """
        stats = dict(self.stats)
        
        # Ortalama karakter değişikliği (eğer dize işleme yapıldıysa)
        if stats["processed_count"] > 0 and stats["total_chars_in"] > 0:
            stats["avg_chars_in"] = stats["total_chars_in"] / stats["processed_count"]
            stats["avg_chars_out"] = stats["total_chars_out"] / stats["processed_count"]
            
            # Toplam karakter değişimi yüzdesi
            stats["char_change_percent"] = (stats["total_chars_out"] - stats["total_chars_in"]) / stats["total_chars_in"] * 100
            
        return stats


class ProcessingPipeline(BaseProcessor):
    """
    Birden fazla işlemciyi sırayla uygulayan işleme hattı.
    
    Bu sınıf, bir dizi işlemciyi zincir halinde uygulayarak
    verileri adım adım işler.
    """
    
    def __init__(
        self,
        processors: List[BaseProcessor],
        skip_empty: bool = True,
        name: Optional[str] = None
    ):
        """
        ProcessingPipeline sınıfı başlatıcısı.
        
        Args:
            processors: Uygulanacak işlemciler listesi
            skip_empty: Boş verileri işlemeden geç (varsayılan: True)
            name: İşleme hattı adı (varsayılan: None)
        """
        super().__init__(name=name or "ProcessingPipeline")
        
        if not processors:
            raise ValueError("İşlemciler listesi boş olamaz.")
            
        self.processors = processors
        self.skip_empty = skip_empty
        
        # İşleme hattı istatistikleri
        self.stats.update({
            "skipped_empty": 0,
            "final_empty": 0
        })
    
    def process(self, data: Any) -> Any:
        """
        Veriyi işleme hattından geçirir.
        
        Args:
            data: İşlenecek veri
            
        Returns:
            Any: Tüm işlemcilerden geçirilmiş veri
        """
        if self.skip_empty:
            if data is None or (isinstance(data, (str, list, dict, tuple, set)) and not data):
                self.stats["skipped_empty"] += 1
                return data
            
        result = data
        
        # Her işlemciyi sırayla uygula
        for processor in self.processors:
            result = processor(result)
            
            # Boş sonuç kontrolü
            is_empty = result is None or (isinstance(result, (str, list, dict, tuple, set)) and not result)
            
            # Eğer veri boş hale geldiyse ve boş verileri atla seçeneği etkinse
            if is_empty and self.skip_empty:
                self.stats["final_empty"] += 1
                break
                
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        İşleme hattı ve içerdiği işlemcilerin istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İşleme hattı istatistikleri
        """
        stats = super().get_stats()
        
        # Her işlemcinin istatistiklerini ekle
        for i, processor in enumerate(self.processors):
            stats[f"processor_{i}"] = {
                "name": processor.name,
                "type": processor.__class__.__name__,
                "stats": processor.get_stats()
            }
            
        return stats