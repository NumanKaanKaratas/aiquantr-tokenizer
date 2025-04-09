"""
Temel veri kaynağı sınıfı.

Bu modül, diğer tüm veri kaynakları için temel sınıfı tanımlar.
Her veri kaynağı bu temel sınıfı miras almalıdır.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, Union, List

# Logger oluştur
logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """
    Tüm veri kaynakları için temel sınıf.
    
    Bu soyut temel sınıf, veri kaynaklarının uyması gereken
    ortak arayüzü tanımlar.
    """
    
    def __init__(
        self, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        text_key: str = "text",
        min_length: int = 0,
        max_length: Optional[int] = None,
        limit: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        BaseDataSource başlatıcısı.
        
        Args:
            name: Veri kaynağı adı (varsayılan: None - sınıf adı kullanılır)
            description: Veri kaynağı açıklaması
            text_key: Metin verisi anahtarı
            min_length: Minimum metin uzunluğu
            max_length: Maksimum metin uzunluğu
            limit: Maksimum yüklenecek örnek sayısı
            metadata: Ek üst veri
        """
        self.name = name or self.__class__.__name__
        self.description = description
        self.text_key = text_key
        self.min_length = min_length
        self.max_length = max_length
        self.limit = limit
        self.metadata = metadata or {}
        
        # İstatistikler
        self.stats = {
            "total_samples": 0,
            "loaded_samples": 0,
            "skipped_samples": 0,
            "skipped_length": 0,
            "total_chars": 0
        }
    
    @abstractmethod
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        Veri kaynağından veri yükler.
        
        Bu metod her veri kaynağı tarafından uygulanmalıdır.
        
        Yields:
            Dict[str, Any]: Yüklenen veri öğeleri
        """
        pass
    
    def filter_text(self, text: str) -> Optional[str]:
        """
        Metni filtreler.
        
        Args:
            text: Filtrelenecek metin
            
        Returns:
            Optional[str]: Filtrelenmiş metin veya None
        """
        if not text:
            self.stats["skipped_samples"] += 1
            return None
            
        # Uzunluk filtresi
        if self.min_length > 0 and len(text) < self.min_length:
            self.stats["skipped_length"] += 1
            return None
            
        if self.max_length and len(text) > self.max_length:
            # Uzun metni kes
            text = text[:self.max_length]
            
        return text
    
    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Veri öğesini işler ve filtreler.
        
        Args:
            item: İşlenecek veri öğesi
            
        Returns:
            Optional[Dict[str, Any]]: İşlenmiş veri öğesi veya None
        """
        # Metin alanı kontrolü
        if self.text_key not in item:
            logger.warning(f"'{self.text_key}' alanı bulunamadı: {str(item)[:50]}...")
            self.stats["skipped_samples"] += 1
            return None
        
        # Metin filtresi
        text = item[self.text_key]
        filtered_text = self.filter_text(text)
        if filtered_text is None:
            return None
            
        # Filtrelenmiş metni geri ekle
        item[self.text_key] = filtered_text
        
        # İstatistikleri güncelle
        self.stats["loaded_samples"] += 1
        self.stats["total_chars"] += len(filtered_text)
        
        return item
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Veri kaynağı üzerinde doğrudan yinelemeyi destekler.
        
        Yields:
            Dict[str, Any]: Yüklenen ve filtrelenmiş veri öğeleri
        """
        return self.load_data()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Veri kaynağı istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistikler
        """
        return dict(self.stats)
    
    def reset_stats(self) -> None:
        """İstatistikleri sıfırlar."""
        self.stats = {
            "total_samples": 0,
            "loaded_samples": 0,
            "skipped_samples": 0,
            "skipped_length": 0,
            "total_chars": 0
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Veri kaynağı üst verilerini döndürür.
        
        Returns:
            Dict[str, Any]: Üst veriler
        """
        metadata = dict(self.metadata)
        metadata.update({
            "source_name": self.name,
            "source_type": self.__class__.__name__,
            "text_key": self.text_key
        })
        
        if self.description:
            metadata["description"] = self.description
            
        # Temel istatistikleri ekle
        metadata["stats"] = {
            "loaded_samples": self.stats["loaded_samples"],
            "total_chars": self.stats["total_chars"]
        }
        
        return metadata