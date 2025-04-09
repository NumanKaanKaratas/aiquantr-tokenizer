# aiquantr_tokenizer/core/data_collector.py
"""
Veri toplama merkezi.

Bu modül, farklı veri kaynaklarından metin verisi
toplayan ve işleyen bileşenleri içerir.
"""

import logging
from typing import List, Dict, Any, Union, Optional, Set

from ..data.sources.base_source import BaseDataSource
from ..config.config_manager import ConfigManager
from ..data.cleaner import DataCleaner
from ..data.filter import DataFilter

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Farklı veri kaynaklarından veri toplayan sınıf.
    
    Bu sınıf, çeşitli veri kaynaklarını yönetir,
    verileri toplar, temizler ve filtreler.
    """
    
    def __init__(
        self, 
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None,
        sources: Optional[List[BaseDataSource]] = None
    ):
        """
        DataCollector sınıfı başlatıcısı.
        
        Args:
            config: Veri toplama yapılandırması
            sources: Veri kaynakları listesi
        """
        # Yapılandırma yönetimi
        if config is None:
            self.config = {}
        elif isinstance(config, ConfigManager):
            self.config = config.get_data_config()
        else:
            self.config = config
        
        # Veri kaynakları
        self.sources = sources or []
        
        # Veri işleme bileşenleri
        self.cleaner = DataCleaner(self.config.get("clean", {}))
        self.filter = DataFilter(self.config.get("filters", {}))
        
        # İstatistikler
        self.stats = {
            "total_collected": 0,
            "total_filtered": 0,
            "sources_stats": {},
            "filter_stats": {}
        }
        
    def add_source(self, source: BaseDataSource):
        """
        Veri kaynağı ekler.
        
        Args:
            source: Eklenecek veri kaynağı
        """
        self.sources.append(source)
        logger.debug(f"Veri kaynağı eklendi: {source.name}")
        
    def collect_data(self, limit: Optional[int] = None) -> List[str]:
        """
        Tüm kaynaklardan veri toplar.
        
        Args:
            limit: Toplanacak maksimum veri sayısı
            
        Returns:
            List[str]: Toplanan, temizlenen ve filtrelenen metinler
        """
        all_texts = []
        
        # Her kaynaktan veri topla
        for source in self.sources:
            logger.info(f"Veri toplanıyor: {source.name}")
            
            texts = source.load_data()
            source_len = len(texts)
            
            # İstatistikleri güncelle
            self.stats["sources_stats"][source.name] = {
                "raw_count": source_len
            }
            self.stats["total_collected"] += source_len
            
            # Temizle ve filtrele
            if self.config.get("clean_text", True):
                texts = self.cleaner.clean_texts(texts)
                logger.info(f"Temizlenen metin sayısı: {len(texts)}/{source_len}")
            
            # Filtrele
            if self.config.get("apply_filters", True):
                before_filter = len(texts)
                texts, filter_stats = self.filter.filter_texts(texts)
                
                # İstatistikleri güncelle
                filtered_count = before_filter - len(texts)
                self.stats["sources_stats"][source.name]["filtered_count"] = filtered_count
                self.stats["filter_stats"].update(filter_stats)
                logger.info(f"Filtrelenen metin sayısı: {filtered_count}/{before_filter}")
                
            all_texts.extend(texts)
            
            # Limit kontrolü
            if limit and len(all_texts) >= limit:
                all_texts = all_texts[:limit]
                logger.info(f"Limit aşıldı, {limit} metinde kesiliyor")
                break
        
        # Son işlemler
        if self.config.get("deduplicate", True):
            before_dedup = len(all_texts)
            all_texts = list(dict.fromkeys(all_texts))
            logger.info(f"Tekrarlanan metinler kaldırıldı: {before_dedup - len(all_texts)}/{before_dedup}")
            
        # Karıştırma
        if self.config.get("shuffle", True):
            import random
            random.shuffle(all_texts)
            logger.info("Veriler karıştırıldı")
            
        self.stats["final_count"] = len(all_texts)
        logger.info(f"Toplam toplanan metin: {self.stats['total_collected']}")
        logger.info(f"Son metin sayısı: {self.stats['final_count']}")
        
        return all_texts
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Veri toplama istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: Veri toplama istatistikleri
        """
        return self.stats