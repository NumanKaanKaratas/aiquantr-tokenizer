"""
HuggingFace veri kaynağı.

Bu modül, HuggingFace Hub'dan veri kümesi yüklemek için
gerekli sınıfları içerir.
"""

import logging
from typing import Dict, Any, Optional, List, Iterator, Union, Set

from aiquantr_tokenizer.data.sources.base_source import BaseDataSource

# Logger oluştur
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, Dataset, IterableDataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    logger.warning("datasets paketi yüklü değil. HuggingFace veri kümeleri yüklenemez.")


class HuggingFaceDatasetSource(BaseDataSource):
    """
    HuggingFace Hub'daki veri kümelerinden veri yükleyen kaynak.
    """
    
    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
        streaming: bool = False,
        columns: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        HuggingFaceDatasetSource başlatıcısı.
        
        Args:
            dataset_name: HuggingFace veri kümesi adı
            subset: Veri kümesi alt kümesi (varsa)
            split: Veri kümesi bölümü (train, test, validation, ...)
            streaming: Akış modunda veri yükleme
            columns: Yüklenecek sütunlar
            cache_dir: Önbellek dizini
            **kwargs: BaseDataSource için ek parametreler
        """
        super().__init__(**kwargs)
        
        if not HAS_HF_DATASETS:
            raise ImportError("HuggingFace veri kümeleri için 'datasets' paketi gerekli.")
        
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.streaming = streaming
        self.columns = columns
        self.cache_dir = cache_dir
    
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        HuggingFace veri kümesinden veri yükleme.
        
        Yields:
            Dict[str, Any]: Yüklenen veri öğeleri
        """
        try:
            # Veri kümesini yükle
            dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                streaming=self.streaming,
                cache_dir=self.cache_dir
            )
            
            # Metin anahtarı kontrolü
            if not self.streaming and self.text_key not in dataset.column_names:
                # Alternatif metin anahtarları dene
                alt_keys = ['text', 'content', 'sentence', 'input_text', 'document']
                for key in alt_keys:
                    if key in dataset.column_names:
                        logger.warning(f"'{self.text_key}' sütunu bulunamadı. '{key}' kullanılıyor.")
                        self.text_key = key
                        break
                else:
                    # Hiçbir metin sütunu bulunamadı
                    logger.error(f"{self.dataset_name} veri kümesinde metin sütunu bulunamadı.")
                    return
            
            # Sütunları filtrele (eğer belirtilmişse)
            if not self.streaming and self.columns:
                # Varolan sütunları kontrol et
                valid_columns = [col for col in self.columns if col in dataset.column_names]
                if not valid_columns:
                    logger.warning(f"Belirtilen sütunlar bulunamadı. Tüm sütunlar yüklenecek.")
                else:
                    # Metin sütununu ekle (henüz yoksa)
                    if self.text_key not in valid_columns:
                        valid_columns.append(self.text_key)
                    dataset = dataset.select_columns(valid_columns)
            
            # Toplam örnek sayısı (akış modunda bilinmiyor)
            if not self.streaming:
                self.stats["total_samples"] = len(dataset)
            
            # Örnekleri yükle
            count = 0
            for item in dataset:
                # Limit kontrolü
                if self.limit and count >= self.limit:
                    break
                
                # Öğeyi işle
                processed_item = self.process_item(item)
                if processed_item:
                    yield processed_item
                    count += 1
                    
                    # İlerleme kaydı
                    if count % 10000 == 0:
                        logger.info(f"{count:,} örnek yüklendi...")
                    
            logger.info(f"Toplam {count:,} örnek yüklendi: {self.dataset_name} ({self.split})")
                
        except Exception as e:
            logger.error(f"{self.dataset_name} veri kümesi yüklenemedi: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Veri kaynağı üst verilerini döndürür.
        
        Returns:
            Dict[str, Any]: Üst veriler
        """
        metadata = super().get_metadata()
        metadata.update({
            "dataset_name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming
        })
        
        if self.subset:
            metadata["subset"] = self.subset
            
        return metadata