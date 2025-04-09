"""
Özel veri kaynakları.

Bu modül, kullanıcı tanımlı veri kaynaklarını ve
veri kaynağı adaptörlerini içerir.
"""

import logging
from typing import Dict, Any, Optional, List, Iterator, Union, Set, Callable

from aiquantr_tokenizer.data.sources.base_source import BaseDataSource

# Logger oluştur
logger = logging.getLogger(__name__)


class CustomDataSource(BaseDataSource):
    """
    Özel işlev veya iterable nesnelerden veri yükleyen kaynak.
    """
    
    def __init__(
        self,
        data_source: Union[Callable[[], Iterator[Any]], List[Any], Iterator[Any]],
        item_mapping: Optional[Callable[[Any], Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        CustomDataSource başlatıcısı.
        
        Args:
            data_source: Veri kaynağı (bir fonksiyon, liste veya iterator)
            item_mapping: Veri öğelerini haritalama fonksiyonu (varsayılan: None)
            **kwargs: BaseDataSource için ek parametreler
        """
        super().__init__(**kwargs)
        
        self.data_source = data_source
        self.item_mapping = item_mapping
    
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        Özel veri kaynağından veri yükler.
        
        Yields:
            Dict[str, Any]: Yüklenen veri öğeleri
        """
        try:
            # Veri kaynağını al
            if callable(self.data_source):
                data_iter = self.data_source()
            elif hasattr(self.data_source, '__iter__'):
                data_iter = iter(self.data_source)
            else:
                logger.error(f"Desteklenmeyen veri kaynağı türü: {type(self.data_source)}")
                return
            
            # Veri öğelerini işle
            count = 0
            for raw_item in data_iter:
                # Limit kontrolü
                if self.limit and count >= self.limit:
                    break
                
                # İstatistik güncelle
                self.stats["total_samples"] += 1
                
                # Veriyi haritalama
                if self.item_mapping:
                    try:
                        item = self.item_mapping(raw_item)
                    except Exception as e:
                        logger.error(f"Öğe haritalama hatası: {str(e)}")
                        self.stats["skipped_samples"] += 1
                        continue
                elif isinstance(raw_item, dict):
                    item = raw_item
                elif isinstance(raw_item, str):
                    item = {self.text_key: raw_item}
                else:
                    # Desteklenmeyen veri türü
                    logger.warning(f"Desteklenmeyen veri öğesi türü: {type(raw_item)}")
                    self.stats["skipped_samples"] += 1
                    continue
                
                # Öğeyi işle
                processed_item = self.process_item(item)
                if processed_item:
                    yield processed_item
                    count += 1
                    
        except Exception as e:
            logger.error(f"Özel veri kaynağı yükleme hatası: {str(e)}")