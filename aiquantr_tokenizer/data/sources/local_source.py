"""
Yerel dosya veri kaynakları.

Bu modül, yerel dosya sistemindeki dosyalardan ve dizinlerden
veri yüklemeyi sağlayan sınıfları içerir.
"""

import os
import re
import json
import glob
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator, Union, Set

from aiquantr_tokenizer.data.sources.base_source import BaseDataSource

# Logger oluştur
logger = logging.getLogger(__name__)

# İsteğe bağlı bağımlılıklar
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas yüklü değil. CSV/Excel yükleme işlevleri sınırlı olacak.")


class LocalFileSource(BaseDataSource):
    """
    Yerel bir dosyadan veri yükleyen kaynak.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        encoding: str = "utf-8",
        delimiter: str = ",",
        jsonl_mode: bool = False,
        **kwargs
    ):
        """
        LocalFileSource başlatıcısı.
        
        Args:
            file_path: Dosya yolu
            file_type: Dosya türü (varsayılan: None - uzantıdan algıla)
            encoding: Dosya kodlaması
            delimiter: CSV dosyaları için ayırıcı karakter
            jsonl_mode: JSON dosyaları için JSONL formatı mı?
            **kwargs: BaseDataSource için ek parametreler
        """
        super().__init__(**kwargs)
        
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.delimiter = delimiter
        self.jsonl_mode = jsonl_mode
        
        # Dosya türünü belirleme
        if file_type:
            self.file_type = file_type.lower()
        else:
            # Uzantıdan tahmin et
            suffix = self.file_path.suffix.lower()
            if suffix in ['.txt', '.text', '.md']:
                self.file_type = 'text'
            elif suffix in ['.csv']:
                self.file_type = 'csv'
            elif suffix in ['.json', '.jsonl']:
                self.file_type = 'json'
                if suffix == '.jsonl':
                    self.jsonl_mode = True
            elif suffix in ['.xlsx', '.xls']:
                self.file_type = 'excel'
            else:
                self.file_type = 'text'  # Varsayılan olarak metin kabul et
        
        # Dosya varlık kontrolü
        if not self.file_path.exists():
            logger.error(f"{self.file_path} dosyası bulunamadı.")
        elif not self.file_path.is_file():
            logger.error(f"{self.file_path} bir dosya değil.")
    
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        Dosyadan veri yükleme.
        
        Yields:
            Dict[str, Any]: Yüklenen veri öğeleri
        """
        if not self.file_path.exists() or not self.file_path.is_file():
            return
        
        try:
            # Dosya tipine göre yükleme
            if self.file_type == 'text':
                yield from self._load_text_file()
            elif self.file_type == 'csv':
                yield from self._load_csv_file()
            elif self.file_type == 'json':
                yield from self._load_json_file()
            elif self.file_type == 'excel':
                yield from self._load_excel_file()
            else:
                logger.error(f"Desteklenmeyen dosya türü: {self.file_type}")
                
        except Exception as e:
            logger.error(f"{self.file_path} dosyası yüklenemedi: {str(e)}")
    
    def _load_text_file(self) -> Iterator[Dict[str, Any]]:
        """
        Metin dosyasını yükler.
        
        Yields:
            Dict[str, Any]: Metin içeriği
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                content = f.read()
            
            # İstatistikleri güncelle
            self.stats["total_samples"] += 1
            
            # Öğeyi işle
            item = {self.text_key: content}
            processed_item = self.process_item(item)
            
            if processed_item:
                yield processed_item
                
        except Exception as e:
            logger.error(f"Metin dosyası yükleme hatası: {str(e)}")
    
    def _load_csv_file(self) -> Iterator[Dict[str, Any]]:
        """
        CSV dosyasını yükler.
        
        Yields:
            Dict[str, Any]: CSV satırları
        """
        if not HAS_PANDAS:
            logger.error("CSV dosyası yüklemek için pandas gereklidir.")
            with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                content = f.read()
            
            # İstatistikleri güncelle
            self.stats["total_samples"] += 1
            
            # Öğeyi işle
            item = {self.text_key: content}
            processed_item = self.process_item(item)
            
            if processed_item:
                yield processed_item
            return
        
        try:
            # DataFrame olarak yükle
            df = pd.read_csv(self.file_path, delimiter=self.delimiter, encoding=self.encoding)
            
            # Metin sütunu kontrolü
            if self.text_key not in df.columns:
                first_text_col = None
                for col in df.columns:
                    if df[col].dtype == 'object':
                        first_text_col = col
                        break
                
                if first_text_col:
                    logger.warning(f"'{self.text_key}' sütunu bulunamadı. '{first_text_col}' kullanılıyor.")
                    self.text_key = first_text_col
                else:
                    logger.error(f"Metin sütunu bulunamadı: {df.columns.tolist()}")
                    return
            
            # Satırları işle
            count = 0
            for _, row in df.iterrows():
                # Limiti kontrol et
                if self.limit and count >= self.limit:
                    break
                
                # Satırı sözlüğe dönüştür
                item = row.to_dict()
                self.stats["total_samples"] += 1
                
                # Öğeyi işle
                processed_item = self.process_item(item)
                if processed_item:
                    yield processed_item
                    count += 1
                
        except Exception as e:
            logger.error(f"CSV dosyası yükleme hatası: {str(e)}")
    
    def _load_json_file(self) -> Iterator[Dict[str, Any]]:
        """
        JSON dosyasını yükler.
        
        Yields:
            Dict[str, Any]: JSON nesneleri
        """
        try:
            if self.jsonl_mode:
                # JSONL formatı - her satır ayrı bir JSON nesnesi
                with open(self.file_path, 'r', encoding=self.encoding) as f:
                    count = 0
                    for line in f:
                        # Limiti kontrol et
                        if self.limit and count >= self.limit:
                            break
                        
                        # Boş satırları atla
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            item = json.loads(line)
                            self.stats["total_samples"] += 1
                            
                            # Öğeyi işle
                            processed_item = self.process_item(item)
                            if processed_item:
                                yield processed_item
                                count += 1
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Geçersiz JSON satırı: {line[:50]}...")
                            
            else:
                # Standart JSON formatı
                with open(self.file_path, 'r', encoding=self.encoding) as f:
                    data = json.load(f)
                
                # Veri yapısını kontrol et
                if isinstance(data, dict):
                    # Tek bir nesne
                    self.stats["total_samples"] += 1
                    processed_item = self.process_item(data)
                    if processed_item:
                        yield processed_item
                        
                elif isinstance(data, list):
                    # Nesneler listesi
                    count = 0
                    for item in data:
                        # Limiti kontrol et
                        if self.limit and count >= self.limit:
                            break
                        
                        if isinstance(item, dict):
                            self.stats["total_samples"] += 1
                            processed_item = self.process_item(item)
                            if processed_item:
                                yield processed_item
                                count += 1
                        else:
                            logger.warning(f"JSON listesinde geçersiz öğe: {str(item)[:50]}...")
                else:
                    logger.error(f"Desteklenmeyen JSON yapısı: {type(data)}")
                    
        except Exception as e:
            logger.error(f"JSON dosyası yükleme hatası: {str(e)}")
    
    def _load_excel_file(self) -> Iterator[Dict[str, Any]]:
        """
        Excel dosyasını yükler.
        
        Yields:
            Dict[str, Any]: Excel satırları
        """
        if not HAS_PANDAS:
            logger.error("Excel dosyası yüklemek için pandas gereklidir.")
            return
        
        try:
            # DataFrame olarak yükle
            df = pd.read_excel(self.file_path)
            
            # Metin sütunu kontrolü
            if self.text_key not in df.columns:
                first_text_col = None
                for col in df.columns:
                    if df[col].dtype == 'object':
                        first_text_col = col
                        break
                
                if first_text_col:
                    logger.warning(f"'{self.text_key}' sütunu bulunamadı. '{first_text_col}' kullanılıyor.")
                    self.text_key = first_text_col
                else:
                    logger.error(f"Metin sütunu bulunamadı: {df.columns.tolist()}")
                    return
            
            # Satırları işle
            count = 0
            for _, row in df.iterrows():
                # Limiti kontrol et
                if self.limit and count >= self.limit:
                    break
                
                # Satırı sözlüğe dönüştür
                item = row.to_dict()
                self.stats["total_samples"] += 1
                
                # Öğeyi işle
                processed_item = self.process_item(item)
                if processed_item:
                    yield processed_item
                    count += 1
                
        except Exception as e:
            logger.error(f"Excel dosyası yükleme hatası: {str(e)}")


class LocalDirSource(BaseDataSource):
    """
    Yerel bir dizindeki dosyalardan veri yükleyen kaynak.
    """
    
    def __init__(
        self,
        directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        """
        LocalDirSource başlatıcısı.
        
        Args:
            directory: Dizin yolu
            pattern: Arama deseni (ör: "*.txt")
            recursive: Alt dizinlere de bak
            file_types: Yüklenecek dosya türleri (ör: [".txt", ".csv"])
            exclude_patterns: Dışlanacak desenler
            **kwargs: BaseDataSource için ek parametreler
        """
        super().__init__(**kwargs)
        
        self.directory = Path(directory)
        self.pattern = pattern
        self.recursive = recursive
        self.file_types = [t.lower() if not t.startswith('.') else t.lower() for t in file_types] if file_types else None
        self.exclude_patterns = exclude_patterns or []
        
        # Dizin kontrolü
        if not self.directory.exists():
            logger.error(f"{self.directory} dizini bulunamadı.")
        elif not self.directory.is_dir():
            logger.error(f"{self.directory} bir dizin değil.")
    
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        Dizindeki dosyalardan veri yükleme.
        
        Yields:
            Dict[str, Any]: Yüklenen veri öğeleri
        """
        if not self.directory.exists() or not self.directory.is_dir():
            return
        
        # Dosya listesini al
        if self.recursive:
            # Python 3.10 Path.rglob() kullan
            # '**/' karakterlerini kaldırıp ana dizini temel al
            pattern = self.pattern
            files = list(self.directory.rglob(pattern))
        else:
            files = list(self.directory.glob(self.pattern))
        
        # Dosya türü filtresi
        if self.file_types:
            files = [f for f in files if f.suffix.lower() in self.file_types]
        
        # Dışlama desenleri
        for exclude in self.exclude_patterns:
            files = [f for f in files if not re.search(exclude, str(f))]
        
        # Dosya sayısını logla
        logger.info(f"{len(files)} dosya bulundu: {self.directory}")
        
        # Her dosyayı işle
        count = 0
        for file_path in files:
            # Limiti kontrol et
            if self.limit and count >= self.limit:
                break
            
            # Dosya türünü belirle
            file_type = self._determine_file_type(file_path)
            
            # LocalFileSource kullanarak dosyayı yükle
            source = LocalFileSource(
                file_path=file_path,
                file_type=file_type,
                text_key=self.text_key,
                min_length=self.min_length,
                max_length=self.max_length
            )
            
            # Dosyadan veri yükle
            for item in source.load_data():
                yield item
                count += 1
                
                # Limiti kontrol et
                if self.limit and count >= self.limit:
                    break
            
            # İstatistikleri güncelle
            for key, value in source.stats.items():
                if key in self.stats:
                    self.stats[key] += value
    
    def _determine_file_type(self, file_path: Path) -> str:
        """
        Dosya türünü belirler.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            str: Dosya türü
        """
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.text', '.md']:
            return 'text'
        elif suffix in ['.csv']:
            return 'csv'
        elif suffix in ['.json']:
            return 'json'
        elif suffix in ['.jsonl']:
            return 'json'  # JSONL modunda
        elif suffix in ['.xlsx', '.xls']:
            return 'excel'
        else:
            return 'text'  # Varsayılan olarak metin