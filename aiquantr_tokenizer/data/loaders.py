"""
Veri yükleme modülleri.

Bu modül, tokenizer eğitimi için metin verilerini
çeşitli kaynaklardan yükleyen fonksiyonlar ve sınıflar sağlar.
"""

import os
import re
import json
import glob
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterable, Iterator, Tuple, Set

# Logger oluştur
logger = logging.getLogger(__name__)

# İsteğe bağlı bağımlılıklar
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas yüklü değil. CSV/Excel yükleme işlevleri sınırlı olacak.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests yüklü değil. Web verisi yükleme işlevleri devre dışı.")

try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    logger.warning("datasets yüklü değil. HuggingFace veri kümeleri yüklenemez.")
    

class TextDataLoader:
    """
    Metin verilerini çeşitli kaynaklardan yükleyen ana sınıf.
    """
    
    def __init__(
        self,
        text_key: str = "text",
        sample_size: Optional[int] = None,
        random_sample: bool = False,
        min_length: int = 0,
        max_length: Optional[int] = None
    ):
        """
        TextDataLoader başlatıcısı.
        
        Args:
            text_key: Metin alanının anahtarı (varsayılan: "text")
            sample_size: Yüklenecek maksimum örnek sayısı (varsayılan: None = tümü)
            random_sample: Rastgele örnekleme yap (varsayılan: False)
            min_length: Minimum metin uzunluğu (varsayılan: 0)
            max_length: Maksimum metin uzunluğu (varsayılan: None = sınırsız)
        """
        self.text_key = text_key
        self.sample_size = sample_size
        self.random_sample = random_sample
        self.min_length = min_length
        self.max_length = max_length
        
        # İstatistikler
        self.stats = {
            "loaded_files": 0,
            "loaded_samples": 0,
            "skipped_samples": 0,
            "skipped_length": 0,
            "total_characters": 0
        }
    
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
    
    def load_text_file(self, file_path: Union[str, Path]) -> Iterator[str]:
        """
        Metin dosyasını yükler.
        
        Args:
            file_path: Dosya yolu
            
        Yields:
            str: Metin içeriği
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # İstatistikleri güncelle
            self.stats["loaded_files"] += 1
            
            # Metni filtrele
            filtered_content = self.filter_text(content)
            if filtered_content:
                self.stats["loaded_samples"] += 1
                self.stats["total_characters"] += len(filtered_content)
                yield filtered_content
                
        except Exception as e:
            logger.error(f"{file_path} dosyası yüklenemedi: {str(e)}")
    
    def load_jsonl_file(self, file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """
        JSONL dosyasını yükler.
        
        Args:
            file_path: Dosya yolu
            
        Yields:
            Dict[str, Any]: JSON nesnesi
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Boş satırları atla
                        if line.strip():
                            item = json.loads(line)
                            
                            # Üst verileri atla
                            if "__meta__" in item:
                                continue
                                
                            # Metin alanını çıkar
                            if self.text_key in item:
                                text = item[self.text_key]
                                
                                # Filtrele
                                filtered_text = self.filter_text(text)
                                if filtered_text:
                                    # Filtrelenmiş metni geri ekle
                                    item[self.text_key] = filtered_text
                                    self.stats["loaded_samples"] += 1
                                    self.stats["total_characters"] += len(filtered_text)
                                    yield item
                                    
                    except json.JSONDecodeError:
                        logger.warning(f"{file_path} dosyasında geçersiz JSON satırı: {line[:50]}...")
                
            # İstatistikleri güncelle
            self.stats["loaded_files"] += 1
                
        except Exception as e:
            logger.error(f"{file_path} dosyası yüklenemedi: {str(e)}")
    
    def load_csv_file(
        self, 
        file_path: Union[str, Path],
        delimiter: str = ",",
        encoding: str = "utf-8",
        columns: Optional[List[str]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        CSV dosyasını yükler.
        
        Args:
            file_path: Dosya yolu
            delimiter: Ayırıcı karakter
            encoding: Dosya kodlaması
            columns: Yüklenecek sütunlar
            
        Yields:
            Dict[str, Any]: Kayıt
        """
        if not HAS_PANDAS:
            raise ImportError("CSV yükleme için pandas gereklidir.")
            
        try:
            # DataFrame olarak yükle
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
            
            # Sütun filtresi
            if columns:
                df = df[[col for col in columns if col in df.columns]]
            
            # Metin sütunu kontrolü
            if self.text_key not in df.columns:
                logger.error(f"{file_path} dosyasında '{self.text_key}' sütunu bulunamadı.")
                return
            
            # Kayıtlar üzerinde döngü
            for _, row in df.iterrows():
                # Kaydı sözlüğe dönüştür
                item = row.to_dict()
                
                # Metin filtresi
                text = item.get(self.text_key, "")
                filtered_text = self.filter_text(text)
                
                if filtered_text:
                    item[self.text_key] = filtered_text
                    self.stats["loaded_samples"] += 1
                    self.stats["total_characters"] += len(filtered_text)
                    yield item
            
            # İstatistikleri güncelle
            self.stats["loaded_files"] += 1
                
        except Exception as e:
            logger.error(f"{file_path} dosyası yüklenemedi: {str(e)}")
    
    def load_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "**/*.*",
        recursive: bool = True,
        file_types: Optional[List[str]] = None
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """
        Bir dizinden dosyaları yükler.
        
        Args:
            directory: Dizin yolu
            pattern: Arama deseni
            recursive: Alt dizinlere de bak
            file_types: Yüklenecek dosya türleri
            
        Yields:
            Union[str, Dict[str, Any]]: Yüklenen içerik
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"{directory} dizini bulunamadı.")
            return
        
        # Varsayılan dosya türleri
        if file_types is None:
            file_types = [".txt", ".csv", ".jsonl", ".json"]
        
        # Dosya listesini al
        glob_pattern = pattern if pattern else "**/*.*"
        if not recursive and "**" in glob_pattern:
            glob_pattern = glob_pattern.replace("**", "*")
            
        file_paths = list(directory.glob(glob_pattern))
        
        # Dosya türü filtresi
        if file_types:
            file_paths = [f for f in file_paths if f.is_file() and f.suffix.lower() in file_types]
        
        # Rastgele örnekleme
        if self.random_sample and self.sample_size and len(file_paths) > self.sample_size:
            file_paths = random.sample(file_paths, self.sample_size)
        
        # Dosyaları yükle
        sample_count = 0
        for file_path in file_paths:
            if not file_path.is_file():
                continue
                
            # Maksimum örnek sayısı kontrolü
            if self.sample_size and sample_count >= self.sample_size:
                break
            
            # Dosya türüne göre uygun yükleyiciyi seç
            suffix = file_path.suffix.lower()
            
            if suffix == ".txt":
                for content in self.load_text_file(file_path):
                    yield content
                    sample_count += 1
                    if self.sample_size and sample_count >= self.sample_size:
                        break
            
            elif suffix == ".jsonl":
                for item in self.load_jsonl_file(file_path):
                    yield item
                    sample_count += 1
                    if self.sample_size and sample_count >= self.sample_size:
                        break
                        
            elif suffix == ".json":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # JSON formatı kontrolü
                    if isinstance(data, dict) and self.text_key in data:
                        # Tek bir kayıt
                        text = data[self.text_key]
                        filtered_text = self.filter_text(text)
                        
                        if filtered_text:
                            data[self.text_key] = filtered_text
                            self.stats["loaded_samples"] += 1
                            self.stats["total_characters"] += len(filtered_text)
                            yield data
                            sample_count += 1
                    
                    elif isinstance(data, list):
                        # Kayıt listesi
                        for item in data:
                            if isinstance(item, dict) and self.text_key in item:
                                text = item[self.text_key]
                                filtered_text = self.filter_text(text)
                                
                                if filtered_text:
                                    item[self.text_key] = filtered_text
                                    self.stats["loaded_samples"] += 1
                                    self.stats["total_characters"] += len(filtered_text)
                                    yield item
                                    sample_count += 1
                                    
                                    if self.sample_size and sample_count >= self.sample_size:
                                        break
                    
                    self.stats["loaded_files"] += 1
                        
                except Exception as e:
                    logger.error(f"{file_path} JSON dosyası yüklenemedi: {str(e)}")
            
            elif suffix == ".csv":
                for item in self.load_csv_file(file_path):
                    yield item
                    sample_count += 1
                    if self.sample_size and sample_count >= self.sample_size:
                        break
    
    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        streaming: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """
        HuggingFace veri kümesinden veri yükler.
        
        Args:
            dataset_name: Veri kümesi adı
            split: Bölüm adı (varsayılan: "train")
            streaming: Akış modu (varsayılan: False)
            
        Yields:
            Dict[str, Any]: Yüklenen kayıt
        """
        if not HAS_HF_DATASETS:
            raise ImportError("HuggingFace veri kümeleri için 'datasets' paketi gereklidir.")
        
        try:
            # Veri kümesini yükle
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            
            # Metin anahtarı kontrolü
            if not streaming:
                if self.text_key not in dataset.column_names:
                    # Alternatif metin anahtarları dene
                    alt_keys = ['text', 'content', 'sentence', 'input_text']
                    for key in alt_keys:
                        if key in dataset.column_names:
                            logger.warning(f"'{self.text_key}' sütunu bulunamadı. '{key}' kullanılıyor.")
                            self.text_key = key
                            break
                    else:
                        logger.error(f"{dataset_name} veri kümesinde metin sütunu bulunamadı.")
                        return
            
            # Örneklere eriş
            sample_count = 0
            for item in dataset:
                if self.sample_size and sample_count >= self.sample_size:
                    break
                
                # Metin kontrolü
                if self.text_key in item:
                    text = item[self.text_key]
                    filtered_text = self.filter_text(text)
                    
                    if filtered_text:
                        item[self.text_key] = filtered_text
                        self.stats["loaded_samples"] += 1
                        self.stats["total_characters"] += len(filtered_text)
                        yield item
                        sample_count += 1
            
            # İstatistikleri güncelle
            self.stats["loaded_files"] += 1
                
        except Exception as e:
            logger.error(f"{dataset_name} veri kümesi yüklenemedi: {str(e)}")
    
    def load_from_web(
        self,
        urls: List[str],
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10
    ) -> Iterator[str]:
        """
        Web sayfalarından içerik yükler.
        
        Args:
            urls: URL listesi
            headers: HTTP başlıkları
            timeout: Zaman aşımı süresi
            
        Yields:
            str: Sayfa içeriği
        """
        if not HAS_REQUESTS:
            raise ImportError("Web yükleme için 'requests' paketi gereklidir.")
        
        sample_count = 0
        for url in urls:
            if self.sample_size and sample_count >= self.sample_size:
                break
                
            try:
                # Web isteği yap
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                
                # İçeriği çıkar
                content = response.text
                filtered_content = self.filter_text(content)
                
                if filtered_content:
                    self.stats["loaded_samples"] += 1
                    self.stats["total_characters"] += len(filtered_content)
                    yield filtered_content
                    sample_count += 1
                
                # İstatistikleri güncelle
                self.stats["loaded_files"] += 1
                    
            except Exception as e:
                logger.error(f"{url} adresinden veri yüklenemedi: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Yükleme istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistikler
        """
        return dict(self.stats)


def load_text_files(
    file_paths: List[Union[str, Path]],
    min_length: int = 0,
    encoding: str = "utf-8",
    errors: str = "replace"
) -> Iterator[str]:
    """
    Metin dosyalarını yükler.
    
    Args:
        file_paths: Dosya yolları
        min_length: Minimum dosya uzunluğu
        encoding: Dosya kodlaması
        errors: Hata işleme modu
        
    Yields:
        str: Dosya içeriği
    """
    for path in file_paths:
        try:
            with open(path, 'r', encoding=encoding, errors=errors) as f:
                content = f.read()
                
            if len(content) >= min_length:
                yield content
                
        except Exception as e:
            logger.error(f"{path} dosyası yüklenemedi: {str(e)}")


def load_sentences_from_text(
    text: str,
    min_length: int = 10,
    max_length: Optional[int] = None,
    split_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Metinden cümleleri ayırır.
    
    Args:
        text: Metin
        min_length: Minimum cümle uzunluğu
        max_length: Maksimum cümle uzunluğu
        split_patterns: Cümle ayırma desenleri
        
    Returns:
        List[str]: Cümleler listesi
    """
    if not text:
        return []
        
    # Varsayılan ayırma desenleri
    if not split_patterns:
        split_patterns = [r'(?<=[.!?])\s+']
    
    sentences = [text]
    
    # Her deseni uygula
    for pattern in split_patterns:
        new_sentences = []
        for sentence in sentences:
            parts = re.split(pattern, sentence)
            new_sentences.extend([p.strip() for p in parts if p.strip()])
        sentences = new_sentences
    
    # Uzunluk filtresi
    if min_length > 0:
        sentences = [s for s in sentences if len(s) >= min_length]
        
    if max_length:
        sentences = [s[:max_length] if len(s) > max_length else s for s in sentences]
    
    return sentences