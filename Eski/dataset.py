# aiquantr_tokenizer/data/dataset.py
"""
Tokenizer eğitimi için veri kümesi sınıfları.

Bu modül, çeşitli kaynaklardan verileri yüklemek ve işlemek için
temel veri kümesi sınıfları sağlar.
"""

import os
import random
import logging
import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Iterator, Callable, Tuple, Generator, Set

# Logger oluştur
logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """
    Tüm veri kümeleri için temel sınıf.
    
    Bu soyut temel sınıf, veri kümesi nesneleri için gereken minimum
    işlevselliği tanımlar.
    """
    
    def __init__(self):
        self.stats = {"total_samples": 0}
    
    @abstractmethod
    def __getitem__(self, idx: int) -> str:
        """
        Belirli bir indeksteki örneği döndürür.
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            str: Metin örneği
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Veri kümesindeki toplam örnek sayısını döndürür.
        
        Returns:
            int: Toplam örnek sayısı
        """
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """
        Veri kümesindeki örnekleri dolaşmak için bir yineleyici döndürür.
        
        Returns:
            Iterator[str]: Örnekler üzerinde yineleyici
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Veri kümesi istatistiklerini döndürür.
        
        Returns:
            Dict[str, Any]: İstatistikler sözlüğü
        """
        if "total_samples" not in self.stats:
            self.stats["total_samples"] = len(self)
        return self.stats
    
    def get_subset(self, start_idx: int, end_idx: Optional[int] = None) -> "BaseDataset":
        """
        Veri kümesinin bir alt kümesini döndürür.
        
        Args:
            start_idx: Başlangıç indeksi (dahil)
            end_idx: Bitiş indeksi (hariç) (varsayılan: None - veri kümesi sonuna kadar)
            
        Returns:
            BaseDataset: Veri kümesinin alt kümesi
            
        Raises:
            IndexError: İndeksler geçerli aralıkta değilse
        """
        if end_idx is None:
            end_idx = len(self)
            
        if start_idx < 0 or start_idx >= len(self):
            raise IndexError(f"Başlangıç indeksi ({start_idx}) geçerli aralıkta değil [0, {len(self)})")
        
        if end_idx < start_idx or end_idx > len(self):
            raise IndexError(f"Bitiş indeksi ({end_idx}) geçerli aralıkta değil [{start_idx}, {len(self)}]")
        
        return self._create_subset(start_idx, end_idx)
    
    @abstractmethod
    def _create_subset(self, start_idx: int, end_idx: int) -> "BaseDataset":
        """
        Alt sınıflar tarafından uygulanacak alt küme oluşturma yöntemi.
        
        Args:
            start_idx: Başlangıç indeksi (dahil)
            end_idx: Bitiş indeksi (hariç)
            
        Returns:
            BaseDataset: Veri kümesinin alt kümesi
        """
        pass
    
    def shuffle(self, seed: Optional[int] = None) -> "BaseDataset":
        """
        Veri kümesini karıştırır ve yeni bir örnek döndürür.
        
        Args:
            seed: Rastgele sayı üreteci için başlangıç değeri (varsayılan: None)
            
        Returns:
            BaseDataset: Karıştırılmış veri kümesi
        """
        # Bu metot alt sınıflar tarafından etkili bir şekilde uygulanabilir
        return self


class TextDataset(BaseDataset):
    """
    Metin verilerini içeren veri kümesi.
    
    Bu sınıf, bellekte tutulan veya gerektiğinde dosyalardan yüklenen
    düz metin verileri için bir veri kümesi sağlar.
    """
    
    def __init__(
        self, 
        texts: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        lazy_loading: bool = False
    ):
        """
        TextDataset sınıfı başlatıcısı.
        
        Args:
            texts: Metin örnekleri listesi (varsayılan: None)
            file_paths: Metin dosya yolları listesi (varsayılan: None)
            lazy_loading: Dosyaları ihtiyaç duyulduğunda yükle (varsayılan: False)
            
        Note:
            En az texts veya file_paths parametrelerinden biri sağlanmalıdır.
            
        Raises:
            ValueError: Her iki parametre de None ise
        """
        super().__init__()
        
        self._texts = texts or []
        self._file_paths = [Path(p) for p in file_paths] if file_paths else []
        self._lazy_loading = lazy_loading and bool(self._file_paths)
        
        if not self._texts and not self._file_paths:
            raise ValueError("En az texts veya file_paths parametrelerinden biri sağlanmalıdır.")
        
        # Eğer tembel yükleme etkinse, dosyaları şimdi yükleme
        if not self._lazy_loading and self._file_paths:
            for file_path in self._file_paths:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        self._texts.append(f.read())
                except Exception as e:
                    logger.error(f"Dosya okunamadı ({file_path}): {e}")
        
        # İstatistikleri güncelle
        self._update_stats()
    
    def _update_stats(self):
        """Veri kümesi istatistiklerini günceller."""
        self.stats.update({
            "total_samples": len(self),
            "memory_loaded_samples": len(self._texts),
            "lazy_loaded_files": len(self._file_paths) if self._lazy_loading else 0
        })
    
    def __getitem__(self, idx: int) -> str:
        """
        Belirli bir indeksteki metin örneğini döndürür.
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            str: Metin örneği
            
        Raises:
            IndexError: İndeks geçerli aralıkta değilse
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"İndeks ({idx}) geçerli aralıkta değil [0, {len(self)})")
        
        if not self._lazy_loading or idx < len(self._texts):
            return self._texts[idx]
        
        # Tembel yükleme için dosya oku
        file_idx = idx - len(self._texts)
        try:
            with open(self._file_paths[file_idx], "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Dosya okunamadı ({self._file_paths[file_idx]}): {e}")
            return ""  # Boş metin döndür
    
    def __len__(self) -> int:
        """
        Veri kümesindeki toplam örnek sayısını döndürür.
        
        Returns:
            int: Toplam örnek sayısı
        """
        return len(self._texts) + (len(self._file_paths) if self._lazy_loading else 0)
    
    def __iter__(self) -> Iterator[str]:
        """
        Veri kümesindeki örnekleri dolaşmak için bir yineleyici döndürür.
        
        Returns:
            Iterator[str]: Örnekler üzerinde yineleyici
        """
        # Önce belleğe yüklenmiş metinleri işle
        yield from self._texts
        
        # Ardından, tembel yükleme etkinse, dosyalardan oku
        if self._lazy_loading:
            for file_path in self._file_paths:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        yield f.read()
                except Exception as e:
                    logger.error(f"Dosya okunamadı ({file_path}): {e}")
    
    def _create_subset(self, start_idx: int, end_idx: int) -> "TextDataset":
        """
        Veri kümesinin bir alt kümesini oluşturur.
        
        Args:
            start_idx: Başlangıç indeksi (dahil)
            end_idx: Bitiş indeksi (hariç)
            
        Returns:
            TextDataset: Veri kümesinin alt kümesi
        """
        subset_texts = []
        subset_file_paths = []
        
        memory_text_count = len(self._texts)
        
        # Belleğe yüklenmiş metinleri işle
        if start_idx < memory_text_count:
            end_memory_idx = min(end_idx, memory_text_count)
            subset_texts = self._texts[start_idx:end_memory_idx]
            
        # Tembel yüklenen dosyaları işle
        if end_idx > memory_text_count and self._lazy_loading:
            start_file_idx = max(0, start_idx - memory_text_count)
            end_file_idx = end_idx - memory_text_count
            subset_file_paths = self._file_paths[start_file_idx:end_file_idx]
        
        return TextDataset(
            texts=subset_texts,
            file_paths=subset_file_paths,
            lazy_loading=self._lazy_loading
        )
    
    def shuffle(self, seed: Optional[int] = None) -> "TextDataset":
        """
        Veri kümesini karıştırır ve yeni bir örnek döndürür.
        
        Args:
            seed: Rastgele sayı üreteci için başlangıç değeri (varsayılan: None)
            
        Returns:
            TextDataset: Karıştırılmış veri kümesi
            
        Note:
            Tembel yükleme etkinse, önce tüm dosyaları belleğe yükler.
        """
        # Tohumu ayarla
        rng = random.Random(seed)
        
        # Tüm metinleri belleğe yükle
        all_texts = list(self._texts)
        
        if self._lazy_loading:
            for file_path in self._file_paths:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        all_texts.append(f.read())
                except Exception as e:
                    logger.error(f"Dosya okunamadı ({file_path}): {e}")
        
        # Karıştır
        rng.shuffle(all_texts)
        
        # Yeni veri kümesi döndür
        return TextDataset(texts=all_texts, lazy_loading=False)


class CodeDataset(TextDataset):
    """
    Kod verilerini içeren özel veri kümesi.
    
    Bu sınıf, kod dosyaları için ek meta verileri izleyen
    bir TextDataset uzantısıdır.
    """
    
    def __init__(
        self, 
        texts: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        languages: Optional[List[str]] = None,
        lazy_loading: bool = False,
        metadata: Optional[Dict[int, Dict[str, Any]]] = None
    ):
        """
        CodeDataset sınıfı başlatıcısı.
        
        Args:
            texts: Kod örnekleri listesi (varsayılan: None)
            file_paths: Kod dosya yolları listesi (varsayılan: None)
            languages: Her örneğin dili (varsayılan: None - dosya uzantılarından çıkarılır)
            lazy_loading: Dosyaları ihtiyaç duyulduğunda yükle (varsayılan: False)
            metadata: Her örnek için ek meta veriler (varsayılan: None)
            
        Note:
            En az texts veya file_paths parametrelerinden biri sağlanmalıdır.
        """
        super().__init__(texts=texts, file_paths=file_paths, lazy_loading=lazy_loading)
        
        self._metadata = metadata or {}
        
        # Dilleri tanımla
        self._languages = []
        
        if languages:
            self._languages = list(languages)
        else:
            # Dosya uzantılarından dilleri çıkar
            self._languages = ["unknown"] * len(self._texts)
            for i, file_path in enumerate(self._file_paths):
                extension = Path(file_path).suffix.lower()[1:]  # `.py` -> `py`
                if extension:
                    # Eğer tembel yükleme yapılıyorsa, gerektiğinde dilleri doldur
                    if self._lazy_loading:
                        if i < len(self._languages):
                            self._languages[i] = extension
                    else:
                        self._languages.append(extension)
        
        # İstatistikleri güncelle
        self._update_stats()
    
    def _update_stats(self):
        """Veri kümesi istatistiklerini günceller."""
        super()._update_stats()
        
        # Kod özel istatistikler
        language_counts = {}
        for lang in self._languages:
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        self.stats.update({
            "languages": language_counts,
            "metadata_count": len(self._metadata)
        })
    
    def get_language(self, idx: int) -> str:
        """
        Belirli bir indeksteki örneğin dilini döndürür.
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            str: Dil kodu veya tanımı
            
        Raises:
            IndexError: İndeks geçerli aralıkta değilse
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"İndeks ({idx}) geçerli aralıkta değil [0, {len(self)})")
        
        if idx < len(self._languages):
            return self._languages[idx]
        return "unknown"
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Belirli bir indeksteki örneğin meta verilerini döndürür.
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            Dict[str, Any]: Meta veriler
        """
        return self._metadata.get(idx, {})
    
    def _create_subset(self, start_idx: int, end_idx: int) -> "CodeDataset":
        """
        Veri kümesinin bir alt kümesini oluşturur.
        
        Args:
            start_idx: Başlangıç indeksi (dahil)
            end_idx: Bitiş indeksi (hariç)
            
        Returns:
            CodeDataset: Veri kümesinin alt kümesi
        """
        text_dataset = super()._create_subset(start_idx, end_idx)
        
        # Dilleri ve meta verileri alt kümeye aktarma
        subset_languages = self._languages[start_idx:min(end_idx, len(self._languages))]
        
        subset_metadata = {}
        for idx in range(start_idx, end_idx):
            if idx in self._metadata:
                subset_metadata[idx - start_idx] = self._metadata[idx]
        
        return CodeDataset(
            texts=text_dataset._texts,
            file_paths=text_dataset._file_paths,
            languages=subset_languages,
            lazy_loading=text_dataset._lazy_loading,
            metadata=subset_metadata
        )
    
    def shuffle(self, seed: Optional[int] = None) -> "CodeDataset":
        """
        Veri kümesini karıştırır ve yeni bir örnek döndürür.
        
        Args:
            seed: Rastgele sayı üreteci için başlangıç değeri (varsayılan: None)
            
        Returns:
            CodeDataset: Karıştırılmış veri kümesi
        """
        # Tohumu ayarla
        rng = random.Random(seed)
        
        # Tüm metinleri ve dilleri belleğe yükle
        all_texts = list(self._texts)
        all_languages = list(self._languages)
        all_metadata = {}
        
        if self._lazy_loading:
            for i, file_path in enumerate(self._file_paths):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        all_texts.append(f.read())
                    
                    # Dili ve meta verileri de ekle
                    if i + len(self._texts) < len(self._languages):
                        all_languages.append(self._languages[i + len(self._texts)])
                    else:
                        all_languages.append("unknown")
                    
                    if i + len(self._texts) in self._metadata:
                        all_metadata[len(all_texts) - 1] = self._metadata[i + len(self._texts)]
                        
                except Exception as e:
                    logger.error(f"Dosya okunamadı ({file_path}): {e}")
        
        # Karıştırmak için indeks listesi oluştur
        indices = list(range(len(all_texts)))
        rng.shuffle(indices)
        
        # Karıştırılmış dizileri oluştur
        shuffled_texts = [all_texts[i] for i in indices]
        shuffled_languages = [all_languages[i] if i < len(all_languages) else "unknown" for i in indices]
        
        shuffled_metadata = {}
        for new_idx, old_idx in enumerate(indices):
            if old_idx in all_metadata:
                shuffled_metadata[new_idx] = all_metadata[old_idx]
        
        # Yeni veri kümesi döndür
        return CodeDataset(
            texts=shuffled_texts, 
            languages=shuffled_languages,
            lazy_loading=False,
            metadata=shuffled_metadata
        )


class MixedDataset(BaseDataset):
    """
    Birden fazla veri kümesini birleştiren karma veri kümesi.
    
    Bu sınıf, çeşitli veri kümelerini (örn. metin ve kod)
    tek bir veri kümesinde birleştirir.
    """
    
    def __init__(
        self,
        datasets: List[BaseDataset],
        weights: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        MixedDataset sınıfı başlatıcısı.
        
        Args:
            datasets: Birleştirilecek veri kümeleri listesi
            weights: Her veri kümesi için örnekleme ağırlıkları (varsayılan: None - eşit)
            metadata: Ek meta veriler (varsayılan: None)
            
        Raises:
            ValueError: datasets boşsa veya weights geçerli değilse
        """
        super().__init__()
        
        if not datasets:
            raise ValueError("Veri kümeleri listesi boş olamaz.")
        
        self._datasets = datasets
        self._dataset_sizes = [len(ds) for ds in datasets]
        self._total_size = sum(self._dataset_sizes)
        self._dataset_offsets = list(itertools.accumulate([0] + self._dataset_sizes[:-1]))
        
        # Ağırlıkları kontrol et ve normalleştir
        if weights is not None:
            if len(weights) != len(datasets):
                raise ValueError(f"Ağırlıklar listesi ({len(weights)}) veri kümeleri listesi ({len(datasets)}) ile aynı uzunlukta olmalı.")
            if not all(w >= 0 for w in weights):
                raise ValueError("Tüm ağırlıklar sıfır veya pozitif olmalı.")
            if sum(weights) == 0:
                raise ValueError("En az bir ağırlık sıfırdan büyük olmalı.")
            
            # Ağırlıkları normalleştir
            total_weight = sum(weights)
            self._weights = [w / total_weight for w in weights]
        else:
            # Eşit ağırlıklar
            self._weights = [1.0 / len(datasets)] * len(datasets)
        
        self._metadata = metadata or {}
        
        # İstatistikleri güncelle
        self._update_stats()
    
    def _update_stats(self):
        """Veri kümesi istatistiklerini günceller."""
        dataset_counts = {}
        for i, ds in enumerate(self._datasets):
            dataset_name = ds.__class__.__name__
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + self._dataset_sizes[i]
        
        self.stats.update({
            "total_samples": self._total_size,
            "dataset_counts": dataset_counts,
            "weights": self._weights
        })
        
        # Alt veri kümelerinin istatistiklerini ekle
        for i, ds in enumerate(self._datasets):
            ds_stats = ds.get_stats()
            dataset_name = ds.__class__.__name__
            self.stats[f"{dataset_name}_{i}_stats"] = ds_stats
    
    def __getitem__(self, idx: int) -> str:
        """
        Belirli bir indeksteki örneği döndürür.
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            str: Metin veya kod örneği
            
        Raises:
            IndexError: İndeks geçerli aralıkta değilse
        """
        if idx < 0 or idx >= self._total_size:
            raise IndexError(f"İndeks ({idx}) geçerli aralıkta değil [0, {self._total_size})")
        
        # Doğru veri kümesini ve içindeki indeksi bul
        dataset_idx = 0
        local_idx = idx
        
        for i, offset in enumerate(self._dataset_offsets):
            if idx >= offset and idx < offset + self._dataset_sizes[i]:
                dataset_idx = i
                local_idx = idx - offset
                break
        
        return self._datasets[dataset_idx][local_idx]
    
    def __len__(self) -> int:
        """
        Veri kümesindeki toplam örnek sayısını döndürür.
        
        Returns:
            int: Toplam örnek sayısı
        """
        return self._total_size
    
    def __iter__(self) -> Iterator[str]:
        """
        Veri kümesindeki örnekleri dolaşmak için bir yineleyici döndürür.
        
        Returns:
            Iterator[str]: Örnekler üzerinde yineleyici
        """
        for dataset in self._datasets:
            yield from dataset
    
    def get_dataset_info(self, idx: int) -> Tuple[int, int]:
        """
        Belirli bir indeksteki örneğin hangi veri kümesinden geldiğini döndürür.
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            Tuple[int, int]: (veri kümesi indeksi, yerel indeks)
            
        Raises:
            IndexError: İndeks geçerli aralıkta değilse
        """
        if idx < 0 or idx >= self._total_size:
            raise IndexError(f"İndeks ({idx}) geçerli aralıkta değil [0, {self._total_size})")
        
        for i, offset in enumerate(self._dataset_offsets):
            if idx >= offset and idx < offset + self._dataset_sizes[i]:
                return i, idx - offset
                
        # Buraya ulaşmamalı
        raise RuntimeError("Dahili hata: Veri kümesi indeksi bulunamadı")
    
    def _create_subset(self, start_idx: int, end_idx: int) -> "MixedDataset":
        """
        Veri kümesinin bir alt kümesini oluşturur.
        
        Args:
            start_idx: Başlangıç indeksi (dahil)
            end_idx: Bitiş indeksi (hariç)
            
        Returns:
            MixedDataset: Veri kümesinin alt kümesi
        """
        subsets = []
        subset_weights = []
        
        # Her bir veri kümesinin alt kümesini oluştur
        for i, dataset in enumerate(self._datasets):
            ds_start = self._dataset_offsets[i]
            ds_end = ds_start + self._dataset_sizes[i]
            
            # Bu veri kümesi aralık içinde mi kontrol et
            if end_idx <= ds_start or start_idx >= ds_end:
                # Bu veri kümesi tamamen aralık dışında
                continue
            
            # Aralık içindeki parçayı al
            ds_subset_start = max(0, start_idx - ds_start)
            ds_subset_end = min(self._dataset_sizes[i], end_idx - ds_start)
            
            # Alt küme oluştur
            if ds_subset_end > ds_subset_start:
                subset = dataset.get_subset(ds_subset_start, ds_subset_end)
                subsets.append(subset)
                subset_weights.append(self._weights[i])
        
        # Alt küme ağırlıklarını normalleştir
        if subset_weights:
            total_weight = sum(subset_weights)
            subset_weights = [w / total_weight for w in subset_weights]
        
        return MixedDataset(datasets=subsets, weights=subset_weights)
    
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[str]:
        """
        Veri kümelerinin ağırlıklarına göre örnekler çeker.
        
        Args:
            n: Çekilecek örnek sayısı (varsayılan: 1)
            seed: Rastgele sayı üreteci için başlangıç değeri (varsayılan: None)
            
        Returns:
            List[str]: Örnekler listesi
        """
        # Tohumu ayarla
        rng = random.Random(seed)
        
        samples = []
        for _ in range(n):
            # Ağırlıklara göre bir veri kümesi seç
            dataset_idx = rng.choices(
                population=range(len(self._datasets)),
                weights=self._weights,
                k=1
            )[0]
            
            # Seçilen veri kümesinden rastgele bir örnek çek
            dataset = self._datasets[dataset_idx]
            if len(dataset) > 0:
                sample_idx = rng.randint(0, len(dataset) - 1)
                samples.append(dataset[sample_idx])
        
        return samples
    
    def shuffle(self, seed: Optional[int] = None) -> "MixedDataset":
        """
        Her bir alt veri kümesini karıştırır.
        
        Args:
            seed: Rastgele sayı üreteci için başlangıç değeri (varsayılan: None)
            
        Returns:
            MixedDataset: Karıştırılmış veri kümesi
        """
        shuffled_datasets = []
        
        for i, dataset in enumerate(self._datasets):
            # Farklı alt veri kümeleri için farklı tohumlar kullan
            ds_seed = None if seed is None else seed + i
            shuffled_datasets.append(dataset.shuffle(seed=ds_seed))
        
        return MixedDataset(
            datasets=shuffled_datasets,
            weights=self._weights
        )


class StreamingDataset(BaseDataset):
    """
    Çok büyük veri kümeleri için akış tabanlı veri kümesi.
    
    Bu sınıf, bellekte tutulamayacak kadar büyük veri kümeleri için
    akış tabanlı bir yaklaşım sağlar.
    """
    
    def __init__(
        self,
        generator_fn: Callable[[], Generator[str, None, None]],
        estimated_size: Optional[int] = None,
        buffer_size: int = 1000,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        StreamingDataset sınıfı başlatıcısı.
        
        Args:
            generator_fn: Veri örnekleri üreten bir generator fonksiyon
            estimated_size: Yaklaşık veri kümesi boyutu (varsayılan: None)
            buffer_size: Tampon boyutu (varsayılan: 1000)
            metadata: Ek meta veriler (varsayılan: None)
            
        Note:
            Bu veri kümesi türü, büyük veri kümeleri için bellek verimli
            bir şekilde çalışmak üzere tasarlanmıştır. Tüm örnekler asla
            bellekte tutulmaz.
        """
        super().__init__()
        
        self._generator_fn = generator_fn
        self._buffer_size = max(1, buffer_size)
        self._buffer = []
        self._buffer_start_idx = 0
        self._estimated_size = estimated_size
        self._metadata = metadata or {}
        
        # Akışı başlat ve tamponu doldur
        self._stream = self._generator_fn()
        self._fill_buffer()
        
        # İstatistikleri güncelle
        self._update_stats()
    
    def _update_stats(self):
        """Veri kümesi istatistiklerini günceller."""
        self.stats.update({
            "total_samples": self._estimated_size or "bilinmiyor",
            "buffer_size": self._buffer_size,
            "buffer_filled": len(self._buffer),
            "streaming": True
        })
    
    def _fill_buffer(self):
        """Tamponu doldurur."""
        while len(self._buffer) < self._buffer_size:
            try:
                item = next(self._stream)
                self._buffer.append(item)
            except StopIteration:
                # Akış tamamlandı
                break
    
    def __getitem__(self, idx: int) -> str:
        """
        Belirli bir indeksteki örneği döndürür.
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            str: Metin örneği
            
        Raises:
            IndexError: İndeks geçerli aralıkta değilse veya akış tamamlandıysa
        """
        if idx < self._buffer_start_idx:
            # İndeks tampon penceresinden önce
            raise IndexError(f"İndeks ({idx}) tampon penceresinden önce ({self._buffer_start_idx})")
        
        buffer_idx = idx - self._buffer_start_idx
        
        if buffer_idx < len(self._buffer):
            # İndeks tampon içinde
            return self._buffer[buffer_idx]
        
        # Tampon dışı, akışı ilerlet
        while buffer_idx >= len(self._buffer):
            try:
                item = next(self._stream)
                self._buffer.append(item)
                
                # Tampon çok büyürse, başını kırp
                if len(self._buffer) > self._buffer_size * 2:
                    items_to_remove = len(self._buffer) - self._buffer_size
                    self._buffer = self._buffer[items_to_remove:]
                    self._buffer_start_idx += items_to_remove
                    buffer_idx -= items_to_remove
                    
            except StopIteration:
                # Akış tamamlandı, istenen indeks kapsam dışı
                raise IndexError(f"İndeks ({idx}) akış sınırlarının dışında")
        
        return self._buffer[buffer_idx]
    
    def __len__(self) -> int:
        """
        Veri kümesindeki tahmini toplam örnek sayısını döndürür.
        
        Returns:
            int: Tahmini toplam örnek sayısı (bilinmiyorsa kayıtlı öğe sayısı)
        """
        if self._estimated_size is not None:
            return self._estimated_size
        
        # Belirli bir boyut yoksa, şu ana kadar görülen öğelerin sayısını kullan
        return self._buffer_start_idx + len(self._buffer)
    
    def __iter__(self) -> Iterator[str]:
        """
        Veri kümesindeki örnekleri dolaşmak için bir yineleyici döndürür.
        
        Bu metot tampon kullanmayı atlar ve doğrudan temel akışa erişir.
        
        Returns:
            Iterator[str]: Örnekler üzerinde yineleyici
        """
        # Yeni bir akış oluştur
        return self._generator_fn()
    
    def _create_subset(self, start_idx: int, end_idx: int) -> "BaseDataset":
        """
        Veri kümesinin bir alt kümesini oluşturur.
        
        Akış veri kümesinde bu metot, belirtilen aralıktaki
        örnekleri içeren bir TextDataset döndürür.
        
        Args:
            start_idx: Başlangıç indeksi (dahil)
            end_idx: Bitiş indeksi (hariç)
            
        Returns:
            TextDataset: Veri kümesinin alt kümesi
            
        Raises:
            IndexError: İndeks tampon penceresinden önceyse
        """
        if start_idx < self._buffer_start_idx:
            # İndeks tampon penceresinden önce
            raise IndexError(f"Başlangıç indeksi ({start_idx}) tampon penceresinden önce ({self._buffer_start_idx})")
        
        # İstenen aralıktaki öğeleri belleğe yükle
        subset_texts = []
        current_idx = start_idx
        
        while current_idx < end_idx:
            try:
                subset_texts.append(self[current_idx])
                current_idx += 1
            except IndexError:
                # Akış sınırına ulaşıldı
                break
        
        return TextDataset(texts=subset_texts)