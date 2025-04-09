"""
Veri dışa aktarma modülleri.

Bu modül, tokenizer eğitimi için hazırlanan verileri
çeşitli formatlarda dışa aktarmak için fonksiyonlar sağlar.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterable, Callable, BinaryIO

import numpy as np

# Logger oluştur
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas yüklü değil. DataFrame aktarımı kullanılamaz.")
    
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py yüklü değil. HDF5 formatı kullanılamaz.")


def export_jsonl(
    texts: Iterable[Union[str, Dict[str, Any]]],
    output_file: Union[str, Path],
    text_key: str = "text",
    append: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Verileri JSONL (her satırda bir JSON objesi) formatında dışa aktarır.
    
    Args:
        texts: Metin veya sözlük listesi
        output_file: Çıktı dosyası
        text_key: Metin anahtarı (sade metin için)
        append: Varolan dosyaya ekle (varsayılan: False)
        metadata: Üst veri bilgileri
        
    Returns:
        int: Aktarılan kayıt sayısı
    """
    mode = "a" if append else "w"
    count = 0
    
    with open(output_file, mode, encoding="utf-8") as f:
        # Üst veriler
        if metadata and not append:
            # Üst verileri ilk satıra özel alan olarak ekle
            meta_obj = {"__meta__": metadata}
            f.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")
        
        # Kayıtları yaz
        for item in texts:
            if isinstance(item, str):
                # Metin dizesini sözlüğe dönüştür
                obj = {text_key: item}
            else:
                # Zaten sözlük formatında
                obj = item
                
            # JSON satırı yaz
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    
    logger.info(f"{count} kayıt {output_file} dosyasına aktarıldı.")
    return count


def export_txt(
    texts: Iterable[str],
    output_file: Union[str, Path],
    append: bool = False,
    separator: str = "\n\n"
) -> int:
    """
    Verileri düz metin formatında dışa aktarır.
    
    Args:
        texts: Metin listesi
        output_file: Çıktı dosyası
        append: Varolan dosyaya ekle (varsayılan: False)
        separator: Metinler arası ayırıcı
        
    Returns:
        int: Aktarılan metin sayısı
    """
    mode = "a" if append else "w"
    count = 0
    
    with open(output_file, mode, encoding="utf-8") as f:
        first = not append
        
        for text in texts:
            if not first:
                f.write(separator)
            else:
                first = False
                
            f.write(text)
            count += 1
    
    logger.info(f"{count} metin {output_file} dosyasına aktarıldı.")
    return count


def export_csv(
    texts: Iterable[Union[str, Dict[str, Any]]],
    output_file: Union[str, Path],
    text_key: str = "text",
    delimiter: str = ",",
    columns: Optional[List[str]] = None,
    append: bool = False,
    add_header: bool = True
) -> int:
    """
    Verileri CSV formatında dışa aktarır.
    
    Args:
        texts: Metin veya sözlük listesi
        output_file: Çıktı dosyası
        text_key: Metin anahtarı (sade metin için)
        delimiter: CSV ayırıcı karakter
        columns: Sütun listesi (None = tüm sütunlar)
        append: Varolan dosyaya ekle
        add_header: Başlık satırı ekle (append=True iken False yapılabilir)
        
    Returns:
        int: Aktarılan kayıt sayısı
    """
    if not HAS_PANDAS:
        raise ImportError("CSV aktarımı için pandas gereklidir.")
    
    # Veriyi DataFrame'e dönüştür
    data = []
    for item in texts:
        if isinstance(item, str):
            data.append({text_key: item})
        else:
            data.append(item)
    
    df = pd.DataFrame(data)
    
    # Sütunları filtrele
    if columns:
        df = df[[col for col in columns if col in df.columns]]
    
    # CSV'ye aktar
    mode = 'a' if append else 'w'
    header = add_header
    
    df.to_csv(
        output_file,
        sep=delimiter,
        index=False,
        mode=mode,
        header=header,
        encoding='utf-8'
    )
    
    count = len(df)
    logger.info(f"{count} kayıt {output_file} dosyasına aktarıldı.")
    return count


def export_hdf5(
    texts: Iterable[Union[str, Dict[str, Any]]],
    output_file: Union[str, Path],
    dataset_name: str = "texts",
    text_key: str = "text",
    metadata: Optional[Dict[str, Any]] = None,
    compression: Optional[str] = "gzip",
    chunk_size: int = 1000
) -> int:
    """
    Verileri HDF5 formatında dışa aktarır.
    
    Args:
        texts: Metin veya sözlük listesi
        output_file: Çıktı dosyası
        dataset_name: HDF5 veri kümesi adı
        text_key: Metin anahtarı (sade metin için)
        metadata: Üst veri bilgileri
        compression: Sıkıştırma algoritması ("gzip", "lzf" veya None)
        chunk_size: Veri yığını boyutu
        
    Returns:
        int: Aktarılan kayıt sayısı
    """
    if not HAS_H5PY:
        raise ImportError("HDF5 aktarımı için h5py gereklidir.")
    
    # Önce verileri listeye dönüştür (hdf5 doğrudan iterator alamaz)
    data_list = []
    for item in texts:
        if isinstance(item, str):
            data_list.append(item)
        else:
            # Sözlük formatında, metin alanını çıkar
            data_list.append(item.get(text_key, ""))
    
    # Veri sayısı
    count = len(data_list)
    
    # Boş kontrolü
    if not count:
        logger.warning("Aktarılacak veri yok.")
        return 0
        
    # HDF5 dosyası oluştur
    with h5py.File(output_file, 'w') as f:
        # String veri seti oluştur
        dt = h5py.special_dtype(vlen=str)
        dataset = f.create_dataset(
            dataset_name,
            shape=(count,),
            dtype=dt,
            compression=compression,
            chunks=(min(chunk_size, count),)
        )
        
        # Verileri aktar
        dataset[:] = data_list
        
        # Üst verileri ekle
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    dataset.attrs[key] = value
                else:
                    # Karmaşık nesneleri JSON olarak depola
                    dataset.attrs[key] = json.dumps(value)
    
    logger.info(f"{count} kayıt {output_file} dosyasına aktarıldı.")
    return count


def split_dataset(
    texts: List[Union[str, Dict[str, Any]]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: Optional[int] = None
) -> Dict[str, List]:
    """
    Veri kümesini eğitim/doğrulama/test alt kümelerine böler.
    
    Args:
        texts: Veri listesi
        train_ratio: Eğitim kümesi oranı (varsayılan: 0.8)
        valid_ratio: Doğrulama kümesi oranı (varsayılan: 0.1)
        test_ratio: Test kümesi oranı (varsayılan: 0.1)
        shuffle: Karıştırma yap (varsayılan: True)
        random_seed: Rastgele tohum değeri
        
    Returns:
        Dict[str, List]: Bölünmüş veri kümeleri
    """
    # Oranların toplamı 1.0 olmalı
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        logger.warning(f"Oran toplamı 1.0 değil: {total_ratio}. Oranlar normalize edilecek.")
        train_ratio /= total_ratio
        valid_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Veri kopyası oluştur
    data_copy = texts.copy()
    
    # Karıştırma
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(data_copy)
    
    # Veri boyutu
    n = len(data_copy)
    train_size = int(n * train_ratio)
    valid_size = int(n * valid_ratio)
    
    # Bölme
    train_data = data_copy[:train_size]
    valid_data = data_copy[train_size:train_size + valid_size]
    test_data = data_copy[train_size + valid_size:]
    
    result = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }
    
    # Log
    logger.info(f"Veri kümesi bölündü: Eğitim={len(train_data)}, "
                f"Doğrulama={len(valid_data)}, Test={len(test_data)}")
    
    return result


def export_huggingface_format(
    texts: Iterable[Union[str, Dict[str, Any]]],
    output_dir: Union[str, Path],
    text_key: str = "text",
    train_valid_test: bool = True,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
    """
    Verileri HuggingFace veri kümesi formatında dışa aktarır.
    
    Args:
        texts: Metin veya sözlük listesi
        output_dir: Çıktı dizini
        text_key: Metin anahtarı (sade metin için)
        train_valid_test: Eğitim/doğrulama/test olarak böl
        train_ratio: Eğitim kümesi oranı
        valid_ratio: Doğrulama kümesi oranı
        test_ratio: Test kümesi oranı
        random_seed: Rastgele tohum değeri
        metadata: Üst veri bilgileri
        
    Returns:
        Dict[str, int]: Her alt küme için aktarılan kayıt sayısı
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tüm veriyi listeye dönüştür
    data = []
    for item in texts:
        if isinstance(item, str):
            data.append({text_key: item})
        else:
            data.append(item)
    
    # Dosya yolları
    dataset_files = {}
    
    if train_valid_test:
        # Verileri böl
        splits = split_dataset(
            data,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )
        
        # Her bölümü ayrı dosyaya aktar
        counts = {}
        for split_name, split_data in splits.items():
            if split_data:  # Boş değilse
                split_file = output_dir / f"{split_name}.jsonl"
                count = export_jsonl(
                    split_data, 
                    split_file,
                    text_key=text_key,
                    metadata=(metadata if split_name == "train" else None)
                )
                counts[split_name] = count
                dataset_files[split_name] = str(split_file)
    else:
        # Tamamını tek dosyaya aktar
        train_file = output_dir / "train.jsonl"
        count = export_jsonl(data, train_file, text_key=text_key, metadata=metadata)
        counts = {"train": count}
        dataset_files["train"] = str(train_file)
    
    # Dataset_info.json dosyası oluştur
    dataset_info = {
        "description": metadata.get("description", "Tokenizer eğitim veri kümesi"),
        "citation": metadata.get("citation", ""),
        "homepage": metadata.get("homepage", ""),
        "license": metadata.get("license", ""),
        "features": {
            text_key: {
                "dtype": "string",
                "_type": "Value"
            }
        },
        "splits": {
            name: {"name": name, "num_bytes": os.path.getsize(file), "num_examples": counts.get(name, 0)}
            for name, file in dataset_files.items()
        },
        "version": metadata.get("version", "1.0.0")
    }
    
    # Diğer alanlar
    if "features" in metadata:
        for feature_name, feature_type in metadata["features"].items():
            if feature_name != text_key:
                dataset_info["features"][feature_name] = {
                    "dtype": feature_type,
                    "_type": "Value"
                }
    
    # Dataset_info.json dosyasını yaz
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"HuggingFace veri kümesi {output_dir} dizinine aktarıldı.")
    return counts