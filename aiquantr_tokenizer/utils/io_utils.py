# aiquantr_tokenizer/utils/io_utils.py
"""
Dosya işlemleri için yardımcı fonksiyonlar.

Bu modül, dosya ve dizin işlemleri için gerekli fonksiyonları sağlar.
"""

import os
import json
import yaml
import csv
import gzip
import shutil
import tempfile
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO, Callable, Tuple, Iterator

# Logger oluştur
logger = logging.getLogger(__name__)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Dizinin var olduğundan emin olur, yoksa oluşturur.
    
    Args:
        directory: Oluşturulacak dizin yolu
        
    Returns:
        Path: Dizin yolu
        
    Raises:
        OSError: Dizin oluşturulamazsa
    """
    directory = Path(directory)
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Dizin oluşturuldu: {directory}")
        except OSError as e:
            logger.error(f"Dizin oluşturulamadı ({directory}): {e}")
            raise
    return directory


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Dosyanın var olup olmadığını kontrol eder.
    
    Args:
        file_path: Kontrol edilecek dosya yolu
        
    Returns:
        bool: Dosya varsa True, yoksa False
    """
    return Path(file_path).is_file()


def read_file(
    file_path: Union[str, Path], 
    encoding: str = "utf-8", 
    mode: str = "r"
) -> str:
    """
    Dosyayı okur ve içeriğini döndürür.
    
    Args:
        file_path: Okunacak dosya yolu
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        mode: Okuma modu (varsayılan: "r")
        
    Returns:
        str: Dosya içeriği
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        IOError: Dosya okunamazsa
    """
    try:
        with open(file_path, mode, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Dosya bulunamadı: {file_path}")
        raise
    except IOError as e:
        logger.error(f"Dosya okunamadı ({file_path}): {e}")
        raise


def write_file(
    file_path: Union[str, Path], 
    content: str, 
    encoding: str = "utf-8", 
    mode: str = "w",
    create_dirs: bool = True
) -> None:
    """
    Dosyaya içerik yazar.
    
    Args:
        file_path: Yazılacak dosya yolu
        content: Yazılacak içerik
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        mode: Yazma modu (varsayılan: "w")
        create_dirs: Gerekirse dizinleri oluştur (varsayılan: True)
        
    Raises:
        IOError: Dosya yazılamazsa
    """
    try:
        if create_dirs:
            ensure_dir(Path(file_path).parent)
            
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
            logger.debug(f"Dosyaya yazıldı: {file_path}")
    except IOError as e:
        logger.error(f"Dosyaya yazılamadı ({file_path}): {e}")
        raise


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    JSON dosyasını yükler ve Python nesnesine dönüştürür.
    
    Args:
        file_path: JSON dosyası yolu
        
    Returns:
        Dict[str, Any]: JSON içeriği
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        json.JSONDecodeError: JSON dosyası doğru formatta değilse
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON dosyası bulunamadı: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON dosyası geçerli değil ({file_path}): {e}")
        raise


def save_json(
    file_path: Union[str, Path], 
    data: Dict[str, Any], 
    indent: int = 2, 
    ensure_ascii: bool = False
) -> None:
    """
    Python nesnesini JSON dosyasına kaydeder.
    
    Args:
        file_path: JSON dosyası yolu
        data: Kaydedilecek veri
        indent: Girinti seviyesi (varsayılan: 2)
        ensure_ascii: ASCII olmayan karakterleri escape et (varsayılan: False)
        
    Raises:
        IOError: Dosya yazılamazsa
    """
    try:
        ensure_dir(Path(file_path).parent)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logger.debug(f"JSON dosyası kaydedildi: {file_path}")
    except IOError as e:
        logger.error(f"JSON dosyası yazılamadı ({file_path}): {e}")
        raise


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    YAML dosyasını yükler ve Python nesnesine dönüştürür.
    
    Args:
        file_path: YAML dosyası yolu
        
    Returns:
        Dict[str, Any]: YAML içeriği
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        yaml.YAMLError: YAML dosyası doğru formatta değilse
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"YAML dosyası bulunamadı: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML dosyası geçerli değil ({file_path}): {e}")
        raise


def save_yaml(
    file_path: Union[str, Path], 
    data: Dict[str, Any], 
    default_flow_style: bool = False
) -> None:
    """
    Python nesnesini YAML dosyasına kaydeder.
    
    Args:
        file_path: YAML dosyası yolu
        data: Kaydedilecek veri
        default_flow_style: Akış stili kullanılsın mı (varsayılan: False)
        
    Raises:
        IOError: Dosya yazılamazsa
    """
    try:
        ensure_dir(Path(file_path).parent)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data, 
                f, 
                default_flow_style=default_flow_style, 
                allow_unicode=True
            )
            logger.debug(f"YAML dosyası kaydedildi: {file_path}")
    except IOError as e:
        logger.error(f"YAML dosyası yazılamadı ({file_path}): {e}")
        raise


def copy_file(
    src: Union[str, Path], 
    dst: Union[str, Path], 
    create_dirs: bool = True
) -> None:
    """
    Dosyayı bir konumdan diğerine kopyalar.
    
    Args:
        src: Kaynak dosya yolu
        dst: Hedef dosya yolu
        create_dirs: Gerekirse hedef dizinleri oluştur (varsayılan: True)
        
    Raises:
        FileNotFoundError: Kaynak dosya bulunamazsa
        IOError: Kopyalama işlemi başarısız olursa
    """
    try:
        if create_dirs:
            ensure_dir(Path(dst).parent)
            
        shutil.copy2(src, dst)
        logger.debug(f"Dosya kopyalandı: {src} -> {dst}")
    except FileNotFoundError:
        logger.error(f"Kaynak dosya bulunamadı: {src}")
        raise
    except IOError as e:
        logger.error(f"Dosya kopyalanamadı ({src} -> {dst}): {e}")
        raise


def delete_file(
    file_path: Union[str, Path], 
    ignore_missing: bool = True
) -> bool:
    """
    Dosyayı siler.
    
    Args:
        file_path: Silinecek dosya yolu
        ignore_missing: Dosya yoksa hata verme (varsayılan: True)
        
    Returns:
        bool: Dosya başarıyla silindiyse True
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa ve ignore_missing False ise
        IOError: Silme işlemi başarısız olursa
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Dosya silindi: {file_path}")
            return True
        elif not ignore_missing:
            logger.error(f"Silinecek dosya bulunamadı: {file_path}")
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        return False
    except IOError as e:
        logger.error(f"Dosya silinemedi ({file_path}): {e}")
        raise


def list_files(
    directory: Union[str, Path], 
    pattern: str = "*", 
    recursive: bool = False
) -> List[Path]:
    """
    Belirtilen dizindeki dosyaları listeler.
    
    Args:
        directory: Listelenecek dizin
        pattern: Dosya deseni (glob formatı) (varsayılan: "*")
        recursive: Alt dizinleri de dolaş (varsayılan: False)
        
    Returns:
        List[Path]: Dosya yolları listesi
        
    Raises:
        FileNotFoundError: Dizin bulunamazsa
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Dizin bulunamadı: {directory}")
            raise FileNotFoundError(f"Dizin bulunamadı: {directory}")
        
        if recursive:
            return list(directory.glob(f"**/{pattern}"))
        else:
            return list(directory.glob(pattern))
    except Exception as e:
        logger.error(f"Dosyalar listelenirken hata ({directory}): {e}")
        raise


def create_temp_file(
    prefix: str = "aiquantr_tokenizer_", 
    suffix: str = "", 
    content: Optional[str] = None
) -> Path:
    """
    Geçici dosya oluşturur.
    
    Args:
        prefix: Dosya adı öneki (varsayılan: "aiquantr_tokenizer_")
        suffix: Dosya uzantısı (varsayılan: "")
        content: Dosyaya yazılacak içerik (varsayılan: None)
        
    Returns:
        Path: Oluşturulan geçici dosyanın yolu
    """
    try:
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        
        if content is not None:
            write_file(path, content)
            
        logger.debug(f"Geçici dosya oluşturuldu: {path}")
        return Path(path)
    except IOError as e:
        logger.error(f"Geçici dosya oluşturulamadı: {e}")
        raise


def create_temp_dir(prefix: str = "aiquantr_tokenizer_") -> Path:
    """
    Geçici dizin oluşturur.
    
    Args:
        prefix: Dizin adı öneki (varsayılan: "aiquantr_tokenizer_")
        
    Returns:
        Path: Oluşturulan geçici dizinin yolu
    """
    try:
        path = tempfile.mkdtemp(prefix=prefix)
        logger.debug(f"Geçici dizin oluşturuldu: {path}")
        return Path(path)
    except IOError as e:
        logger.error(f"Geçici dizin oluşturulamadı: {e}")
        raise


def safe_write_file(
    file_path: Union[str, Path], 
    content: str, 
    encoding: str = "utf-8", 
    mode: str = "w",
    create_dirs: bool = True
) -> None:
    """
    Dosyaya içeriği güvenli bir şekilde yazar (önce geçici bir dosyaya, sonra atomik taşıma).
    
    Bu işlev, yazma sırasında oluşabilecek hatalara karşı koruma sağlar.
    
    Args:
        file_path: Yazılacak dosya yolu
        content: Yazılacak içerik
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        mode: Yazma modu (varsayılan: "w")
        create_dirs: Gerekirse dizinleri oluştur (varsayılan: True)
        
    Raises:
        IOError: Dosya yazılamazsa
    """
    file_path = Path(file_path)
    if create_dirs:
        ensure_dir(file_path.parent)
    
    # Geçici dosya oluştur
    temp_file = create_temp_file(suffix=f".{file_path.name}")
    
    try:
        # İçeriği geçici dosyaya yaz
        write_file(temp_file, content, encoding, mode, create_dirs=False)
        
        # Atomik bir şekilde taşı (güvenli bir şekilde değiştir)
        shutil.move(str(temp_file), str(file_path))
        logger.debug(f"Dosya güvenli bir şekilde yazıldı: {file_path}")
    except Exception as e:
        # Hata durumunda geçici dosyayı temizle
        try:
            if temp_file.exists():
                temp_file.unlink()
        except:
            pass
        logger.error(f"Dosya güvenli bir şekilde yazılamadı ({file_path}): {e}")
        raise


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Dosyanın boyutunu byte cinsinden döndürür.
    
    Args:
        file_path: Dosya yolu
        
    Returns:
        int: Dosya boyutu (byte)
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
    """
    try:
        return Path(file_path).stat().st_size
    except FileNotFoundError:
        logger.error(f"Dosya bulunamadı: {file_path}")
        raise


def get_file_modification_time(file_path: Union[str, Path]) -> float:
    """
    Dosyanın son değiştirilme zamanını döndürür.
    
    Args:
        file_path: Dosya yolu
        
    Returns:
        float: Son değiştirilme zamanı (epoch zamanı)
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
    """
    try:
        return Path(file_path).stat().st_mtime
    except FileNotFoundError:
        logger.error(f"Dosya bulunamadı: {file_path}")
        raise


def read_chunks(
    file_path: Union[str, Path], 
    chunk_size: int = 1024 * 1024, 
    binary: bool = False
) -> Callable[[], Union[str, bytes]]:
    """
    Dosyayı parçalar halinde okumak için bir generator döndürür.
    
    Args:
        file_path: Okunacak dosya yolu
        chunk_size: Her seferinde okunacak bayt sayısı (varsayılan: 1MB)
        binary: İkili modda oku (varsayılan: False)
        
    Returns:
        Callable[[], Union[str, bytes]]: Generator fonksiyon
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        IOError: Dosya okunamazsa
    """
    mode = "rb" if binary else "r"
    encoding = None if binary else "utf-8"
    
    def generator():
        try:
            with open(file_path, mode, encoding=encoding) as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except FileNotFoundError:
            logger.error(f"Dosya bulunamadı: {file_path}")
            raise
        except IOError as e:
            logger.error(f"Dosya parçalar halinde okunamadı ({file_path}): {e}")
            raise
            
    return generator


def count_lines(file_path: Union[str, Path]) -> int:
    """
    Dosyadaki satır sayısını döndürür.
    
    Args:
        file_path: Dosya yolu
        
    Returns:
        int: Satır sayısı
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        logger.error(f"Dosya bulunamadı: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Satırlar sayılırken hata ({file_path}): {e}")
        raise

# Veri yükleme işlevleri (eksik işlevler ekleniyor)
def extract_texts (
    path: Union[str, Path],
    format: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    **kwargs
) -> List[str]:
    """
    Veri dosyası veya dizininden metinleri yükler.
    
    Args:
        path: Veri dosyası veya dizini yolu
        format: Veri formatı (varsayılan: None - otomatik tespit)
        limit: Yüklenecek örnek sayısı sınırı (varsayılan: None - tümü)
        shuffle: Veriyi karıştır (varsayılan: False)
        seed: Rastgele başlangıç değeri (varsayılan: 42)
        **kwargs: Format-spesifik ek parametreler
        
    Returns:
        List[str]: Metin listesi
        
    Raises:
        ValueError: Veri yüklenemezse
    """
    # Yol nesnesine dönüştür
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"Veri yolu bulunamadı: {path}")
    
    # Format otomatik tespiti
    if format is None:
        if path.is_file():
            if path.suffix.lower() in [".json", ".jsonl"]:
                format = "json"
            elif path.suffix.lower() in [".txt", ".text"]:
                format = "text"
            elif path.suffix.lower() in [".csv"]:
                format = "csv"
            elif path.suffix.lower() in [".yaml", ".yml"]:
                format = "yaml"
        else:
            # Dizin - metin dosyalarını tara
            format = "text_dir"
    
    # Veriyi yükle
    texts = []
    
    if format == "json":
        texts = extract_texts_from_json(path, **kwargs)  
    elif format == "text":
        texts = extract_texts_from_file(path, **kwargs) 
    elif format == "text_dir":
        texts = extract_texts_from_directory(path, **kwargs)  
    elif format == "csv":
        texts = extract_texts_from_csv(path, **kwargs)  
    elif format == "yaml":
        texts = extract_texts_from_yaml(path, **kwargs)
    else:
        raise ValueError(f"Bilinmeyen veri formatı: {format}")
    
    # Veriyi karıştır
    if shuffle:
        # Rastgele başlangıç değerini ayarla
        random_state = random.Random(seed)
        random_state.shuffle(texts)
    
    # Limiti uygula
    if limit is not None and len(texts) > limit:
        texts = texts[:limit]
    
    return texts


def extract_texts_from_json (
    path: Union[str, Path], 
    text_field: Optional[str] = None,
    encoding: str = "utf-8",
    **kwargs
) -> List[str]:
    """
    JSON veya JSONL dosyasından metinleri yükler.
    
    Args:
        path: JSON dosyası yolu
        text_field: JSON nesnesindeki metin alanı (varsayılan: None - tüm değer)
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        **kwargs: Ek parametreler
        
    Returns:
        List[str]: Metin listesi
        
    Raises:
        ValueError: JSON verisi yüklenemezse
    """
    path = Path(path)
    texts = []
    
    try:
        # JSONL dosyası
        if path.suffix.lower() == ".jsonl":
            with open(path, 'r', encoding=encoding) as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    item = json.loads(line)
                    if text_field:
                        if isinstance(item, dict) and text_field in item:
                            value = item[text_field]
                            if isinstance(value, str):
                                texts.append(value)
                    else:
                        texts.append(json.dumps(item, ensure_ascii=False))
        # Normal JSON dosyası
        else:
            with open(path, 'r', encoding=encoding) as f:
                data = json.load(f)
                
                # Liste ise
                if isinstance(data, list):
                    for item in data:
                        if text_field:
                            if isinstance(item, dict) and text_field in item:
                                value = item[text_field]
                                if isinstance(value, str):
                                    texts.append(value)
                        else:
                            if isinstance(item, str):
                                texts.append(item)
                            else:
                                texts.append(json.dumps(item, ensure_ascii=False))
                # Tek nesne ise
                else:
                    if text_field:
                        if isinstance(data, dict) and text_field in data:
                            value = data[text_field]
                            if isinstance(value, str):
                                texts.append(value)
                            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                                texts.extend(value)
                    else:
                        texts.append(json.dumps(data, ensure_ascii=False))
                        
        return texts
    except Exception as e:
        logger.error(f"JSON verisi yüklenemedi ({path}): {e}")
        raise ValueError(f"JSON verisi yüklenemedi: {e}")


def extract_texts_from_file (
    path: Union[str, Path],
    encoding: str = "utf-8",
    by_line: bool = True,
    strip: bool = True,
    skip_empty: bool = True,
    **kwargs
) -> List[str]:
    """
    Metin dosyasından metinleri yükler.
    
    Args:
        path: Metin dosyası yolu
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        by_line: Her satırı ayrı bir metin olarak yükle (varsayılan: True)
        strip: Metinlerdeki baş ve sondaki boşlukları temizle (varsayılan: True)
        skip_empty: Boş metinleri atla (varsayılan: True)
        **kwargs: Ek parametreler
        
    Returns:
        List[str]: Metin listesi
        
    Raises:
        ValueError: Metin verisi yüklenemezse
    """
    path = Path(path)
    texts = []
    
    try:
        # Sıkıştırılmış dosyalar için
        if path.suffix.lower() == ".gz":
            open_func = gzip.open
        else:
            open_func = open
        
        # Dosyayı aç ve oku
        with open_func(path, 'rt', encoding=encoding) as f:
            if by_line:
                for line in f:
                    if strip:
                        line = line.strip()
                    
                    if skip_empty and not line:
                        continue
                        
                    texts.append(line)
            else:
                content = f.read()
                if strip:
                    content = content.strip()
                
                if not (skip_empty and not content):
                    texts.append(content)
                    
        return texts
    except Exception as e:
        logger.error(f"Metin verisi yüklenemedi ({path}): {e}")
        raise ValueError(f"Metin verisi yüklenemedi: {e}")


def extract_texts_from_directory (
    directory: Union[str, Path],
    pattern: str = "*.txt",
    recursive: bool = True,
    encoding: str = "utf-8",
    by_file: bool = True,
    **kwargs
) -> List[str]:
    """
    Bir dizindeki metin dosyalarından metinleri yükler.
    
    Args:
        directory: Dizin yolu
        pattern: Dosya deseni (glob formatı) (varsayılan: "*.txt")
        recursive: Alt dizinleri de dolaş (varsayılan: True)
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        by_file: Her dosyayı bir metin olarak yükle (varsayılan: True)
        **kwargs: load_text_data için ek parametreler
        
    Returns:
        List[str]: Metin listesi
        
    Raises:
        ValueError: Dizin veya dosyalar yüklenemezse
    """
    directory = Path(directory)
    texts = []
    
    try:
        files = list_files(directory, pattern, recursive)
        
        for file_path in files:
            if by_file:
                with open(file_path, 'rt', encoding=encoding) as f:
                    content = f.read()
                    texts.append(content)
            else:
                file_texts = extract_texts_from_file (file_path, encoding=encoding, **kwargs)
                texts.extend(file_texts)
                
        return texts
    except Exception as e:
        logger.error(f"Dizindeki metin dosyaları yüklenemedi ({directory}): {e}")
        raise ValueError(f"Dizindeki metin dosyaları yüklenemedi: {e}")


def extract_texts_from_csv (
    path: Union[str, Path],
    text_column: Optional[Union[str, int]] = None,
    delimiter: str = ",",
    has_header: bool = True,
    encoding: str = "utf-8",
    **kwargs
) -> List[str]:
    """
    CSV dosyasından metinleri yükler.
    
    Args:
        path: CSV dosyası yolu
        text_column: Metin sütunu adı veya indeksi (varsayılan: None - tüm satır)
        delimiter: Sütun ayırıcı (varsayılan: ",")
        has_header: Başlık satırı var mı? (varsayılan: True)
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        **kwargs: Ek parametreler
        
    Returns:
        List[str]: Metin listesi
        
    Raises:
        ValueError: CSV verisi yüklenemezse
    """
    path = Path(path)
    texts = []
    
    try:
        with open(path, 'r', encoding=encoding, newline='') as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                # Sütun adı belirtilmişse
                if text_column is not None and isinstance(text_column, str):
                    for row in reader:
                        if text_column in row:
                            texts.append(row[text_column])
                # Tüm satırı kullan
                else:
                    for row in reader:
                        texts.append(delimiter.join(row.values()))
            else:
                reader = csv.reader(f, delimiter=delimiter)
                
                # Sütun indeksi belirtilmişse
                if text_column is not None and isinstance(text_column, int):
                    for row in reader:
                        if len(row) > text_column:
                            texts.append(row[text_column])
                # Tüm satırı kullan
                else:
                    for row in reader:
                        texts.append(delimiter.join(row))
                
        return texts
    except Exception as e:
        logger.error(f"CSV verisi yüklenemedi ({path}): {e}")
        raise ValueError(f"CSV verisi yüklenemedi: {e}")


def extract_texts_from_yaml (
    path: Union[str, Path],
    text_field: Optional[str] = None,
    encoding: str = "utf-8",
    **kwargs
) -> List[str]:
    """
    YAML dosyasından metinleri yükler.
    
    Args:
        path: YAML dosyası yolu
        text_field: YAML nesnesindeki metin alanı (varsayılan: None - tüm değer)
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        **kwargs: Ek parametreler
        
    Returns:
        List[str]: Metin listesi
        
    Raises:
        ValueError: YAML verisi yüklenemezse
    """
    path = Path(path)
    texts = []
    
    try:
        with open(path, 'r', encoding=encoding) as f:
            data = yaml.safe_load(f)
            
            # Liste ise
            if isinstance(data, list):
                for item in data:
                    if text_field:
                        if isinstance(item, dict) and text_field in item:
                            value = item[text_field]
                            if isinstance(value, str):
                                texts.append(value)
                    else:
                        if isinstance(item, str):
                            texts.append(item)
                        else:
                            texts.append(yaml.dump(item, allow_unicode=True))
            # Tek nesne ise
            else:
                if text_field:
                    if isinstance(data, dict) and text_field in data:
                        value = data[text_field]
                        if isinstance(value, str):
                            texts.append(value)
                        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                            texts.extend(value)
                else:
                    texts.append(yaml.dump(data, allow_unicode=True))
                    
        return texts
    except Exception as e:
        logger.error(f"YAML verisi yüklenemedi ({path}): {e}")
        raise ValueError(f"YAML verisi yüklenemedi: {e}")


def split_data(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Veri setini eğitim, doğrulama ve test setlerine böler.
    
    Args:
        texts: Metin listesi
        train_ratio: Eğitim seti oranı (varsayılan: 0.8)
        val_ratio: Doğrulama seti oranı (varsayılan: 0.1)
        test_ratio: Test seti oranı (varsayılan: 0.1)
        shuffle: Bölmeden önce karıştır (varsayılan: True)
        seed: Rastgele başlangıç değeri (varsayılan: 42)
        
    Returns:
        Dict[str, List[str]]: Bölünmüş veri setleri
        
    Raises:
        ValueError: Oranlar geçerli değilse veya veri boşsa
    """
    # Giriş doğrulama
    if not texts:
        raise ValueError("Boş veri seti")
        
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Küçük yuvarlama hataları için tolerans
        raise ValueError(f"Oranların toplamı 1 olmalı: {total_ratio}")
    
    # Veriyi karıştır
    if shuffle:
        random_state = random.Random(seed)
        texts_copy = texts.copy()
        random_state.shuffle(texts_copy)
    else:
        texts_copy = texts
    
    # Bölme indekslerini hesapla
    n = len(texts_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Veriyi böl
    train_texts = texts_copy[:train_end]
    val_texts = texts_copy[train_end:val_end]
    test_texts = texts_copy[val_end:]
    
    return {
        "train": train_texts,
        "val": val_texts,
        "test": test_texts
    }


def save_texts(
    texts: List[str],
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    mode: str = "w",
    delimiter: str = "\n"
) -> None:
    """
    Metin listesini dosyaya kaydeder.
    
    Args:
        texts: Metin listesi
        file_path: Çıkış dosyası yolu
        encoding: Dosya kodlaması (varsayılan: "utf-8")
        mode: Yazma modu (varsayılan: "w")
        delimiter: Metin ayırıcı (varsayılan: "\n")
        
    Raises:
        IOError: Dosya yazılamazsa
    """
    try:
        ensure_dir(Path(file_path).parent)
        with open(file_path, mode, encoding=encoding) as f:
            f.write(delimiter.join(texts))
        logger.debug(f"Metinler dosyaya kaydedildi: {file_path} ({len(texts)} metin)")
    except IOError as e:
        logger.error(f"Metinler dosyaya yazılamadı ({file_path}): {e}")
        raise


def count_tokens(text: str, tokenizer=None) -> int:
    """
    Metindeki token sayısını hesaplar.
    
    Args:
        text: Sayılacak metin
        tokenizer: Kullanılacak tokenizer (varsayılan: None - basit boşluk tabanlı)
        
    Returns:
        int: Token sayısı
    """
    if tokenizer:
        return len(tokenizer.encode(text))
    
    # Basit boşluk tabanlı tokenizasyon
    return len(text.split())