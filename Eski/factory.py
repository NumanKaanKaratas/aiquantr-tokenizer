# aiquantr_tokenizer/data/factory.py
"""
Veri kümesi ve işleyici fabrikaları.

Bu modül, yapılandırma sözlüklerinden veri kümesi ve
işleyici nesneleri oluşturmaya yönelik yardımcı işlevler sağlar.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

from .dataset import BaseDataset, TextDataset, CodeDataset, MixedDataset, StreamingDataset
from .processors import BaseProcessor, TextProcessor, CodeProcessor, DuplicateRemover, ProcessingPipeline
from .loaders import (
    load_from_files,
    load_from_directory,
    load_from_huggingface,
    load_from_jsonl,
    load_from_csv,
    load_mixed_dataset
)

# Logger oluştur
logger = logging.getLogger(__name__)


def create_dataset_from_config(config: Dict[str, Any], base_path: Optional[Path] = None) -> BaseDataset:
    """
    Yapılandırmaya göre bir veri kümesi oluşturur.
    
    Args:
        config: Veri kümesi yapılandırması
        base_path: Yapılandırmadaki yolları çözümlemek için temel yol
        
    Returns:
        BaseDataset: Oluşturulan veri kümesi
        
    Raises:
        ValueError: Yapılandırma geçerli değilse
    """
    if not isinstance(config, dict):
        raise ValueError("Yapılandırma bir sözlük olmalıdır.")
    
    # Gerekli alanların varlığını kontrol et
    if "type" not in config:
        raise ValueError("Yapılandırmada 'type' alanı eksik.")
    
    dataset_type = config["type"].lower()
    
    # Yolları çözümle
    if base_path is not None:
        config = _resolve_paths_in_config(config, base_path)
    
    try:
        if dataset_type == "text":
            return _create_text_dataset(config)
            
        elif dataset_type == "code":
            return _create_code_dataset(config)
            
        elif dataset_type == "huggingface":
            return load_from_huggingface(
                dataset_name=config["dataset_name"],
                subset=config.get("subset"),
                split=config.get("split", "train"),
                text_column=config.get("text_column"),
                streaming=config.get("streaming", True),
                max_samples=config.get("max_samples"),
                cache_dir=config.get("cache_dir")
            )
            
        elif dataset_type == "jsonl":
            return load_from_jsonl(
                file_path=config["path"],
                text_field=config.get("text_field", "text"),
                max_samples=config.get("max_samples"),
                encoding=config.get("encoding", "utf-8")
            )
            
        elif dataset_type == "csv":
            return load_from_csv(
                file_path=config["path"],
                text_column=config.get("text_column"),
                encoding=config.get("encoding", "utf-8"),
                delimiter=config.get("delimiter", ","),
                max_samples=config.get("max_samples")
            )
            
        elif dataset_type == "directory":
            return load_from_directory(
                directory=config["path"],
                patterns=config.get("patterns"),
                recursive=config.get("recursive", True),
                exclude_dirs=config.get("exclude_dirs"),
                exclude_patterns=config.get("exclude_patterns"),
                max_files=config.get("max_files"),
                lazy_loading=config.get("lazy_loading", True),
                language_map=config.get("language_map")
            )
            
        elif dataset_type == "mixed":
            if "datasets" not in config:
                raise ValueError("Karma veri kümesi için 'datasets' alanı gereklidir.")
                
            return load_mixed_dataset(
                dataset_configs=config["datasets"],
                weights=config.get("weights")
            )
            
        else:
            raise ValueError(f"Bilinmeyen veri kümesi türü: {dataset_type}")
    
    except Exception as e:
        logger.error(f"Veri kümesi oluşturulurken hata: {e}")
        raise


def create_processor_from_config(config: Dict[str, Any]) -> BaseProcessor:
    """
    Yapılandırmaya göre bir işleyici oluşturur.
    
    Args:
        config: İşleyici yapılandırması
        
    Returns:
        BaseProcessor: Oluşturulan işleyici
        
    Raises:
        ValueError: Yapılandırma geçerli değilse
    """
    if not isinstance(config, dict):
        raise ValueError("Yapılandırma bir sözlük olmalıdır.")
    
    # Gerekli alanların varlığını kontrol et
    if "type" not in config:
        raise ValueError("Yapılandırmada 'type' alanı eksik.")
    
    processor_type = config["type"].lower()
    
    try:
        if processor_type == "text":
            return TextProcessor(
                lowercase=config.get("lowercase", False),
                fix_unicode=config.get("fix_unicode", True),
                normalize_unicode=config.get("normalize_unicode", True),
                normalize_whitespace=config.get("normalize_whitespace", True),
                replace_urls=config.get("replace_urls", False),
                replace_emails=config.get("replace_emails", False),
                replace_numbers=config.get("replace_numbers", False),
                replace_digits=config.get("replace_digits", False),
                replace_currency_symbols=config.get("replace_currency_symbols", False),
                remove_punct=config.get("remove_punct", False),
                remove_line_breaks=config.get("remove_line_breaks", False),
                fix_html=config.get("fix_html", True),
                min_text_length=config.get("min_text_length", 0),
                max_text_length=config.get("max_text_length"),
                custom_replacements=config.get("custom_replacements"),
                name=config.get("name")
            )
            
        elif processor_type == "code":
            return CodeProcessor(
                remove_comments=config.get("remove_comments", False),
                normalize_whitespace=config.get("normalize_whitespace", True),
                remove_docstrings=config.get("remove_docstrings", False),
                remove_string_literals=config.get("remove_string_literals", False),
                keep_indentation=config.get("keep_indentation", True),
                remove_shebang=config.get("remove_shebang", True),
                min_code_length=config.get("min_code_length", 0),
                max_code_length=config.get("max_code_length"),
                language_specific_rules=config.get("language_specific_rules"),
                name=config.get("name")
            )
            
        elif processor_type == "duplicate_remover":
            return DuplicateRemover(
                hash_method=config.get("hash_method", "exact"),
                min_similarity=config.get("min_similarity", 1.0),
                case_sensitive=config.get("case_sensitive", True),
                whitespace_sensitive=config.get("whitespace_sensitive", False),
                name=config.get("name")
            )
            
        elif processor_type == "pipeline":
            if "processors" not in config:
                raise ValueError("İşleme hattı için 'processors' alanı gereklidir.")
                
            # Alt işleyicileri oluştur
            processors = []
            for proc_config in config["processors"]:
                processors.append(create_processor_from_config(proc_config))
                
            return ProcessingPipeline(
                processors=processors,
                skip_empty=config.get("skip_empty", True),
                name=config.get("name")
            )
            
        else:
            raise ValueError(f"Bilinmeyen işleyici türü: {processor_type}")
    
    except Exception as e:
        logger.error(f"İşleyici oluşturulurken hata: {e}")
        raise


def _create_text_dataset(config: Dict[str, Any]) -> TextDataset:
    """
    Yapılandırmaya göre bir metin veri kümesi oluşturur.
    
    Args:
        config: Metin veri kümesi yapılandırması
        
    Returns:
        TextDataset: Oluşturulan metin veri kümesi
    """
    # Gerekli alanların varlığını kontrol et
    if "texts" not in config and "file_paths" not in config:
        raise ValueError("Metin veri kümesi için 'texts' veya 'file_paths' alanlarından biri gereklidir.")
    
    return TextDataset(
        texts=config.get("texts"),
        file_paths=config.get("file_paths"),
        lazy_loading=config.get("lazy_loading", False)
    )


def _create_code_dataset(config: Dict[str, Any]) -> CodeDataset:
    """
    Yapılandırmaya göre bir kod veri kümesi oluşturur.
    
    Args:
        config: Kod veri kümesi yapılandırması
        
    Returns:
        CodeDataset: Oluşturulan kod veri kümesi
    """
    # Gerekli alanların varlığını kontrol et
    if "texts" not in config and "file_paths" not in config:
        raise ValueError("Kod veri kümesi için 'texts' veya 'file_paths' alanlarından biri gereklidir.")
    
    return CodeDataset(
        texts=config.get("texts"),
        file_paths=config.get("file_paths"),
        languages=config.get("languages"),
        lazy_loading=config.get("lazy_loading", False),
        metadata=config.get("metadata")
    )


def _resolve_paths_in_config(config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
    """
    Yapılandırma sözlüğündeki yolları çözümler.
    
    Args:
        config: Yapılandırma sözlüğü
        base_path: Temel yol
        
    Returns:
        Dict[str, Any]: Çözümlenmiş yollarla güncellenen yapılandırma
    """
    result = dict(config)
    
    # Yol alanlarını kontrol et ve güncelle
    if "path" in result and isinstance(result["path"], str):
        result["path"] = str(base_path / result["path"])
    
    if "file_paths" in result and isinstance(result["file_paths"], list):
        result["file_paths"] = [str(base_path / path) for path in result["file_paths"]]
    
    # Alt yapılandırma alanlarını kontrol et
    if "datasets" in result and isinstance(result["datasets"], list):
        result["datasets"] = [_resolve_paths_in_config(ds, base_path) for ds in result["datasets"]]
    
    return result