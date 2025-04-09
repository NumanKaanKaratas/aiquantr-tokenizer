"""
Veri işleme paketi.

Bu paket, token eğitimi için veri hazırlama, temizleme,
filtreleme, analiz etme ve dışa aktarma işlevleri sağlar.
"""

from aiquantr_tokenizer.data.cleaner import clean_text, clean_code, clean_file, clean_batch
from aiquantr_tokenizer.data.filter import TextFilter, filter_dataset, remove_similar_samples
from aiquantr_tokenizer.data.loaders import TextDataLoader, load_text_files, load_sentences_from_text
from aiquantr_tokenizer.data.analyzer import DatasetAnalyzer, compute_text_stats, estimate_tokenizer_vocabulary_size
from aiquantr_tokenizer.data.exporter import (
    export_jsonl, export_txt, export_csv, export_hdf5, 
    export_huggingface_format, split_dataset
)

__all__ = [
    # Temizleme
    "clean_text", "clean_code", "clean_file", "clean_batch",
    
    # Filtreleme
    "TextFilter", "filter_dataset", "remove_similar_samples",
    
    # Yükleme
    "TextDataLoader", "load_text_files", "load_sentences_from_text",
    
    # Analiz
    "DatasetAnalyzer", "compute_text_stats", "estimate_tokenizer_vocabulary_size",
    
    # Dışa aktarma
    "export_jsonl", "export_txt", "export_csv", "export_hdf5",
    "export_huggingface_format", "split_dataset"
]