# aiquantr_tokenizer/tokenizers/evaluation.py
"""
Tokenizer değerlendirme işlevleri.

Bu modül, tokenizer'ların çeşitli metriklerle 
değerlendirilmesi için işlevler sağlar.
"""

import time
import logging
import math
import statistics
from collections import Counter
from typing import List, Dict, Any, Union, Optional, Set, Tuple, Callable

import numpy as np

from .base import BaseTokenizer

# Logger oluştur
logger = logging.getLogger(__name__)

# Varsayılan olarak hesaplanacak metrikler
DEFAULT_METRICS = [
    "vocabulary_coverage",
    "compression_ratio",
    "token_frequency",
    "encoding_speed",
    "rare_tokens",
    "token_length",
    "reconstruction_accuracy"
]


def evaluate_tokenizer(
    tokenizer: BaseTokenizer,
    texts: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Tokenizer'ı verilen metinler üzerinde değerlendirir.
    
    Args:
        tokenizer: Değerlendirilecek tokenizer
        texts: Değerlendirme metinleri
        metrics: Hesaplanacak metrikler (varsayılan: None - tüm metrikler)
        
    Returns:
        Dict[str, Any]: Değerlendirme sonuçları
    """
    start_time = time.time()
    
    # Hangi metriklerin hesaplanacağını belirle
    if metrics is None:
        metrics_to_compute = DEFAULT_METRICS
    else:
        metrics_to_compute = metrics
        
    logger.info(f"{len(texts)} metin üzerinde {len(metrics_to_compute)} metrik hesaplanacak")
    
    # Tüm metinleri tokenize et
    tokenized_data = []
    token_ids_data = []
    
    for text in texts:
        # Token dizilerini ve ID'lerini hesapla
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        
        tokenized_data.append(tokens)
        token_ids_data.append(token_ids)
    
    # Sonuçları depolayacak sözlük
    results = {
        "num_samples": len(texts),
        "vocab_size": tokenizer.get_vocab_size(),
        "evaluation_time": time.time() - start_time
    }
    
    # İstenen metrikleri hesapla
    if "vocabulary_coverage" in metrics_to_compute:
        results["vocabulary_coverage"] = calculate_vocabulary_coverage(tokenizer, tokenized_data)
        
    if "compression_ratio" in metrics_to_compute:
        results["compression_ratio"] = calculate_compression_ratio(texts, tokenized_data)
        
    if "token_frequency" in metrics_to_compute:
        results["token_frequency"] = calculate_token_frequency(tokenized_data)
        
    if "encoding_speed" in metrics_to_compute:
        results["encoding_speed"] = calculate_encoding_speed(tokenizer, texts)
        
    if "rare_tokens" in metrics_to_compute:
        results["rare_tokens"] = calculate_rare_tokens(tokenized_data)
        
    if "token_length" in metrics_to_compute:
        results["token_length"] = calculate_token_length(tokenized_data)
        
    if "reconstruction_accuracy" in metrics_to_compute:
        results["reconstruction_accuracy"] = calculate_reconstruction_accuracy(tokenizer, texts)
    
    # Ek metrikler
    if "sequence_length" in metrics_to_compute:
        results["sequence_length"] = calculate_sequence_length(token_ids_data)
        
    if "special_token_usage" in metrics_to_compute:
        results["special_token_usage"] = calculate_special_token_usage(tokenizer, token_ids_data)
        
    if "unk_token_analysis" in metrics_to_compute:
        results["unk_token_analysis"] = calculate_unk_token_analysis(tokenizer, texts)
        
    # Özel metrikler
    custom_metrics = [m for m in metrics_to_compute if m not in results and m != "all"]
    for metric_name in custom_metrics:
        metric_func = globals().get(f"calculate_{metric_name}")
        if metric_func and callable(metric_func):
            try:
                results[metric_name] = metric_func(tokenizer, texts, tokenized_data, token_ids_data)
            except Exception as e:
                logger.warning(f"'{metric_name}' metriği hesaplanırken hata: {e}")
    
    results["total_evaluation_time"] = time.time() - start_time
    logger.info(f"Değerlendirme tamamlandı ({results['total_evaluation_time']:.2f}s)")
    return results


def calculate_vocabulary_coverage(
    tokenizer: BaseTokenizer,
    tokenized_data: List[List[str]]
) -> Dict[str, Any]:
    """
    Sözlük kapsamını hesaplar.
    
    Args:
        tokenizer: Değerlendirilen tokenizer
        tokenized_data: Tokenize edilmiş metinler
        
    Returns:
        Dict[str, Any]: Sözlük kapsam metrikleri
    """
    # Tüm tokenları düzleştir
    all_tokens = [token for tokens in tokenized_data for token in tokens]
    
    # Benzersiz tokenları hesapla
    unique_tokens = set(all_tokens)
    
    # Sözlük büyüklüğü
    vocab_size = tokenizer.get_vocab_size()
    
    # Sözlükteki tokenlar
    vocab = tokenizer.get_vocab()
    
    # Benzersiz token sayısının sözlük boyutuna oranı
    coverage_ratio = len(unique_tokens) / vocab_size if vocab_size > 0 else 0
    
    return {
        "unique_tokens": len(unique_tokens),
        "vocab_size": vocab_size,
        "coverage_ratio": coverage_ratio,
        "most_common_tokens": Counter(all_tokens).most_common(10)
    }


def calculate_compression_ratio(
    texts: List[str],
    tokenized_data: List[List[str]]
) -> Dict[str, float]:
    """
    Sıkıştırma oranını hesaplar.
    
    Args:
        texts: Orijinal metinler
        tokenized_data: Tokenize edilmiş metinler
        
    Returns:
        Dict[str, float]: Sıkıştırma oranı metrikleri
    """
    # Toplam karakter sayısı
    total_chars = sum(len(text) for text in texts)
    
    # Toplam token sayısı
    total_tokens = sum(len(tokens) for tokens in tokenized_data)
    
    # Sıkıştırma oranı: karakter / token
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    
    # Ortalama token başına karakter
    chars_per_token = compression_ratio
    
    # Karakter-token oranları
    char_to_token_ratios = [
        len(text) / len(tokens) if tokens else 0
        for text, tokens in zip(texts, tokenized_data)
    ]
    
    # Medyan ve standart sapma
    median_ratio = statistics.median(char_to_token_ratios) if char_to_token_ratios else 0
    std_ratio = statistics.stdev(char_to_token_ratios) if len(char_to_token_ratios) > 1 else 0
    
    return {
        "compression_ratio": compression_ratio,
        "chars_per_token": chars_per_token,
        "median_ratio": median_ratio,
        "std_ratio": std_ratio,
        "min_ratio": min(char_to_token_ratios) if char_to_token_ratios else 0,
        "max_ratio": max(char_to_token_ratios) if char_to_token_ratios else 0
    }


def calculate_token_frequency(tokenized_data: List[List[str]]) -> Dict[str, Any]:
    """
    Token frekans dağılımını hesaplar.
    
    Args:
        tokenized_data: Tokenize edilmiş metinler
        
    Returns:
        Dict[str, Any]: Token frekans metrikleri
    """
    # Tüm tokenları düzleştir
    all_tokens = [token for tokens in tokenized_data for token in tokens]
    
    # Frekans sayacı
    token_counter = Counter(all_tokens)
    
    # Toplam token sayısı
    total_tokens = len(all_tokens)
    
    # Benzersiz token sayısı
    unique_tokens = len(token_counter)
    
    # En yaygın tokenlar
    most_common = token_counter.most_common(20)
    
    # Token çeşitliliği (benzersiz / toplam)
    token_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Frekans istatistikleri
    frequencies = list(token_counter.values())
    
    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "token_diversity": token_diversity,
        "most_common": most_common,
        "mean_frequency": statistics.mean(frequencies) if frequencies else 0,
        "median_frequency": statistics.median(frequencies) if frequencies else 0,
        "max_frequency": max(frequencies) if frequencies else 0,
        "min_frequency": min(frequencies) if frequencies else 0
    }


def calculate_encoding_speed(
    tokenizer: BaseTokenizer,
    texts: List[str],
    num_repeats: int = 3
) -> Dict[str, float]:
    """
    Kodlama hızını hesaplar.
    
    Args:
        tokenizer: Değerlendirilen tokenizer
        texts: Kodlanacak metinler
        num_repeats: Tekrar sayısı (varsayılan: 3)
        
    Returns:
        Dict[str, float]: Kodlama hızı metrikleri
    """
    # Metinleri birleştir
    total_chars = sum(len(text) for text in texts)
    
    # Kodlama hızını ölç
    encode_times = []
    decode_times = []
    
    for _ in range(num_repeats):
        # Kodlama
        encode_start = time.time()
        token_ids = [tokenizer.encode(text) for text in texts]
        encode_end = time.time()
        
        # Kod çözme
        decode_start = time.time()
        _ = [tokenizer.decode(ids) for ids in token_ids]
        decode_end = time.time()
        
        encode_times.append(encode_end - encode_start)
        decode_times.append(decode_end - decode_start)
    
    # Ortalamaları hesapla
    avg_encode_time = statistics.mean(encode_times)
    avg_decode_time = statistics.mean(decode_times)
    
    # Saniyede karakter olarak hızı hesapla
    encode_speed = total_chars / avg_encode_time if avg_encode_time > 0 else 0
    decode_speed = total_chars / avg_decode_time if avg_decode_time > 0 else 0
    
    return {
        "encode_speed_chars_per_sec": encode_speed,
        "decode_speed_chars_per_sec": decode_speed,
        "avg_encode_time_sec": avg_encode_time,
        "avg_decode_time_sec": avg_decode_time,
        "total_chars": total_chars
    }


def calculate_rare_tokens(tokenized_data: List[List[str]]) -> Dict[str, Any]:
    """
    Nadir kullanılan tokenları analiz eder.
    
    Args:
        tokenized_data: Tokenize edilmiş metinler
        
    Returns:
        Dict[str, Any]: Nadir token analizi
    """
    # Tüm tokenları düzleştir
    all_tokens = [token for tokens in tokenized_data for token in tokens]
    
    # Frekans sayacı
    token_counter = Counter(all_tokens)
    
    # Toplam token sayısı
    total_tokens = len(all_tokens)
    
    # Nadir tokenları filtrele (frekans <= 2)
    rare_tokens = {token: freq for token, freq in token_counter.items() if freq <= 2}
    
    # Tekil görülen tokenlar
    singleton_tokens = {token: freq for token, freq in token_counter.items() if freq == 1}
    
    return {
        "total_tokens": total_tokens,
        "num_rare_tokens": len(rare_tokens),
        "num_singleton_tokens": len(singleton_tokens),
        "rare_token_ratio": len(rare_tokens) / len(token_counter) if token_counter else 0,
        "singleton_ratio": len(singleton_tokens) / len(token_counter) if token_counter else 0,
        "rare_token_examples": list(rare_tokens.keys())[:10]
    }


def calculate_token_length(tokenized_data: List[List[str]]) -> Dict[str, Any]:
    """
    Token uzunluğu istatistiklerini hesaplar.
    
    Args:
        tokenized_data: Tokenize edilmiş metinler
        
    Returns:
        Dict[str, Any]: Token uzunluğu metrikleri
    """
    # Token uzunluklarını topla
    token_lengths = [len(token) for tokens in tokenized_data for token in tokens]
    
    # Boş liste kontrolü
    if not token_lengths:
        return {
            "mean_length": 0,
            "median_length": 0,
            "min_length": 0,
            "max_length": 0,
            "std_length": 0
        }
    
    # İstatistikler
    mean_length = statistics.mean(token_lengths)
    median_length = statistics.median(token_lengths)
    min_length = min(token_lengths)
    max_length = max(token_lengths)
    std_length = statistics.stdev(token_lengths) if len(token_lengths) > 1 else 0
    
    # Uzunluk dağılımı
    length_distribution = Counter(token_lengths)
    
    return {
        "mean_length": mean_length,
        "median_length": median_length,
        "min_length": min_length,
        "max_length": max_length,
        "std_length": std_length,
        "length_distribution": sorted(length_distribution.items())
    }


def calculate_reconstruction_accuracy(
    tokenizer: BaseTokenizer,
    texts: List[str]
) -> Dict[str, Any]:
    """
    Yeniden yapılandırma doğruluğunu hesaplar.
    
    Args:
        tokenizer: Değerlendirilen tokenizer
        texts: Orijinal metinler
        
    Returns:
        Dict[str, Any]: Yeniden yapılandırma metrikleri
    """
    perfect_matches = 0
    char_errors = 0
    total_chars = 0
    
    reconstructed_texts = []
    error_samples = []
    
    for i, text in enumerate(texts):
        # Kodla ve kod çöz
        token_ids = tokenizer.encode(text)
        reconstructed = tokenizer.decode(token_ids)
        
        reconstructed_texts.append(reconstructed)
        
        # Karakter sayıları
        text_chars = len(text)
        total_chars += text_chars
        
        # Tam eşleşme kontrolü
        if text == reconstructed:
            perfect_matches += 1
        else:
            # Levenshtein mesafesi hesapla
            distance = compute_levenshtein_distance(text, reconstructed)
            char_errors += distance
            
            # İlk 5 hata örneğini kaydet
            if len(error_samples) < 5:
                error_samples.append({
                    "index": i,
                    "original": text[:100] + ("..." if len(text) > 100 else ""),
                    "reconstructed": reconstructed[:100] + ("..." if len(reconstructed) > 100 else ""),
                    "distance": distance
                })
    
    # Karakter hata oranı
    char_error_rate = char_errors / total_chars if total_chars > 0 else 0
    
    # Tam eşleşme oranı
    perfect_match_rate = perfect_matches / len(texts) if texts else 0
    
    return {
        "perfect_match_rate": perfect_match_rate,
        "char_error_rate": char_error_rate,
        "perfect_matches": perfect_matches,
        "total_samples": len(texts),
        "error_samples": error_samples
    }


def calculate_sequence_length(token_ids_data: List[List[int]]) -> Dict[str, Any]:
    """
    Kodlanmış dizilerin uzunluk istatistiklerini hesaplar.
    
    Args:
        token_ids_data: Kodlanmış token ID'leri
        
    Returns:
        Dict[str, Any]: Dizi uzunluğu metrikleri
    """
    # Dizi uzunluklarını topla
    sequence_lengths = [len(seq) for seq in token_ids_data]
    
    # Boş liste kontrolü
    if not sequence_lengths:
        return {
            "mean_length": 0,
            "median_length": 0,
            "min_length": 0,
            "max_length": 0,
            "std_length": 0
        }
    
    # İstatistikler
    mean_length = statistics.mean(sequence_lengths)
    median_length = statistics.median(sequence_lengths)
    min_length = min(sequence_lengths)
    max_length = max(sequence_lengths)
    std_length = statistics.stdev(sequence_lengths) if len(sequence_lengths) > 1 else 0
    
    # Yüzdelik dilimleri hesapla
    percentiles = {
        "95th": np.percentile(sequence_lengths, 95) if sequence_lengths else 0,
        "99th": np.percentile(sequence_lengths, 99) if sequence_lengths else 0,
        "90th": np.percentile(sequence_lengths, 90) if sequence_lengths else 0
    }
    
    return {
        "mean_length": mean_length,
        "median_length": median_length,
        "min_length": min_length,
        "max_length": max_length,
        "std_length": std_length,
        "percentiles": percentiles
    }


def calculate_special_token_usage(
    tokenizer: BaseTokenizer,
    token_ids_data: List[List[int]]
) -> Dict[str, Any]:
    """
    Özel token kullanım istatistiklerini hesaplar.
    
    Args:
        tokenizer: Değerlendirilen tokenizer
        token_ids_data: Kodlanmış token ID'leri
        
    Returns:
        Dict[str, Any]: Özel token kullanımı metrikleri
    """
    # Özel token ID'lerini belirle
    special_token_ids = {}
    for token_type, token in tokenizer.special_tokens.items():
        try:
            token_id = tokenizer.token_to_id(token)
            special_token_ids[token_type] = token_id
        except Exception:
            pass
            
    # Kullanım sayılarını hesapla
    usage_counts = {token_type: 0 for token_type in special_token_ids}
    total_tokens = 0
    
    for seq in token_ids_data:
        total_tokens += len(seq)
        
        for token_type, token_id in special_token_ids.items():
            usage_counts[token_type] += seq.count(token_id)
    
    # Kullanım oranları
    usage_rates = {
        token_type: count / total_tokens if total_tokens > 0 else 0
        for token_type, count in usage_counts.items()
    }
    
    return {
        "usage_counts": usage_counts,
        "usage_rates": usage_rates,
        "total_tokens": total_tokens
    }


def calculate_unk_token_analysis(
    tokenizer: BaseTokenizer,
    texts: List[str]
) -> Dict[str, Any]:
    """
    Bilinmeyen token kullanımını analiz eder.
    
    Args:
        tokenizer: Değerlendirilen tokenizer
        texts: Orijinal metinler
        
    Returns:
        Dict[str, Any]: Bilinmeyen token analizi
    """
    # Bilinmeyen token ID'sini al
    unk_token = tokenizer.special_tokens.get("unk_token", "[UNK]")
    unk_id = tokenizer.token_to_id(unk_token)
    
    total_unk = 0
    total_tokens = 0
    texts_with_unk = 0
    
    # UNK içeren metin örnekleri
    unk_examples = []
    
    for idx, text in enumerate(texts):
        # Metni tokenize et
        token_ids = tokenizer.encode(text)
        total_tokens += len(token_ids)
        
        # UNK sayısını hesapla
        unk_count = token_ids.count(unk_id)
        total_unk += unk_count
        
        if unk_count > 0:
            texts_with_unk += 1
            
            # İlk 5 örneği kaydet
            if len(unk_examples) < 5:
                tokens = tokenizer.tokenize(text)
                unk_examples.append({
                    "index": idx,
                    "text": text[:100] + ("..." if len(text) > 100 else ""),
                    "unk_count": unk_count,
                    "unk_rate": unk_count / len(token_ids) if token_ids else 0,
                    "tokens": tokens[:20] + (["..."] if len(tokens) > 20 else [])
                })
    
    # UNK oranları
    unk_rate = total_unk / total_tokens if total_tokens > 0 else 0
    texts_with_unk_rate = texts_with_unk / len(texts) if texts else 0
    
    return {
        "total_unk": total_unk,
        "total_tokens": total_tokens,
        "unk_rate": unk_rate,
        "texts_with_unk": texts_with_unk,
        "texts_with_unk_rate": texts_with_unk_rate,
        "unk_examples": unk_examples
    }


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """
    İki dizi arasındaki Levenshtein mesafesini hesaplar.
    
    Args:
        s1: Birinci dizi
        s2: İkinci dizi
        
    Returns:
        int: Levenshtein mesafesi
    """
    # Diziler boşsa, farklı uzunluktaki diziyi dönüştürmek için
    # gereken işlem sayısını döndür
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    # Uzunluklar
    len_s1 = len(s1)
    len_s2 = len(s2)
    
    # Matris başlat
    matrix = [[0 for _ in range(len_s2 + 1)] for _ in range(len_s1 + 1)]
    
    # İlk satır ve sütunu doldur
    for i in range(len_s1 + 1):
        matrix[i][0] = i
    for j in range(len_s2 + 1):
        matrix[0][j] = j
    
    # Matrisi doldur
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,  # Silme
                matrix[i][j-1] + 1,  # Ekleme
                matrix[i-1][j-1] + cost  # Değiştirme
            )
    
    return matrix[len_s1][len_s2]