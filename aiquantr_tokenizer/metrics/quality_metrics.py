"""
Kalite metrikleri modülü.

Bu modül, tokenizer eğitimi için veri kalitesini ve
tokenizer performansını değerlendiren işlevler içerir.
"""

import math
import logging
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Union, Set

import numpy as np

# Logger oluştur
logger = logging.getLogger(__name__)

# İsteğe bağlı bağımlılıklar
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib yüklü değil. Görselleştirme devre dışı.")


def calculate_text_diversity(text: str) -> Dict[str, float]:
    """
    Metin çeşitliliği metrikleri hesaplar.
    
    Args:
        text: Değerlendirilecek metin
        
    Returns:
        Dict[str, float]: Metin çeşitliliği metrikleri
    """
    if not text:
        return {
            "type_token_ratio": 0.0,
            "hapax_legomena_ratio": 0.0,
            "shannon_entropy": 0.0,
            "simpson_diversity": 0.0
        }
    
    # Kelimeler
    words = text.lower().split()
    word_count = len(words)
    
    # Kelime sıklıkları
    word_freqs = Counter(words)
    unique_words = len(word_freqs)
    
    # Tip-token oranı (TTR)
    ttr = unique_words / word_count if word_count > 0 else 0
    
    # Hapax legomena (yalnızca bir kez geçen kelimeler) oranı
    hapax_count = sum(1 for word, count in word_freqs.items() if count == 1)
    hapax_ratio = hapax_count / word_count if word_count > 0 else 0
    
    # Shannon entropi (bit)
    probabilities = [count / word_count for count in word_freqs.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities) if probabilities else 0
    
    # Simpson çeşitlilik indeksi
    simpson = 1 - sum(count * (count - 1) for count in word_freqs.values()) / (word_count * (word_count - 1)) if word_count > 1 else 0
    
    return {
        "type_token_ratio": ttr,
        "hapax_legomena_ratio": hapax_ratio,
        "shannon_entropy": entropy,
        "simpson_diversity": simpson
    }


def calculate_token_distribution(
    tokens: List[str], 
    num_bins: int = 50
) -> Dict[str, Any]:
    """
    Token dağılımı metrikleri hesaplar.
    
    Args:
        tokens: Token listesi
        num_bins: Histogram için grup sayısı
        
    Returns:
        Dict[str, Any]: Token dağılım metrikleri
    """
    if not tokens:
        return {
            "total_tokens": 0,
            "unique_tokens": 0,
            "frequency_stats": {},
            "entropy": 0.0
        }
    
    # Token sıklıkları
    token_freqs = Counter(tokens)
    
    # Temel istatistikler
    total_tokens = len(tokens)
    unique_tokens = len(token_freqs)
    
    # Sıklık istatistikleri
    freq_counts = list(token_freqs.values())
    
    frequency_stats = {
        "min": min(freq_counts),
        "max": max(freq_counts),
        "mean": sum(freq_counts) / len(freq_counts),
        "median": np.median(freq_counts),
        "std": np.std(freq_counts)
    }
    
    # En sık kullanılan tokenler
    most_common = token_freqs.most_common(20)
    
    # Shannon entropi (bit)
    probabilities = [count / total_tokens for count in token_freqs.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    
    # Histogram verisi
    hist_values, hist_bins = np.histogram(freq_counts, bins=num_bins)
    
    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "frequency_stats": frequency_stats,
        "most_common": most_common,
        "entropy": entropy,
        "histogram": {
            "values": hist_values.tolist(),
            "bins": hist_bins.tolist()
        }
    }


def evaluate_coverage(
    tokenizer, 
    texts: List[str],
    unk_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tokenizer'ın veri kümesi üzerindeki kapsamını değerlendirir.
    
    Args:
        tokenizer: Değerlendirilecek tokenizer (encode metodu olmalı)
        texts: Değerlendirme metinleri
        unk_token: Bilinmeyen token
        
    Returns:
        Dict[str, Any]: Kapsam metrikleri
    """
    if not texts:
        return {
            "coverage": 1.0,
            "unk_ratio": 0.0,
            "avg_token_per_word": 0.0
        }
    
    total_tokens = 0
    total_unks = 0
    total_words = 0
    per_text_coverage = []
    
    for text in texts:
        if not text:
            continue
            
        # Kelime sayısı
        words = text.split()
        word_count = len(words)
        total_words += word_count
        
        try:
            # Tokenize et
            tokens = tokenizer.encode(text)
            token_count = len(tokens)
            total_tokens += token_count
            
            # Bilinmeyen token sayısı
            if unk_token is not None:
                unk_count = tokens.count(unk_token) if isinstance(tokens, list) else 0
                total_unks += unk_count
                
                # Bu metin için kapsam
                text_coverage = 1.0 - (unk_count / token_count) if token_count > 0 else 1.0
                per_text_coverage.append(text_coverage)
                
        except Exception as e:
            logger.error(f"Tokenizer hatası: {str(e)}")
    
    # Genel kapsam ve metrikler
    coverage = 1.0 - (total_unks / total_tokens) if total_tokens > 0 else 1.0
    unk_ratio = total_unks / total_tokens if total_tokens > 0 else 0.0
    avg_token_per_word = total_tokens / total_words if total_words > 0 else 0.0
    
    return {
        "coverage": coverage,
        "unk_ratio": unk_ratio,
        "avg_token_per_word": avg_token_per_word,
        "per_text_coverage": {
            "min": min(per_text_coverage) if per_text_coverage else 1.0,
            "max": max(per_text_coverage) if per_text_coverage else 1.0,
            "avg": sum(per_text_coverage) / len(per_text_coverage) if per_text_coverage else 1.0
        }
    }


def evaluate_tokenizer_quality(
    tokenizer, 
    texts: List[str],
    reference_tokenizer = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Tokenizer kalitesini çeşitli metriklerle değerlendirir.
    
    Args:
        tokenizer: Değerlendirilecek tokenizer
        texts: Değerlendirme metinleri
        reference_tokenizer: Kıyaslama için referans tokenizer (isteğe bağlı)
        metrics: Hesaplanacak metrikler listesi
        
    Returns:
        Dict[str, Any]: Kalite metrikleri
    """
    # Varsayılan metrikler
    all_metrics = ["coverage", "compression", "consistency", "reversibility"]
    metrics = metrics or all_metrics
    
    results = {}
    
    # Kapsam değerlendirmesi
    if "coverage" in metrics:
        coverage_metrics = evaluate_coverage(tokenizer, texts)
        results["coverage"] = coverage_metrics
    
    # Sıkıştırma oranı
    if "compression" in metrics:
        compression = calculate_compression_ratio(tokenizer, texts)
        results["compression"] = compression
    
    # Tutarlılık değerlendirmesi
    if "consistency" in metrics:
        consistency = evaluate_consistency(tokenizer, texts)
        results["consistency"] = consistency
    
    # Tersine çevrilebilirlik
    if "reversibility" in metrics:
        reversibility = evaluate_reversibility(tokenizer, texts)
        results["reversibility"] = reversibility
    
    # Referans tokenizer ile kıyaslama
    if reference_tokenizer is not None:
        ref_results = {}
        
        # Kapsam kıyaslaması
        if "coverage" in metrics:
            ref_coverage = evaluate_coverage(reference_tokenizer, texts)
            ref_results["coverage"] = ref_coverage
            
            # Fark
            results["coverage_diff"] = coverage_metrics["coverage"] - ref_coverage["coverage"]
        
        # Sıkıştırma kıyaslaması
        if "compression" in metrics:
            ref_compression = calculate_compression_ratio(reference_tokenizer, texts)
            ref_results["compression"] = ref_compression
            
            # Fark
            compression_key = "avg_compression_ratio"
            results["compression_diff"] = results["compression"][compression_key] - ref_compression[compression_key]
        
        results["reference_metrics"] = ref_results
    
    return results


def calculate_compression_ratio(
    tokenizer, 
    texts: List[str]
) -> Dict[str, Any]:
    """
    Tokenizer'ın metin sıkıştırma oranını hesaplar.
    
    Args:
        tokenizer: Değerlendirilecek tokenizer
        texts: Değerlendirme metinleri
        
    Returns:
        Dict[str, Any]: Sıkıştırma metrikleri
    """
    if not texts:
        return {
            "avg_compression_ratio": 0.0,
            "char_to_token_ratio": 0.0
        }
    
    compression_ratios = []
    total_chars = 0
    total_tokens = 0
    
    for text in texts:
        if not text:
            continue
            
        # Karakter sayısı
        char_count = len(text)
        total_chars += char_count
        
        try:
            # Tokenize et
            tokens = tokenizer.encode(text)
            token_count = len(tokens)
            total_tokens += token_count
            
            # Sıkıştırma oranı
            if token_count > 0:
                ratio = char_count / token_count
                compression_ratios.append(ratio)
                
        except Exception as e:
            logger.error(f"Tokenizer hatası: {str(e)}")
    
    # Ortalama sıkıştırma oranı
    avg_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0
    
    # Genel karakter/token oranı
    char_to_token_ratio = total_chars / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "avg_compression_ratio": avg_ratio,
        "char_to_token_ratio": char_to_token_ratio,
        "compression_std": np.std(compression_ratios) if compression_ratios else 0.0,
        "min_compression": min(compression_ratios) if compression_ratios else 0.0,
        "max_compression": max(compression_ratios) if compression_ratios else 0.0
    }


def evaluate_consistency(
    tokenizer, 
    texts: List[str]
) -> Dict[str, float]:
    """
    Tokenizer tutarlılığını değerlendirir.
    
    Args:
        tokenizer: Değerlendirilecek tokenizer
        texts: Değerlendirme metinleri
        
    Returns:
        Dict[str, float]: Tutarlılık metrikleri
    """
    if not texts:
        return {
            "consistency_score": 1.0,
            "stable_segment_ratio": 1.0
        }
    
    consistency_scores = []
    stable_segments = 0
    total_segments = 0
    
    # Her metin için
    for text in texts:
        if not text:
            continue
        
        # Metin uzunluğunun %80'i
        segment_length = int(len(text) * 0.8)
        
        if segment_length < 5:
            continue
            
        # Metnin alt parçası
        segment = text[:segment_length]
        
        try:
            # Tam metin ve segment için token listeleri
            full_tokens = tokenizer.encode(text)
            segment_tokens = tokenizer.encode(segment)
            
            # Alt segment tokenlarının tam tokenları ile eşleşip eşleşmediğini kontrol et
            if len(segment_tokens) <= len(full_tokens):
                # Tokenların başlangıç kısmı eşleşiyor mu?
                is_stable = segment_tokens == full_tokens[:len(segment_tokens)]
                
                if is_stable:
                    stable_segments += 1
                
                total_segments += 1
                
                # Tutarlılık skoru (yüzde olarak eşleşen tokenler)
                matching_tokens = sum(1 for i in range(min(len(segment_tokens), len(full_tokens))) 
                                    if segment_tokens[i] == full_tokens[i])
                score = matching_tokens / len(segment_tokens) if len(segment_tokens) > 0 else 1.0
                consistency_scores.append(score)
                
        except Exception as e:
            logger.error(f"Tokenizer tutarlılık hatası: {str(e)}")
    
    # Ortalama tutarlılık skoru
    avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
    
    # Kararlı segment oranı
    stable_ratio = stable_segments / total_segments if total_segments > 0 else 1.0
    
    return {
        "consistency_score": avg_consistency,
        "stable_segment_ratio": stable_ratio
    }


def evaluate_reversibility(
    tokenizer, 
    texts: List[str]
) -> Dict[str, float]:
    """
    Tokenizer tersine çevrilebilirliğini değerlendirir.
    
    Args:
        tokenizer: Değerlendirilecek tokenizer (encode ve decode metodları olmalı)
        texts: Değerlendirme metinleri
        
    Returns:
        Dict[str, float]: Tersine çevrilebilirlik metrikleri
    """
    if not texts:
        return {
            "reversibility_score": 1.0,
            "exact_match_ratio": 1.0
        }
    
    reversibility_scores = []
    exact_matches = 0
    total_texts = 0
    
    # Tokenizer'ın decode metodu olduğundan emin ol
    if not hasattr(tokenizer, 'decode'):
        logger.error("Tokenizer'da 'decode' metodu bulunamadı.")
        return {
            "reversibility_score": 0.0,
            "exact_match_ratio": 0.0,
            "error": "Tokenizer'da 'decode' metodu yok"
        }
    
    # Her metin için
    for text in texts:
        if not text:
            continue
            
        total_texts += 1
        
        try:
            # Tokenize et ve sonra geri çevir
            tokens = tokenizer.encode(text)
            decoded_text = tokenizer.decode(tokens)
            
            # Tam eşleşme mi?
            if text == decoded_text:
                exact_matches += 1
                reversibility_scores.append(1.0)
            else:
                # Kısmi eşleşme skoru (karakter bazlı benzerlik)
                score = calculate_similarity(text, decoded_text)
                reversibility_scores.append(score)
                
        except Exception as e:
            logger.error(f"Tokenizer tersine çevirme hatası: {str(e)}")
            reversibility_scores.append(0.0)
    
    # Ortalama tersine çevrilebilirlik skoru
    avg_reversibility = sum(reversibility_scores) / len(reversibility_scores) if reversibility_scores else 0.0
    
    # Tam eşleşme oranı
    exact_ratio = exact_matches / total_texts if total_texts > 0 else 0.0
    
    return {
        "reversibility_score": avg_reversibility,
        "exact_match_ratio": exact_ratio
    }


def calculate_similarity(text1: str, text2: str) -> float:
    """
    İki metin arasındaki benzerliği hesaplar.
    
    Karakter bazlı benzerlik skoru döndürür (0.0 - 1.0 arası).
    
    Args:
        text1: Birinci metin
        text2: İkinci metin
        
    Returns:
        float: Benzerlik skoru (0.0 - 1.0)
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    # Levenshtein mesafesi hesapla (basit dinamik programlama)
    len1, len2 = len(text1), len(text2)
    
    # Çok uzun metinler için kabaca hesaplama
    if len1 > 1000 or len2 > 1000:
        # Örnek al
        samples = 5
        sample_size = min(100, min(len1, len2) // samples)
        
        similarities = []
        for i in range(samples):
            start_pos = i * (min(len1, len2) // samples)
            sample1 = text1[start_pos:start_pos + sample_size]
            sample2 = text2[start_pos:start_pos + sample_size]
            
            # Örnek benzerliğini hesapla
            sample_sim = 1.0 - levenshtein_distance(sample1, sample2) / max(len(sample1), len(sample2))
            similarities.append(sample_sim)
        
        # Ortalama benzerlik
        return sum(similarities) / len(similarities)
    
    # Normal uzunlukta metinler için tam hesaplama
    distance = levenshtein_distance(text1, text2)
    
    # Benzerlik skoru
    return 1.0 - distance / max(len1, len2)


def levenshtein_distance(text1: str, text2: str) -> int:
    """
    İki metin arasındaki Levenshtein mesafesini hesaplar.
    
    Args:
        text1: Birinci metin
        text2: İkinci metin
        
    Returns:
        int: Levenshtein mesafesi
    """
    if not text1:
        return len(text2)
    if not text2:
        return len(text1)
    
    # Dinamik programlama matrisini hazırla
    m, n = len(text1), len(text2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Başlangıç değerleri
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Mesafeyi hesapla
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if text1[i - 1] == text2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Silme
                dp[i][j - 1] + 1,      # Ekleme
                dp[i - 1][j - 1] + cost  # Değiştirme
            )
    
    return dp[m][n]


def plot_token_distribution(
    distribution: Dict[str, Any],
    output_file: Optional[str] = None,
    title: str = "Token Frequency Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Token dağılımını görselleştirir.
    
    Args:
        distribution: calculate_token_distribution'dan dönen sözlük
        output_file: Çıktı dosya yolu (None ise ekranda gösterilir)
        title: Grafik başlığı
        figsize: Grafik boyutu
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib yüklü değil. Görselleştirme yapılamıyor.")
        return
    
    # Histogram verisi
    if "histogram" not in distribution:
        logger.error("Dağılımda histogram verisi bulunamadı.")
        return
    
    hist_values = distribution["histogram"]["values"]
    hist_bins = distribution["histogram"]["bins"]
    
    # Grafiği oluştur
    plt.figure(figsize=figsize)
    plt.bar(
        range(len(hist_values)),
        hist_values,
        width=0.8,
        align='center',
        alpha=0.7
    )
    
    # En yüksek frekans değerlerini etiketlerle göster
    max_count = max(hist_values)
    for i, count in enumerate(hist_values):
        if count > max_count * 0.5:  # Sadece yüksek değerleri etiketle
            plt.text(i, count + max_count * 0.02, str(count), ha='center')
    
    # Eksen etiketleri ve başlık
    plt.xlabel('Token Frequency')
    plt.ylabel('Count')
    plt.title(title)
    
    # X ekseni etiketleri
    bin_labels = []
    for i in range(len(hist_bins) - 1):
        if i % 5 == 0 or i == len(hist_bins) - 2:  # Her 5 etikette bir göster
            bin_labels.append(f"{int(hist_bins[i])}-{int(hist_bins[i+1])}")
        else:
            bin_labels.append("")
    
    plt.xticks(range(len(hist_values)), bin_labels, rotation=45, ha='right')
    
    # Özet istatistikleri göster
    stats_text = (
        f"Total Tokens: {distribution['total_tokens']:,}\n"
        f"Unique Tokens: {distribution['unique_tokens']:,}\n"
        f"Type-Token Ratio: {distribution['unique_tokens']/distribution['total_tokens']:.4f}\n"
        f"Mean Frequency: {distribution['frequency_stats']['mean']:.2f}\n"
        f"Entropy: {distribution['entropy']:.2f} bits"
    )
    
    plt.figtext(0.15, 0.8, stats_text, fontsize=10,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    plt.tight_layout()
    
    # Kaydet veya göster
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Grafik kaydedildi: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_comparison(
    results: Dict[str, Any],
    reference_results: Dict[str, Any],
    output_file: Optional[str] = None,
    title: str = "Tokenizer Quality Comparison",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    İki tokenizer'ın kalite metriklerini karşılaştırır.
    
    Args:
        results: Birinci tokenizer sonuçları
        reference_results: İkinci tokenizer sonuçları
        output_file: Çıktı dosya yolu (None ise ekranda gösterilir)
        title: Grafik başlığı
        figsize: Grafik boyutu
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib yüklü değil. Görselleştirme yapılamıyor.")
        return
    
    # Karşılaştırılacak metrikler
    metrics = [
        ('coverage', 'Coverage', 'coverage'),
        ('compression', 'Compression Ratio', 'avg_compression_ratio'),
        ('consistency', 'Consistency', 'consistency_score'),
        ('reversibility', 'Reversibility', 'reversibility_score')
    ]
    
    # Metrikleri çıkar
    values1 = []
    values2 = []
    labels = []
    
    for metric_key, metric_label, sub_key in metrics:
        if metric_key in results:
            if sub_key in results[metric_key]:
                values1.append(results[metric_key][sub_key])
                labels.append(metric_label)
                
                # Referans değeri
                if metric_key in reference_results and sub_key in reference_results[metric_key]:
                    values2.append(reference_results[metric_key][sub_key])
                else:
                    values2.append(0)
    
    # Grafiği oluştur
    x = range(len(labels))
    width = 0.35
    
    plt.figure(figsize=figsize)
    
    rects1 = plt.bar([i - width/2 for i in x], values1, width, label='Current Tokenizer')
    rects2 = plt.bar([i + width/2 for i in x], values2, width, label='Reference Tokenizer')
    
    # Etiketler ve başlık
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    
    # Değerleri göster
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    # Kaydet veya göster
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Grafik kaydedildi: {output_file}")
    else:
        plt.show()
    
    plt.close()