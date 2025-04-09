"""
Veri analiz modülleri.

Bu modül, veri kümelerini analiz etmek, istatistikler çıkarmak ve
tokenizer eğitimindeki veri kalitesini değerlendirmek için fonksiyonlar sağlar.
"""

import re
import json
import logging
import collections
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterable, Counter, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Logger oluştur
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Veri kümesi analiz sınıfı.
    
    Bu sınıf, bir veri kümesindeki metin verilerini analiz ederek
    çeşitli istatistikler ve görselleştirmeler üretir.
    """
    
    def __init__(self, max_vocab_size: int = 100000):
        """
        DatasetAnalyzer sınıfı başlatıcısı.
        
        Args:
            max_vocab_size: Analiz edilecek maksimum kelime/token sayısı
        """
        self.max_vocab_size = max_vocab_size
        self.reset_stats()
        
    def reset_stats(self):
        """Tüm istatistikleri sıfırlar."""
        self.stats = {
            "sample_count": 0,
            "total_chars": 0,
            "total_words": 0,
            "avg_sample_length": 0,
            "avg_words_per_sample": 0,
            "vocab_size": 0
        }
        self.word_counts = collections.Counter()
        self.char_counts = collections.Counter()
        self.sample_lengths = []
        self.word_counts_per_sample = []
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Tek bir metin örneği için istatistikler üretir.
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Dict[str, Any]: Metin istatistikleri
        """
        if not text:
            return {
                "length": 0,
                "words": 0,
                "unique_words": 0,
                "unique_chars": 0
            }
        
        # Kelime ve karakter sayıları
        words = text.split()
        chars = list(text)
        
        # Tekil kelime ve karakterler
        unique_words = set(words)
        unique_chars = set(chars)
        
        return {
            "length": len(text),
            "words": len(words),
            "unique_words": len(unique_words),
            "unique_chars": len(unique_chars)
        }
        
    def analyze_dataset(self, texts: Iterable[str]) -> Dict[str, Any]:
        """
        Tüm veri kümesi için analiz yapar ve istatistikler üretir.
        
        Args:
            texts: Analiz edilecek metinler
            
        Returns:
            Dict[str, Any]: Veri kümesi istatistikleri
        """
        self.reset_stats()
        
        # Veri kümesi üzerinde döngü
        for text in texts:
            if not text:
                continue
                
            # İstatistikleri güncelle
            self.stats["sample_count"] += 1
            self.stats["total_chars"] += len(text)
            
            # Kelimeler
            words = text.split()
            self.stats["total_words"] += len(words)
            self.word_counts_per_sample.append(len(words))
            
            # Kelime frekansları
            for word in words:
                self.word_counts[word] += 1
            
            # Karakter frekansları
            for char in text:
                self.char_counts[char] += 1
                
            # Örnek uzunluğu
            self.sample_lengths.append(len(text))
        
        # Özet istatistikler hesapla
        if self.stats["sample_count"] > 0:
            self.stats["avg_sample_length"] = self.stats["total_chars"] / self.stats["sample_count"]
            self.stats["avg_words_per_sample"] = self.stats["total_words"] / self.stats["sample_count"]
            
        self.stats["vocab_size"] = len(self.word_counts)
        self.stats["char_vocab_size"] = len(self.char_counts)
        
        # Medyan istatistikleri
        if self.sample_lengths:
            self.stats["median_sample_length"] = np.median(self.sample_lengths)
            self.stats["median_words_per_sample"] = np.median(self.word_counts_per_sample)
        
        # En sık kelimeler ve karakter istatistikleri
        most_common_words = self.word_counts.most_common(min(100, len(self.word_counts)))
        most_common_chars = self.char_counts.most_common(min(100, len(self.char_counts)))
        
        self.stats["top_words"] = [(word, count) for word, count in most_common_words]
        self.stats["top_chars"] = [(char, count) for char, count in most_common_chars]
        
        # Dağılım istatistikleri
        if self.sample_lengths:
            self.stats["length_percentiles"] = {
                "min": min(self.sample_lengths),
                "25%": np.percentile(self.sample_lengths, 25),
                "50%": np.percentile(self.sample_lengths, 50),
                "75%": np.percentile(self.sample_lengths, 75),
                "max": max(self.sample_lengths)
            }
            
        if self.word_counts_per_sample:
            self.stats["words_percentiles"] = {
                "min": min(self.word_counts_per_sample),
                "25%": np.percentile(self.word_counts_per_sample, 25),
                "50%": np.percentile(self.word_counts_per_sample, 50),
                "75%": np.percentile(self.word_counts_per_sample, 75),
                "max": max(self.word_counts_per_sample)
            }
        
        return self.stats
        
    def plot_length_distribution(self, output_file: Optional[str] = None, 
                                bins: int = 50, figsize: Tuple[int, int] = (10, 6)):
        """
        Örnek uzunluğu dağılımını görselleştirir.
        
        Args:
            output_file: Çıktı dosyası adı (varsayılan: None - ekrana göster)
            bins: Histogram için dilim sayısı
            figsize: Şekil boyutu
        """
        if not self.sample_lengths:
            logger.warning("Görselleştirme için veri yok. Önce analyze_dataset() çağırın.")
            return
        
        plt.figure(figsize=figsize)
        plt.hist(self.sample_lengths, bins=bins, alpha=0.7)
        plt.title("Örnek Uzunluğu Dağılımı")
        plt.xlabel("Örnek Uzunluğu (karakter)")
        plt.ylabel("Örnek Sayısı")
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    def plot_words_distribution(self, output_file: Optional[str] = None,
                               bins: int = 50, figsize: Tuple[int, int] = (10, 6)):
        """
        Kelime sayısı dağılımını görselleştirir.
        
        Args:
            output_file: Çıktı dosyası adı (varsayılan: None - ekrana göster)
            bins: Histogram için dilim sayısı
            figsize: Şekil boyutu
        """
        if not self.word_counts_per_sample:
            logger.warning("Görselleştirme için veri yok. Önce analyze_dataset() çağırın.")
            return
        
        plt.figure(figsize=figsize)
        plt.hist(self.word_counts_per_sample, bins=bins, alpha=0.7)
        plt.title("Kelime Sayısı Dağılımı")
        plt.xlabel("Kelime Sayısı")
        plt.ylabel("Örnek Sayısı")
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()
            
        plt.close()
    
    def plot_top_words(self, n: int = 20, output_file: Optional[str] = None, 
                       figsize: Tuple[int, int] = (12, 8)):
        """
        En sık kullanılan kelimeleri görselleştirir.
        
        Args:
            n: Gösterilecek kelime sayısı
            output_file: Çıktı dosyası adı (varsayılan: None - ekrana göster)
            figsize: Şekil boyutu
        """
        if not self.word_counts:
            logger.warning("Görselleştirme için veri yok. Önce analyze_dataset() çağırın.")
            return
        
        top_words = self.word_counts.most_common(n)
        words, counts = zip(*top_words)
        
        plt.figure(figsize=figsize)
        plt.bar(range(len(words)), counts, align='center', alpha=0.7)
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.title(f"En Sık Kullanılan {n} Kelime")
        plt.xlabel("Kelime")
        plt.ylabel("Frekans")
        plt.tight_layout()
        plt.grid(True, axis='y', alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()
            
        plt.close()
    
    def plot_top_chars(self, n: int = 20, output_file: Optional[str] = None, 
                       figsize: Tuple[int, int] = (12, 8)):
        """
        En sık kullanılan karakterleri görselleştirir.
        
        Args:
            n: Gösterilecek karakter sayısı
            output_file: Çıktı dosyası adı (varsayılan: None - ekrana göster)
            figsize: Şekil boyutu
        """
        if not self.char_counts:
            logger.warning("Görselleştirme için veri yok. Önce analyze_dataset() çağırın.")
            return
        
        top_chars = self.char_counts.most_common(n)
        chars, counts = zip(*top_chars)
        
        # Özel karakterleri okunabilir yap
        display_chars = []
        for char in chars:
            if char.isspace():
                display_chars.append(f"SPACE ({ord(char)})")
            elif char == '\n':
                display_chars.append("\\n")
            elif char == '\t':
                display_chars.append("\\t")
            elif char == '\r':
                display_chars.append("\\r")
            elif not char.isprintable():
                display_chars.append(f"U+{ord(char):04X}")
            else:
                display_chars.append(char)
        
        plt.figure(figsize=figsize)
        plt.bar(range(len(display_chars)), counts, align='center', alpha=0.7)
        plt.xticks(range(len(display_chars)), display_chars, rotation=45, ha='right')
        plt.title(f"En Sık Kullanılan {n} Karakter")
        plt.xlabel("Karakter")
        plt.ylabel("Frekans")
        plt.tight_layout()
        plt.grid(True, axis='y', alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()
            
        plt.close()
    
    def save_stats(self, output_file: Union[str, Path]):
        """
        Analiz istatistiklerini JSON dosyasına kaydeder.
        
        Args:
            output_file: Çıktı dosyası yolu
        """
        if not self.stats:
            logger.warning("Kaydedilecek istatistik yok. Önce analyze_dataset() çağırın.")
            return
        
        # JSON uyumlu hale getir
        stats = dict(self.stats)
        
        # NumPy değerleri standart Python türlerine dönüştür
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                stats[key] = value.tolist()
            elif isinstance(value, np.generic):
                stats[key] = value.item()
        
        # İstatistik alt sözlükleri için de aynı işlemi yap
        for key, value in stats.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        stats[key][k] = v.tolist()
                    elif isinstance(v, np.generic):
                        stats[key][k] = v.item()
        
        # JSON'a kaydet
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        logger.info(f"İstatistikler {output_file} dosyasına kaydedildi.")
    
    def generate_report(self, output_dir: Union[str, Path], 
                        prefix: str = "dataset_analysis",
                        save_plots: bool = True):
        """
        Kapsamlı bir analiz raporu oluşturur.
        
        Args:
            output_dir: Çıktı dizini
            prefix: Çıktı dosyaları için ön ek
            save_plots: Grafikleri kaydet (varsayılan: True)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # İstatistikleri kaydet
        self.save_stats(output_dir / f"{prefix}_stats.json")
        
        # Grafikleri kaydet
        if save_plots:
            self.plot_length_distribution(output_dir / f"{prefix}_length_dist.png")
            self.plot_words_distribution(output_dir / f"{prefix}_words_dist.png")
            self.plot_top_words(output_file=output_dir / f"{prefix}_top_words.png")
            self.plot_top_chars(output_file=output_dir / f"{prefix}_top_chars.png")
        
        # Özet rapor
        report = f"""# Veri Kümesi Analiz Raporu

## Genel İstatistikler
- Örnek Sayısı: {self.stats.get('sample_count', 0):,}
- Toplam Karakter Sayısı: {self.stats.get('total_chars', 0):,}
- Toplam Kelime Sayısı: {self.stats.get('total_words', 0):,}
- Ortalama Örnek Uzunluğu: {self.stats.get('avg_sample_length', 0):.2f} karakter
- Ortalama Kelime Sayısı: {self.stats.get('avg_words_per_sample', 0):.2f} kelime/örnek
- Kelime Dağarcığı Boyutu: {self.stats.get('vocab_size', 0):,} benzersiz kelime
- Karakter Dağarcığı: {self.stats.get('char_vocab_size', 0)} benzersiz karakter

## Uzunluk İstatistikleri
"""

        # Uzunluk yüzdesi istatistikleri
        if 'length_percentiles' in self.stats:
            report += f"""- Min Uzunluk: {self.stats['length_percentiles'].get('min', 0):,} karakter
- 25. Yüzde: {self.stats['length_percentiles'].get('25%', 0):.1f} karakter
- Medyan Uzunluk: {self.stats['length_percentiles'].get('50%', 0):.1f} karakter
- 75. Yüzde: {self.stats['length_percentiles'].get('75%', 0):.1f} karakter
- Maks Uzunluk: {self.stats['length_percentiles'].get('max', 0):,} karakter
"""

        # Kelime istatistikleri
        if 'words_percentiles' in self.stats:
            report += f"""
## Kelime İstatistikleri
- Min Kelime: {self.stats['words_percentiles'].get('min', 0)} kelime/örnek
- 25. Yüzde: {self.stats['words_percentiles'].get('25%', 0):.1f} kelime/örnek
- Medyan Kelime: {self.stats['words_percentiles'].get('50%', 0):.1f} kelime/örnek
- 75. Yüzde: {self.stats['words_percentiles'].get('75%', 0):.1f} kelime/örnek
- Maks Kelime: {self.stats['words_percentiles'].get('max', 0)} kelime/örnek
"""

        # En sık kelime ve karakterler
        report += "\n## En Sık Kullanılan Kelimeler\n"
        if 'top_words' in self.stats and self.stats['top_words']:
            report += "| Kelime | Frekans |\n|--------|--------|\n"
            for word, count in self.stats['top_words'][:20]:
                report += f"| `{word}` | {count:,} |\n"
        
        report += "\n## En Sık Kullanılan Karakterler\n"
        if 'top_chars' in self.stats and self.stats['top_chars']:
            report += "| Karakter | Unicode | Frekans |\n|----------|---------|--------|\n"
            for char, count in self.stats['top_chars'][:20]:
                if char.isspace():
                    char_display = "SPACE"
                    unicode_display = f"U+{ord(char):04X}"
                elif not char.isprintable():
                    char_display = f"U+{ord(char):04X}"
                    unicode_display = f"U+{ord(char):04X}"
                else:
                    char_display = char
                    unicode_display = f"U+{ord(char):04X}"
                report += f"| `{char_display}` | {unicode_display} | {count:,} |\n"
        
        # Raporu kaydet
        with open(output_dir / f"{prefix}_report.md", "w", encoding="utf-8") as f:
            f.write(report)
            
        logger.info(f"Rapor {output_dir} dizinine kaydedildi.")


def compute_text_stats(text: str) -> Dict[str, Any]:
    """
    Tek bir metin için temel istatistikler hesaplar.
    
    Args:
        text: Analiz edilecek metin
        
    Returns:
        Dict[str, Any]: Hesaplanan istatistikler
    """
    if not text:
        return {
            "length": 0,
            "words": 0,
            "lines": 0,
            "unique_words": 0,
            "unique_chars": 0
        }
    
    lines = text.split('\n')
    words = text.split()
    chars = list(text)
    
    # Benzersiz kelime ve karakterler
    unique_words = set(words)
    unique_chars = set(chars)
    
    return {
        "length": len(text),
        "words": len(words),
        "lines": len(lines),
        "avg_words_per_line": len(words) / max(1, len(lines)),
        "avg_chars_per_word": len(text) / max(1, len(words)),
        "unique_words": len(unique_words),
        "unique_chars": len(unique_chars),
        "top_characters": collections.Counter(chars).most_common(10),
        "lexical_diversity": len(unique_words) / max(1, len(words))
    }


def estimate_tokenizer_vocabulary_size(
    texts: Iterable[str],
    tokenization_strategy: str = "word",
    min_frequency: int = 2,
    sample_size: Optional[int] = None,
    bpe_merges: Optional[int] = None
) -> Dict[str, Any]:
    """
    Tokenizer eğitimi için gerekli kelime dağarcığı boyutunu tahmin eder.
    
    Args:
        texts: Metin örnekleri
        tokenization_strategy: Tokenizasyon stratejisi ("word", "byte", "char", "bpe")
        min_frequency: Minimum token frekansı
        sample_size: Analiz için örnek boyutu (None = tümü)
        bpe_merges: BPE tokenizer için birleştirme sayısı
        
    Returns:
        Dict[str, Any]: Tahmin sonuçları
    """
    # Örnekleme
    if sample_size is not None:
        sampled_texts = []
        count = 0
        for text in texts:
            sampled_texts.append(text)
            count += 1
            if count >= sample_size:
                break
        texts = sampled_texts
    
    results = {
        "strategy": tokenization_strategy,
        "min_frequency": min_frequency
    }
    
    all_tokens = []
    
    # Tokenizasyon stratejisine göre işlem
    if tokenization_strategy == "word":
        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)
    
    elif tokenization_strategy == "char":
        for text in texts:
            tokens = list(text)
            all_tokens.extend(tokens)
    
    elif tokenization_strategy == "byte":
        for text in texts:
            tokens = [f"<{ord(c):02x}>" for c in text]
            all_tokens.extend(tokens)
    
    elif tokenization_strategy == "bpe":
        # Basitleştirilmiş BPE simülasyonu - gerçek implementasyon farklıdır
        # Önce karakter tokenizasyonu yap
        char_tokens = []
        for text in texts:
            char_tokens.extend(list(text))
        
        # Frekans sözlüğü oluştur
        token_counts = collections.Counter(char_tokens)
        
        # Sadece min_frequency'den yüksek olanları tut
        filtered_tokens = [token for token, count in token_counts.items() if count >= min_frequency]
        
        # BPE birleştirme sayısı tahmini
        if bpe_merges is None:
            bpe_merges = len(filtered_tokens) // 2
        
        # Sadece temel tahmin için basit hesaplama
        estimated_vocab_size = len(filtered_tokens) + min(bpe_merges, len(filtered_tokens) * 5)
        
        results.update({
            "char_vocab_size": len(token_counts),
            "filtered_char_vocab_size": len(filtered_tokens),
            "estimated_bpe_merges": bpe_merges,
            "estimated_vocab_size": estimated_vocab_size
        })
        
        return results
    
    # Token frekansları
    token_counts = collections.Counter(all_tokens)
    
    # Filtreleme
    filtered_tokens = [token for token, count in token_counts.items() if count >= min_frequency]
    
    # Sonuçları topla
    results.update({
        "total_tokens": len(all_tokens),
        "unique_tokens": len(token_counts),
        "filtered_vocab_size": len(filtered_tokens),
        "top_tokens": token_counts.most_common(20)
    })
    
    return results