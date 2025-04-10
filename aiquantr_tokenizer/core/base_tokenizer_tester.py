# base_tokenizer_tester.py

import json
import time
import os
import logging
import re
import random
from typing import Dict, List, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher, Differ

class BaseTokenizerTester:
    """
    Tokenizer'ları test etmek ve değerlendirmek için temel sınıf.
    Google'ın kullandığı roundtrip doğruluk testi ve diğer metrikleri içerir.
    """
    
    def __init__(self, 
                 test_name: str,
                 results_dir: str = "test_results",
                 logging_level: int = logging.INFO):
        """
        BaseTokenizerTester sınıfı başlatıcısı.
        
        Args:
            test_name: Test grubunun adı (ör: "python", "php", "natural_language")
            results_dir: Test sonuçlarının kaydedileceği dizin
            logging_level: Günlükleme seviyesi
        """
        self.test_name = test_name
        self.results_dir = results_dir
        self.results = {}
        self.test_data = {}
        self.tokenizers = {}
        
        # Sonuç dizinini oluştur
        os.makedirs(results_dir, exist_ok=True)
        
        # Logger ayarları
        self.logger = logging.getLogger(f"tokenizer_test.{test_name}")
        self.logger.setLevel(logging_level)
        
        if not self.logger.handlers:
            # Dosya handler
            file_handler = logging.FileHandler(f"{results_dir}/{test_name}_test.log")
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)
            
            # Konsol handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(levelname)s: %(message)s'
            ))
            self.logger.addHandler(console_handler)
    
    def load_test_data(self, data_path: str) -> Dict[str, Any]:
        """
        Test verilerini yükler.
        
        Args:
            data_path: Test verisi yolu (JSON dosyası)
            
        Returns:
            Dict: Yüklenen test verisi
        """
        self.logger.info(f"Test verisi yükleniyor: {data_path}")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.test_data = data
            self.logger.info(f"Test verisi yüklendi. {sum(len(samples) for samples in data.values() if isinstance(samples, list))} örnek bulundu.")
            return data
        except Exception as e:
            self.logger.error(f"Test verisi yüklenirken hata: {str(e)}")
            return {}
    
    def register_tokenizer(self, name: str, tokenizer: Any) -> None:
        """
        Test edilecek bir tokenizer kaydeder.
        
        Args:
            name: Tokenizer adı
            tokenizer: Tokenizer nesnesi
        """
        self.tokenizers[name] = tokenizer
        self.logger.info(f"Tokenizer kaydedildi: {name}")
    
    def evaluate_tokenizers(self, 
                           metrics: Optional[List[str]] = None,
                           sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Tüm kayıtlı tokenizer'ları değerlendirir.
        
        Args:
            metrics: Hesaplanacak metrikler listesi (None ise tümü hesaplanır)
            sample_size: Test edilecek maksimum örnek sayısı (None ise tümü)
            
        Returns:
            Dict: Test sonuçları
        """
        if not self.tokenizers:
            self.logger.error("Hiç tokenizer kaydedilmemiş.")
            return {}
            
        if not self.test_data:
            self.logger.error("Test verisi yüklenmemiş.")
            return {}
        
        results = {
            "test_name": self.test_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tokenizers": {},
            "comparative": {}
        }
        
        self.logger.info(f"{len(self.tokenizers)} tokenizer değerlendiriliyor...")
        
        # Her bir tokenizer için değerlendirme yap
        for name, tokenizer in self.tokenizers.items():
            self.logger.info(f"'{name}' tokenizer'ı değerlendiriliyor...")
            
            # Google tarzı roundtrip doğruluk testi
            google_results = self.evaluate_google_style(name, tokenizer, sample_size)
            
            # Genel değerlendirme
            tokenizer_results = self._evaluate_single_tokenizer(tokenizer, metrics, sample_size)
            
            # Google sonuçlarını birleştir
            tokenizer_results["google_roundtrip"] = google_results
            
            # Tokenizer sonuçlarını ekle
            results["tokenizers"][name] = tokenizer_results
            
        # Karşılaştırmalı sonuçları hesapla
        self._calculate_comparative_metrics(results)
        
        self.results = results
        self.logger.info("Tüm tokenizer'lar değerlendirildi.")
        
        return results
    
    def _evaluate_single_tokenizer(self, 
                                  tokenizer: Any, 
                                  metrics: Optional[List[str]] = None,
                                  sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Tek bir tokenizer'ı değerlendirir.
        
        Args:
            tokenizer: Tokenizer nesnesi
            metrics: Hesaplanacak metrikler listesi
            sample_size: Test edilecek maksimum örnek sayısı
            
        Returns:
            Dict: Tokenizer değerlendirme sonuçları
        """
        results = {
            "overall": {},
            "by_category": {}
        }
        
        # Temel metrik değişkenleri
        total_tokens = 0
        total_chars = 0
        total_time = 0
        total_samples = 0
        roundtrip_matches = 0
        
        # Her kategori için test
        for category, samples in self.test_data.items():
            if not isinstance(samples, list):
                continue
                
            # Örnekleri sınırla
            if sample_size and len(samples) > sample_size:
                category_samples = random.sample(samples, sample_size)
            else:
                category_samples = samples
            
            category_results = {
                "samples": len(category_samples),
                "total_tokens": 0,
                "total_chars": 0,
                "chars_per_token": 0,
                "tokens_per_sample": 0,
                "processing_time": 0,
                "roundtrip_accuracy": 0
            }
            
            category_tokens = 0
            category_chars = 0
            category_time = 0
            category_matches = 0
            
            # Her örnek için test
            for sample in category_samples:
                # İşlem süresi ölçümü
                start_time = time.time()
                
                # Tokenize işlemi
                try:
                    tokens = self._tokenize(tokenizer, sample)
                    
                    # Detokenize işlemi (roundtrip testi)
                    decoded = self._detokenize(tokenizer, tokens)
                    
                    # İstatistik hesaplamaları
                    tokens_count = len(tokens)
                    chars_count = len(sample)
                    
                    category_tokens += tokens_count
                    category_chars += chars_count
                    
                    # Roundtrip eşleşme kontrolü
                    if decoded == sample:
                        category_matches += 1
                        roundtrip_matches += 1
                    
                    # İşlem süresi hesaplaması
                    process_time = time.time() - start_time
                    category_time += process_time
                    
                except Exception as e:
                    self.logger.warning(f"Örnek işlenirken hata: {str(e)}")
                    continue
            
            # Kategori sonuçlarını hesapla
            if len(category_samples) > 0:
                category_results["total_tokens"] = category_tokens
                category_results["total_chars"] = category_chars
                
                if category_tokens > 0:
                    category_results["chars_per_token"] = category_chars / category_tokens
                
                category_results["tokens_per_sample"] = category_tokens / len(category_samples)
                category_results["processing_time"] = category_time
                category_results["tokens_per_second"] = category_tokens / category_time if category_time > 0 else 0
                category_results["roundtrip_accuracy"] = (category_matches / len(category_samples)) * 100
            
            # Genel toplamları güncelle
            total_tokens += category_tokens
            total_chars += category_chars
            total_time += category_time
            total_samples += len(category_samples)
            
            # Kategori sonuçlarını ekle
            results["by_category"][category] = category_results
        
        # Genel sonuçları hesapla
        if total_samples > 0:
            results["overall"]["total_samples"] = total_samples
            results["overall"]["total_tokens"] = total_tokens
            results["overall"]["total_chars"] = total_chars
            results["overall"]["chars_per_token"] = total_chars / total_tokens if total_tokens > 0 else 0
            results["overall"]["tokens_per_sample"] = total_tokens / total_samples
            results["overall"]["processing_time"] = total_time
            results["overall"]["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0
            results["overall"]["roundtrip_accuracy"] = (roundtrip_matches / total_samples) * 100
        
        return results
    
    def evaluate_google_style(self, tokenizer_name: str, tokenizer: Any, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Google'ın kullandığı şekilde tokenizer roundtrip testini çalıştırır.
        
        Args:
            tokenizer_name: Tokenizer adı
            tokenizer: Test edilecek tokenizer nesnesi
            sample_size: Her kategoriden test edilecek maksimum örnek sayısı
            
        Returns:
            Dict: Tokenizer roundtrip performans sonuçları
        """
        # Test verilerini hazırla
        test_texts = []
        for category, samples in self.test_data.items():
            if isinstance(samples, list):
                # Örnekleri sınırla
                if sample_size and len(samples) > sample_size:
                    category_samples = random.sample(samples, sample_size)
                else:
                    category_samples = samples
                    
                test_texts.extend(category_samples)
        
        # Genel roundtrip doğruluk testi
        accuracy = self.calculate_roundtrip_accuracy(tokenizer, test_texts)
        
        # Eğer tam eşleşme %100 değilse, sorunlu örnekleri analiz et
        problem_examples = []
        if accuracy["exact_match_percentage"] < 100:
            # En fazla 10 sorunlu örneği incele
            for i in range(min(10, len(test_texts))):
                text = test_texts[i]
                tokens = self._tokenize(tokenizer, text)
                decoded = self._detokenize(tokenizer, tokens)
                
                if text != decoded:
                    analysis = self.analyze_roundtrip_differences(tokenizer, text)
                    problem_examples.append({
                        "original": text,
                        "decoded": decoded,
                        "token_count": len(tokens),
                        "analysis": analysis
                    })
        
        # Sonuçları derle
        results = {
            "tokenizer_name": tokenizer_name,
            "test_sample_size": len(test_texts),
            "accuracy": accuracy,
            "problem_examples_count": len(problem_examples),
            "problem_examples": problem_examples[:5]  # En fazla 5 örnek göster
        }
        
        return results
    
    def calculate_roundtrip_accuracy(self, tokenizer: Any, texts: List[str]) -> Dict[str, Any]:
        """
        Google'ın tokenizer roundtrip doğruluk testini uygular.
        Tam olarak Google'ın SentencePiece ve BERT değerlendirmelerinde kullandığı metrik.
        
        Args:
            tokenizer: Test edilecek tokenizer
            texts: Test edilecek metinler listesi
            
        Returns:
            Dict: Roundtrip doğruluk metrikleri
        """
        exact_matches = 0
        total_chars = 0
        total_different_chars = 0
        
        for text in texts:
            # Text → Tokens → Text dönüşümü
            try:
                # Tokenize edip sonra tekrar decode et
                tokens = self._tokenize(tokenizer, text)
                decoded_text = self._detokenize(tokenizer, tokens)
                
                # Tam eşleşme kontrolü
                if text == decoded_text:
                    exact_matches += 1
                
                # Karakter düzeyinde farklılık hesaplama (SentencePiece yaklaşımı)
                matcher = SequenceMatcher(None, text, decoded_text)
                
                # Orijinal metin uzunluğu
                total_chars += len(text)
                
                # Farklı olan karakterlerin sayısı
                # (Google'ın SentencePiece'deki character_coverage metriği ile aynı)
                common_chars = sum(block.size for block in matcher.get_matching_blocks())
                different_chars = len(text) - common_chars
                total_different_chars += different_chars
            
            except Exception as e:
                self.logger.warning(f"Roundtrip hesaplanamadı: {str(e)}")
                # Hata durumunda istatistikleri tutma
        
        # Google'ın metriklerini aynı isimlerle hesapla
        results = {
            "exact_match_count": exact_matches,
            "total_samples": len(texts),
            "exact_match_ratio": (exact_matches / len(texts)) if texts else 0,
            "character_coverage": (1 - (total_different_chars / total_chars)) if total_chars > 0 else 0,
            "character_error_rate": (total_different_chars / total_chars) if total_chars > 0 else 1
        }
        
        # Yüzde olarak formatla
        results["exact_match_percentage"] = results["exact_match_ratio"] * 100
        results["character_coverage_percentage"] = results["character_coverage"] * 100
        
        return results
    
    # BaseTokenizerTester sınıfına eklenecek metot
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        İki metin arasındaki benzerlik yüzdesini hesaplar.
        
        Args:
            text1: Birinci metin
            text2: İkinci metin
            
        Returns:
            float: Benzerlik yüzdesi (0-100)
        """
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio() * 100
    
    def analyze_roundtrip_differences(self, tokenizer: Any, text: str) -> Dict[str, Any]:
        """
        Google'ın token roundtrip analizinde kullandığı detaylı karşılaştırma.
        Sorunlu tokenları belirlemek için kullanılır.
        
        Args:
            tokenizer: Test edilecek tokenizer
            text: Analiz edilecek metin
        
        Returns:
            Dict: Detaylı analiz sonuçları
        """
        # Tokenize-decode döngüsü
        tokens = self._tokenize(tokenizer, text)
        decoded = self._detokenize(tokenizer, tokens)
        
        # Detaylı fark analizi
        differ = Differ()
        diff = list(differ.compare(text.splitlines(keepends=True), 
                                  decoded.splitlines(keepends=True)))
        
        # Değişiklik tipleri
        additions = len([d for d in diff if d.startswith('+')])
        deletions = len([d for d in diff if d.startswith('-')])
        changes = len([d for d in diff if d.startswith('?')])
        
        # Orijinal tokenların hangi konumda sorun çıkardığını tespit et
        problem_indices = []
        for i in range(len(tokens)):
            partial_tokens = tokens[:i+1]
            partial_decoded = self._detokenize(tokenizer, partial_tokens)
            if not text.startswith(partial_decoded) and not partial_decoded.startswith(text):
                problem_indices.append(i)
        
        # Tokenlar ve metin karakterleri arasındaki eşleşmeyi hesapla
        alignment = {
            "problem_token_indices": problem_indices,
            "problem_token_count": len(problem_indices),
            "total_tokens": len(tokens),
            "token_error_rate": len(problem_indices) / len(tokens) if tokens else 0,
            "additions": additions,
            "deletions": deletions, 
            "changes": changes
        }
        
        return alignment
    
    def _tokenize(self, tokenizer: Any, text: str) -> List[Any]:
        """
        Verilen metni tokenize eder. Alt sınıflar tarafından override edilebilir.
        
        Args:
            tokenizer: Tokenizer nesnesi
            text: Tokenize edilecek metin
            
        Returns:
            List: Token listesi
        """
        # Varsayılan uygulama genel bir tokenizer API'si varsayar
        if hasattr(tokenizer, 'encode'):
            return tokenizer.encode(text)
        elif hasattr(tokenizer, 'tokenize'):
            return tokenizer.tokenize(text)
        else:
            return tokenizer(text)  # Çağrılabilir nesne olarak varsay
    
    def _detokenize(self, tokenizer: Any, tokens: List[Any]) -> str:
        """
        Token listesini metne dönüştürür. Alt sınıflar tarafından override edilebilir.
        
        Args:
            tokenizer: Tokenizer nesnesi
            tokens: Token listesi
            
        Returns:
            str: Dönüştürülmüş metin
        """
        # Varsayılan uygulama genel bir tokenizer API'si varsayar
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(tokens)
        elif hasattr(tokenizer, 'detokenize'):
            return tokenizer.detokenize(tokens)
        elif hasattr(tokenizer, 'convert_tokens_to_string'):
            return tokenizer.convert_tokens_to_string(tokens)
        else:
            # Basit birleştirme stratejisi (son çare)
            return ''.join([str(t) for t in tokens])
    
    def _calculate_comparative_metrics(self, results: Dict[str, Any]) -> None:
        """
        Karşılaştırmalı metrikleri hesaplar ve sonuçlara ekler.
        
        Args:
            results: Değerlendirme sonuçları
        """
        comparative = {
            "chars_per_token": {},
            "tokens_per_second": {},
            "roundtrip_accuracy": {},
            "character_coverage": {}
        }
        
        # Her bir metrik için karşılaştırmayı hesapla
        for name, data in results["tokenizers"].items():
            # Standart metrikler
            if "overall" in data:
                if "chars_per_token" in data["overall"]:
                    comparative["chars_per_token"][name] = data["overall"]["chars_per_token"]
                if "tokens_per_second" in data["overall"]:
                    comparative["tokens_per_second"][name] = data["overall"]["tokens_per_second"]
                if "roundtrip_accuracy" in data["overall"]:
                    comparative["roundtrip_accuracy"][name] = data["overall"]["roundtrip_accuracy"]
            
            # Google tarzı metrikler
            if "google_roundtrip" in data and "accuracy" in data["google_roundtrip"]:
                if "character_coverage_percentage" in data["google_roundtrip"]["accuracy"]:
                    comparative["character_coverage"][name] = data["google_roundtrip"]["accuracy"]["character_coverage_percentage"]
        
        # En iyi/en kötü değerleri hesapla
        for metric, values in comparative.items():
            if values:
                is_lower_better = metric == "chars_per_token"  # Düşük olması iyi olan metrikler
                
                best_name = min(values.items(), key=lambda x: x[1])[0] if is_lower_better else max(values.items(), key=lambda x: x[1])[0]
                worst_name = max(values.items(), key=lambda x: x[1])[0] if is_lower_better else min(values.items(), key=lambda x: x[1])[0]
                
                comparative[metric] = {
                    "values": values,
                    "best": best_name,
                    "worst": worst_name
                }
        
        results["comparative"] = comparative
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Test sonuçlarını JSON dosyasına kaydeder.
        
        Args:
            filename: Dosya adı (None ise otomatik oluşturulur)
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        if not self.results:
            self.logger.warning("Kaydedilecek sonuç bulunamadı. Önce evaluate_tokenizers() çağırın.")
            return ""
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.test_name}_results_{timestamp}.json"
        
        filepath = Path(self.results_dir) / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Sonuçlar kaydedildi: {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Sonuçlar kaydedilirken hata: {str(e)}")
            return ""
    
    def plot_results(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Test sonuçlarını görselleştirir ve grafik dosyalarına kaydeder.
        
        Args:
            output_dir: Grafiklerin kaydedileceği dizin (None ise results_dir kullanılır)
            
        Returns:
            List[str]: Kaydedilen grafik dosyaları yolları
        """
        if not self.results:
            self.logger.warning("Görselleştirilecek sonuç bulunamadı. Önce evaluate_tokenizers() çağırın.")
            return []
        
        if output_dir is None:
            output_dir = self.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Token ekonomisi grafiği
        try:
            plt.figure(figsize=(10, 6))
            
            tokenizer_names = list(self.results["tokenizers"].keys())
            chars_per_token = [self.results["tokenizers"][name]["overall"].get("chars_per_token", 0) for name in tokenizer_names]
            
            plt.bar(tokenizer_names, chars_per_token)
            plt.title('Token Ekonomisi (Karakter/Token)', fontsize=14)
            plt.xlabel('Tokenizer', fontsize=12)
            plt.ylabel('Karakter/Token', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = f"{self.test_name}_token_economy_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            plt.close()
        except Exception as e:
            self.logger.error(f"Token ekonomisi grafiği oluşturulurken hata: {str(e)}")
        
        # Hız performansı grafiği
        try:
            plt.figure(figsize=(10, 6))
            
            tokens_per_second = [self.results["tokenizers"][name]["overall"].get("tokens_per_second", 0) for name in tokenizer_names]
            
            plt.bar(tokenizer_names, tokens_per_second)
            plt.title('Hız Performansı (Token/Saniye)', fontsize=14)
            plt.xlabel('Tokenizer', fontsize=12)
            plt.ylabel('Token/Saniye', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = f"{self.test_name}_speed_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            plt.close()
        except Exception as e:
            self.logger.error(f"Hız performansı grafiği oluşturulurken hata: {str(e)}")
        
        # Roundtrip doğruluğu grafiği (Google tarzı)
        try:
            plt.figure(figsize=(10, 6))
            
            character_coverage = []
            for name in tokenizer_names:
                if "google_roundtrip" in self.results["tokenizers"][name] and "accuracy" in self.results["tokenizers"][name]["google_roundtrip"]:
                    coverage = self.results["tokenizers"][name]["google_roundtrip"]["accuracy"].get("character_coverage_percentage", 0)
                    character_coverage.append(coverage)
                else:
                    character_coverage.append(0)
            
            plt.bar(tokenizer_names, character_coverage)
            plt.title('Karakter Kapsamı (%)', fontsize=14)
            plt.xlabel('Tokenizer', fontsize=12)
            plt.ylabel('Kapsamı (%)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)  # 0-100% aralığında göster
            plt.tight_layout()
            
            filename = f"{self.test_name}_character_coverage_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            plt.close()
        except Exception as e:
            self.logger.error(f"Karakter kapsamı grafiği oluşturulurken hata: {str(e)}")
        
        # Tam eşleşme grafiği
        try:
            plt.figure(figsize=(10, 6))
            
            exact_match = []
            for name in tokenizer_names:
                if "google_roundtrip" in self.results["tokenizers"][name] and "accuracy" in self.results["tokenizers"][name]["google_roundtrip"]:
                    value = self.results["tokenizers"][name]["google_roundtrip"]["accuracy"].get("exact_match_percentage", 0)
                    exact_match.append(value)
                else:
                    exact_match.append(0)
            
            plt.bar(tokenizer_names, exact_match)
            plt.title('Tam Eşleşme Oranı (%)', fontsize=14)
            plt.xlabel('Tokenizer', fontsize=12)
            plt.ylabel('Tam Eşleşme (%)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)  # 0-100% aralığında göster
            plt.tight_layout()
            
            filename = f"{self.test_name}_exact_match_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            plt.close()
        except Exception as e:
            self.logger.error(f"Tam eşleşme grafiği oluşturulurken hata: {str(e)}")
        
        self.logger.info(f"{len(saved_files)} grafik kaydedildi.")
        return saved_files
    
    def print_summary(self) -> None:
        """
        Test sonuçlarının özetini yazdırır.
        """
        if not self.results:
            print("Henüz test sonucu yok. Önce evaluate_tokenizers() çağırın.")
            return
        
        print("\n" + "="*80)
        print(f"  {self.test_name.upper()} TOKENIZER TEST SONUÇLARI")
        print("="*80)
        
        for name, data in self.results["tokenizers"].items():
            overall = data["overall"]
            google_metrics = data.get("google_roundtrip", {}).get("accuracy", {})
            
            print(f"\n{name}:")
            print(f"  Token Ekonomisi: {overall.get('chars_per_token', 0):.2f} karakter/token")
            print(f"  Hız: {overall.get('tokens_per_second', 0):.2f} token/saniye")
            
            # Google tarzı metrikler
            print("\n  Google Roundtrip Metrikleri:")
            print(f"    Tam Eşleşme: {google_metrics.get('exact_match_percentage', 0):.2f}%")
            print(f"    Karakter Kapsamı: {google_metrics.get('character_coverage_percentage', 0):.2f}%")
            print(f"    Karakter Hata Oranı: {google_metrics.get('character_error_rate', 0):.4f}")
            
            # Sorunlu örnek sayısı
            problem_count = data.get("google_roundtrip", {}).get("problem_examples_count", 0)
            if problem_count > 0:
                print(f"\n    Sorunlu Örnek Sayısı: {problem_count}")
        
        print("\nKARŞILAŞTIRMALI SONUÇLAR:")
        
        for metric, metric_data in self.results["comparative"].items():
            if isinstance(metric_data, dict) and "best" in metric_data and "worst" in metric_data:
                metric_name = {
                    "chars_per_token": "Token Ekonomisi (karakter/token)",
                    "tokens_per_second": "Hız (token/saniye)", 
                    "roundtrip_accuracy": "Roundtrip Doğruluğu (%)",
                    "character_coverage": "Karakter Kapsamı (%)"
                }.get(metric, metric)
                
                print(f"\n{metric_name}:")
                best_value = metric_data["values"][metric_data["best"]]
                worst_value = metric_data["values"][metric_data["worst"]]
                
                print(f"  En iyi: {metric_data['best']} ({best_value:.2f})")
                print(f"  En kötü: {metric_data['worst']} ({worst_value:.2f})")
        
        print("\n" + "="*80)