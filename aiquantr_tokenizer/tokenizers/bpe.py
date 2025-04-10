# aiquantr_tokenizer/tokenizers/bpe.py
"""
Byte-Pair Encoding (BPE) tokenizer uygulaması.

Bu modül, BPE tokenizer modelinin implementasyonunu sağlar.
BPE, metin verisindeki en yaygın karakter çiftlerini
yinelemeli olarak birleştiren bir alt kelime tokenizasyon algoritmasıdır.
"""

import os
import time 
import json
import logging
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Iterator, Callable

import numpy as np

from .base import BaseTokenizer, TokenizerTrainer

# Logger oluştur
logger = logging.getLogger(__name__)

import sys

if sys.version_info >= (3, 11):
    # Python 3.11+ için Unicode özellikleri destekleyen desen
    DEFAULT_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
else:
    # Python 3.10 ve öncesi için uyumlu desen
    DEFAULT_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding (BPE) tokenizer implementasyonu.
    
    Bu sınıf, OpenAI GPT ve GPT-2'de kullanılan BPE tokenizer'ın
    bir uygulamasını sağlar.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None,
        character_coverage: float = 1.0,
        split_pattern: str = DEFAULT_SPLIT_PATTERN,
        byte_fallback: bool = True,
        name: Optional[str] = None
    ):
        """
        BPETokenizer sınıfı başlatıcısı.
        
        Args:
            vocab_size: Sözlük boyutu (varsayılan: 30000)
            min_frequency: Minimum token frekansı (varsayılan: 2)
            special_tokens: Özel token eşlemeleri (varsayılan: None)
            character_coverage: Karakter kapsama oranı (varsayılan: 1.0)
            split_pattern: Ön tokenizasyon için ayrım deseni (varsayılan: GPT-2 benzeri)
            byte_fallback: Bilinmeyen karakterler için byte geri dönüşü kullan (varsayılan: True)
            name: Tokenizer adı (varsayılan: None)
        """
        super().__init__(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            name=name or "BPETokenizer"
        )
        
        self.character_coverage = character_coverage
        self.split_pattern = split_pattern
        self.byte_fallback = byte_fallback
        
        # BPE modeli için gerekli veri yapıları
        self.encoder = {}  # Token -> ID
        self.decoder = {}  # ID -> Token
        self.bpe_ranks = {}  # Birleşme sıralamaları
        self.byte_encoder = {}  # Byte -> Unicode
        self.byte_decoder = {}  # Unicode -> Byte
        self.cache = {}  # Belleğe alınmış tokenlar
        
        # Byte encoders/decoders oluştur
        if self.byte_fallback:
            self._init_byte_encoders()
    
    def _init_byte_encoders(self):
        """Byte -> Unicode eşlemelerini oluşturur."""
        # UTF-8 tabanlı byte encoding
        bytes_range = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        unicode_chars = [chr(b) for b in bytes_range]
        
        # Diğer byteler için özel Unicodes
        for b in range(256):
            if b not in bytes_range:
                unicode_chars.append(chr(b + 256))
                
        # Byte eşlemelerini oluştur
        for i, c in enumerate(unicode_chars):
            self.byte_encoder[i] = c
            self.byte_decoder[c] = i
    
    def _unicode_to_bytes(self, text: str) -> List[int]:
        """
        Unicode metni byte dizisine dönüştürür.
        
        Args:
            text: Dönüştürülecek metin
            
        Returns:
            List[int]: Byte değerleri listesi
        """
        return list(text.encode("utf-8"))
    
    def _bytes_to_unicode(self, bytes_list: List[int]) -> str:
        """
        Byte dizisini Unicode metne dönüştürür.
        
        Args:
            bytes_list: Dönüştürülecek byte dizisi
            
        Returns:
            str: Unicode metin
        """
        return "".join(self.byte_encoder.get(b, chr(b)) for b in bytes_list)
    
    def _split_to_words(self, text: str) -> List[str]:
        """
        Metni kelimelere böler (ön tokenizasyon).
        
        Args:
            text: Bölünecek metin
            
        Returns:
            List[str]: Kelimeler listesi
        """
        if not self.split_pattern:
            return list(text)
            
        return re.findall(self.split_pattern, text)
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tek bir kelimeyi tokenize eder.
        
        Args:
            word: Tokenize edilecek kelime
            
        Returns:
            List[str]: BPE alt dizileri
        """
        if not word:
            return []
            
        # Belleğe alınmış sonuç varsa onu kullan
        if word in self.cache:
            return self.cache[word]
            
        # Kelimeyi karakter dizisine dönüştür
        if self.byte_fallback:
            chars = self._bytes_to_unicode(self._unicode_to_bytes(word))
        else:
            chars = list(word)
            
        # Başlangıçta her karakter ayrı bir token
        pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            
        # BPE birleştirme işlemlerini yinelemeli olarak uygula
        while pairs:
            # En düşük sıralamaya sahip çifti bul
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            
            # Eğer çift sözlükte yoksa işlemi sonlandır
            if bigram not in self.bpe_ranks:
                break
                
            # Birleştirme işlemi için kelimeyi yeniden düzenle
            first, second = bigram
            new_chars = []
            i = 0
            
            while i < len(chars):
                try:
                    j = chars.index(first, i)
                    new_chars.extend(chars[i:j])
                    i = j
                except ValueError:
                    new_chars.extend(chars[i:])
                    break
                    
                # Birleştirme işlemi
                if chars[i] == first and i < len(chars) - 1 and chars[i + 1] == second:
                    new_chars.append(first + second)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            
            # Yeni karakterleri güncelle
            chars = new_chars
            
            # Eğer tek karaktere indiyse sonlandır
            if len(chars) == 1:
                break
                
            # Yeni çiftleri oluştur
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
        
        # Sonucu belleğe al
        self.cache[word] = chars
        return chars
    
    def _count_tokens(self, texts: List[str]) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
        """
        Veri kümesindeki token ve çift frekanslarını hesaplar.
        
        Args:
            texts: Eğitim metinleri
            
        Returns:
            Tuple[Dict[str, int], Dict[Tuple[str, str], int]]: 
                (token frekansları, çift frekansları)
        """
        token_freqs = Counter()
        pair_freqs = Counter()
        
        for text in texts:
            # Metni kelimelere böl
            words = self._split_to_words(text)
            
            for word in words:
                if not word:
                    continue
                    
                # Kelimeyi karakterlere böl
                if self.byte_fallback:
                    chars = self._bytes_to_unicode(self._unicode_to_bytes(word))
                else:
                    chars = list(word)
                    
                # Karakter frekanslarını güncelle
                token_freqs.update(chars)
                
                # Çift frekanslarını güncelle
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_freqs[pair] += 1
        
        return token_freqs, pair_freqs
    
    def _get_most_frequent_pair(self, pair_freqs: Dict[Tuple[str, str], int]) -> Optional[Tuple[str, str]]:
        """
        En yaygın karakter çiftini döndürür.
        
        Args:
            pair_freqs: Çift frekansları
            
        Returns:
            Optional[Tuple[str, str]]: En yaygın çift veya hiç yoksa None
        """
        if not pair_freqs:
            return None
            
        return max(pair_freqs.items(), key=lambda x: x[1])[0]
    
    def _merge_pair(
        self,
        pair: Tuple[str, str],
        token_freqs: Dict[str, int],
        pair_freqs: Dict[Tuple[str, str], int]
    ) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
        """
        Verilen çifti birleştirir ve frekansları günceller.
        
        Args:
            pair: Birleştirilecek çift
            token_freqs: Token frekansları
            pair_freqs: Çift frekansları
            
        Returns:
            Tuple[Dict[str, int], Dict[Tuple[str, str], int]]: 
                (güncellenmiş token frekansları, güncellenmiş çift frekansları)
        """
        first, second = pair
        new_token = first + second
        
        # Yeni token frekansını hesapla
        token_freqs[new_token] = pair_freqs[pair]
        
        # Komşu çiftlerin frekanslarını güncelle
        for (prev, curr), freq in list(pair_freqs.items()):
            if curr == first:
                pair_freqs[(prev, new_token)] = pair_freqs.get((prev, new_token), 0) + freq
                
            if curr == second and prev == first:
                pair_freqs[(prev, curr)] -= freq
                if pair_freqs[(prev, curr)] == 0:
                    del pair_freqs[(prev, curr)]
        
        for (curr, next_token), freq in list(pair_freqs.items()):
            if curr == second:
                pair_freqs[(new_token, next_token)] = pair_freqs.get((new_token, next_token), 0) + freq
                
            if curr == first and next_token == second:
                pair_freqs[(curr, next_token)] -= freq
                if pair_freqs[(curr, next_token)] == 0:
                    del pair_freqs[(curr, next_token)]
        
        return token_freqs, pair_freqs
    
    def train(
        self,
        texts: List[str],
        trainer: Optional[TokenizerTrainer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        BPE modelini verilen metinler üzerinde eğitir.
        
        Args:
            texts: Eğitim metinleri
            trainer: Eğitimi yönetecek TokenizerTrainer nesnesi (varsayılan: None)
            **kwargs: Eğitim için ek parametreler
            
        Returns:
            Dict[str, Any]: Eğitim istatistikleri
        """
        # Varsayılan eğitim yapılandırması
        max_iterations = kwargs.get("max_iterations", self.vocab_size - len(self.special_tokens))
        
        # Eğitim yöneticisini yapılandır
        if trainer is None:
            trainer = TokenizerTrainer(
                batch_size=kwargs.get("batch_size", 1000),
                num_iterations=max_iterations,
                show_progress=kwargs.get("show_progress", True)
            )
        
        # Eğitim başlangıcı
        trainer.on_training_begin(self, texts)
        
        # Token ve çift frekanslarını hesapla
        token_freqs, pair_freqs = self._count_tokens(texts)
        logger.info(f"Başlangıç: {len(token_freqs)} benzersiz karakter, {len(pair_freqs)} benzersiz çift")
        
        # Özel tokenları ekle
        for token in self.special_tokens.values():
            if token not in token_freqs:
                token_freqs[token] = self.min_frequency
        
        # Birleştirme sıralamaları için sözlük
        self.bpe_ranks = {}
        vocab = list(token_freqs.keys())
        
        # Tokenları ID'lere eşle
        self.encoder = {token: i for i, token in enumerate(vocab)}
        self.decoder = {i: token for token, i in self.encoder.items()}
        
        # Bellek önbelleğini temizle
        self.cache = {}
        
        # BPE eğitim döngüsü
        for iteration in range(max_iterations):
            trainer.on_iteration_begin(self, iteration)
            
            # En yaygın çifti bul
            most_common_pair = self._get_most_frequent_pair(pair_freqs)
            
            if not most_common_pair or pair_freqs[most_common_pair] < self.min_frequency:
                break
                
            # Yeni token oluştur ve frekansları güncelle
            new_token = most_common_pair[0] + most_common_pair[1]
            vocab.append(new_token)
            self.encoder[new_token] = len(self.encoder)
            self.decoder[len(self.decoder)] = new_token
            
            # Birleştirme sıralamasını kaydet
            self.bpe_ranks[most_common_pair] = iteration
            
            # Frekans tablolarını güncelle
            token_freqs, pair_freqs = self._merge_pair(most_common_pair, token_freqs, pair_freqs)
            
            # İlerleme istatistikleri
            metrics = {
                "vocab_size": len(self.encoder),
                "most_common_pair_freq": pair_freqs[most_common_pair] if most_common_pair in pair_freqs else 0,
                "remaining_pairs": len(pair_freqs)
            }
            
            trainer.on_iteration_end(self, iteration, metrics)
            
            # Erken durdurma kontrolü
            if trainer.should_stop_early(self, iteration, metrics):
                break
        
        # Eğitimi tamamla
        self._is_trained = True
        final_metrics = {
            "vocab_size": len(self.encoder),
            "num_merges": len(self.bpe_ranks),
            "num_base_tokens": len(token_freqs)
        }
        
        # Eğitim meta verilerini güncelle
        self.metadata.update({
            "training_size": len(texts),
            "character_coverage": self.character_coverage,
            "byte_fallback": self.byte_fallback
        })
        
        trainer.on_training_end(self, final_metrics)
        return final_metrics
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Metni token ID'lerine dönüştürür.
        
        Args:
            text: Encode edilecek metin
            add_special_tokens: Başlangıç/bitiş tokenlarını ekle (varsayılan: True)
            **kwargs: Encode için ek parametreler
            
        Returns:
            List[int]: Token ID'leri
        """
        start_time = time.time()
        
        if not text:
            result = []
            
            if add_special_tokens:
                bos_id = self.encoder.get(self.special_tokens["bos_token"])
                eos_id = self.encoder.get(self.special_tokens["eos_token"])
                
                if bos_id is not None:
                    result.append(bos_id)
                if eos_id is not None:
                    result.append(eos_id)
                    
            self.stats["num_encode_calls"] += 1
            self.stats["total_encode_time"] += time.time() - start_time
            return result
            
        # Metni kelimelere böl
        words = self._split_to_words(text)
        tokens = []
        
        # Başlangıç tokeni ekle
        if add_special_tokens and "bos_token" in self.special_tokens:
            bos_id = self.encoder.get(self.special_tokens["bos_token"])
            if bos_id is not None:
                tokens.append(bos_id)
        
        # Her kelimeyi tokenize et
        for word in words:
            if not word:
                continue
                
            word_tokens = self._tokenize_word(word)
            tokens.extend([self.encoder.get(t, self.encoder.get(self.special_tokens["unk_token"])) for t in word_tokens])
        
        # Bitiş tokeni ekle
        if add_special_tokens and "eos_token" in self.special_tokens:
            eos_id = self.encoder.get(self.special_tokens["eos_token"])
            if eos_id is not None:
                tokens.append(eos_id)
        
        # İstatistikleri güncelle
        self.stats["num_encode_calls"] += 1
        self.stats["total_encode_time"] += time.time() - start_time
        
        return tokens
    
    def decode(
    self,
    ids: List[int],
    skip_special_tokens: bool = True,
    **kwargs
    ) -> str:
        """
        Token ID'lerini metne dönüştürür.
        
        Args:
            ids: Decode edilecek token ID'leri
            skip_special_tokens: Özel tokenları atla (varsayılan: True)
            **kwargs: Decode için ek parametreler
            
        Returns:
            str: Elde edilen metin
        """
        start_time = time.time()
        
        if not ids:
            self.stats["num_decode_calls"] += 1
            self.stats["total_decode_time"] += time.time() - start_time
            return ""
        
        # 1. ID'leri doğru tokenlara çevir
        tokens = []
        special_token_ids = set()
        
        if skip_special_tokens:
            special_token_ids = {
                self.encoder.get(token) for token in self.special_tokens.values()
                if token in self.encoder
            }
        
        for token_id in ids:
            if token_id in special_token_ids:
                continue
            
            if token_id in self.decoder:
                tokens.append(self.decoder[token_id])
            else:
                tokens.append(self.special_tokens["unk_token"])
        
        # 2. Tokenları özel işlem yapmadan birleştir
        raw_text = "".join(tokens)
        
        # 3. Byte fallback kullanılıyorsa, UTF-8 dönüşümü yap
        if self.byte_fallback:
            try:
                byte_list = []
                for c in raw_text:
                    # Eğer karakter byte_decoder'da varsa, doğrudan byte değerini kullan
                    if c in self.byte_decoder:
                        byte_list.append(self.byte_decoder[c])
                    # Yoksa, karakteri UTF-8 olarak encode et
                    else:
                        # Maksimum 4 byte olabilir
                        char_bytes = c.encode('utf-8')
                        byte_list.extend(char_bytes)
                
                # Byte listesini UTF-8 metin olarak decode et
                text = bytes(byte_list).decode('utf-8', errors='replace')
            except Exception as e:
                logger.warning(f"Byte dönüşüm hatası: {e}")
                text = raw_text
        else:
            text = raw_text
        
        # İstatistikleri güncelle
        self.stats["num_decode_calls"] += 1
        self.stats["total_decode_time"] += time.time() - start_time
        
        return text
        
    def get_vocab(self) -> Dict[str, int]:
        """
        Tokenizer'ın sözlüğünü döndürür.
        
        Returns:
            Dict[str, int]: Token - ID eşleşmelerini içeren sözlük
        """
        return dict(self.encoder)
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Tokenizer modelini kaydeder.
        
        Args:
            path: Kaydetme yolu
            **kwargs: Kaydetme için ek parametreler
        """
        path = Path(path)
        
        # Ana dizini oluştur
        if path.is_dir():
            save_path = path / "tokenizer.json"
        else:
            save_path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # Model verilerini kaydet
        data = {
            "type": "BPETokenizer",
            "vocab": self.encoder,
            "merges": [(pair[0], pair[1]) for pair in self.bpe_ranks.keys()],
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "character_coverage": self.character_coverage,
            "split_pattern": self.split_pattern,
            "byte_fallback": self.byte_fallback,
            "metadata": self.metadata
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Tokenizer {save_path} konumuna kaydedildi")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "BPETokenizer":
        """
        Tokenizer modelini yükler.
        
        Args:
            path: Yükleme yolu
            **kwargs: Yükleme için ek parametreler
            
        Returns:
            BPETokenizer: Yüklenen tokenizer modeli
            
        Raises:
            ValueError: Model yüklenemezse
        """
        path = Path(path)
        
        # Dosya yolu veya dizin yolu kontrolü
        if path.is_dir():
            load_path = path / "tokenizer.json"
        else:
            load_path = path
            
        if not load_path.exists():
            raise ValueError(f"Tokenizer dosyası bulunamadı: {load_path}")
            
        # Model verilerini yükle
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Model türünü doğrula
        if data.get("type") != "BPETokenizer":
            logger.warning(f"Yüklenen tokenizer türü uyumsuz. Beklenen: BPETokenizer, Alınan: {data.get('type')}")
            
        # Tokenizer'ı yapılandır
        tokenizer = cls(
            vocab_size=data.get("vocab_size", 30000),
            min_frequency=data.get("min_frequency", 2),
            special_tokens=data.get("special_tokens"),
            character_coverage=data.get("character_coverage", 1.0),
            split_pattern=data.get("split_pattern", r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"),
            byte_fallback=data.get("byte_fallback", True)
        )
        
        # Sözlük ve birleştirmeleri yükle
        tokenizer.encoder = data.get("vocab", {})
        tokenizer.decoder = {int(i): t for t, i in tokenizer.encoder.items()}
        
        # Birleştirmeleri sıralamalara dönüştür
        bpe_ranks = {}
        for i, (first, second) in enumerate(data.get("merges", [])):
            bpe_ranks[(first, second)] = i
            
        tokenizer.bpe_ranks = bpe_ranks
        
        # Meta verileri yükle
        if "metadata" in data:
            tokenizer.metadata.update(data["metadata"])
            
        # Eğitilmiş olarak işaretle
        tokenizer.is_trained = True
        
        logger.info(f"Tokenizer {load_path} konumundan yüklendi")
        return tokenizer
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Metni token dizilerine dönüştürür (ID'lere kodlamadan).
        
        Args:
            text: Tokenize edilecek metin
            **kwargs: Tokenize için ek parametreler
            
        Returns:
            List[str]: Token dizileri
        """
        if not text:
            return []
            
        # Metni kelimelere böl
        words = self._split_to_words(text)
        tokens = []
        
        # Her kelimeyi tokenize et
        for word in words:
            if not word:
                continue
                
            tokens.extend(self._tokenize_word(word))
            
        return tokens