# aiquantr_tokenizer/tokenizers/wordpiece.py
"""
WordPiece tokenizer uygulaması.

Bu modül, BERT gibi modellerde kullanılan WordPiece
tokenizer modelinin implementasyonunu sağlar.
"""

import os
import json
import logging
import re
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Iterator, Callable

from .base import BaseTokenizer, TokenizerTrainer

# Logger oluştur
logger = logging.getLogger(__name__)


class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer implementasyonu.
    
    Bu sınıf, BERT ve türevi modellerde kullanılan
    WordPiece tokenizer'ın bir uygulamasını sağlar.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None,
        unk_token: str = "[UNK]",
        word_tokenizer_pattern: str = r"[^\s]+",
        wordpiece_prefix: str = "##",
        strip_accents: bool = False,
        lowercase: bool = False,
        name: Optional[str] = None
    ):
        """
        WordPieceTokenizer sınıfı başlatıcısı.
        
        Args:
            vocab_size: Sözlük boyutu (varsayılan: 30000)
            min_frequency: Minimum token frekansı (varsayılan: 2)
            special_tokens: Özel token eşlemeleri (varsayılan: None)
            unk_token: Bilinmeyen token (varsayılan: "[UNK]")
            word_tokenizer_pattern: Kelime tokenizer deseni (varsayılan: r"[^\s]+")
            wordpiece_prefix: Kelime parçası öneki (varsayılan: "##")
            strip_accents: Aksanları kaldır (varsayılan: False)
            lowercase: Metni küçük harfe çevir (varsayılan: False)
            name: Tokenizer adı (varsayılan: None)
        """
        # Özel tokenları yapılandır
        custom_special_tokens = special_tokens or {}
        if "unk_token" not in custom_special_tokens:
            custom_special_tokens["unk_token"] = unk_token
            
        super().__init__(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=custom_special_tokens,
            name=name or "WordPieceTokenizer"
        )
        
        self.word_tokenizer_pattern = word_tokenizer_pattern
        self.wordpiece_prefix = wordpiece_prefix
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        
        # WordPiece modeli için gerekli veri yapıları
        self.vocab = {}  # Token -> ID
        self.ids_to_tokens = {}  # ID -> Token
        self.unk_token_id = None
        self.word_tokenizer_regex = re.compile(word_tokenizer_pattern)
        self.max_input_chars_per_word = 100
        
        # Önbellek
        self.cache = {}
    
    def _strip_accents(self, text: str) -> str:
        """
        Metinden aksanları kaldırır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Aksanları kaldırılmış metin
        """
        import unicodedata
        
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text
    
    def _preprocess_text(self, text: str) -> str:
        """
        Metni ön işler.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Ön işlenmiş metin
        """
        if self.lowercase:
            text = text.lower()
            
        if self.strip_accents:
            text = self._strip_accents(text)
            
        return text
    
    def _tokenize_to_words(self, text: str) -> List[str]:
        """
        Metni kelimelere böler.
        
        Args:
            text: Tokenize edilecek metin
            
        Returns:
            List[str]: Kelimeler listesi
        """
        return self.word_tokenizer_regex.findall(text)
    
    def _split_word_to_wordpieces(self, word: str) -> List[str]:
        """
        Kelimeyi wordpiece'lere böler.
        
        Args:
            word: Bölünecek kelime
            
        Returns:
            List[str]: WordPiece'ler listesi
        """
        if not word:
            return []
            
        if word in self.cache:
            return self.cache[word]
            
        # Çok uzun kelimeleri unk_token olarak işaretle
        if len(word) > self.max_input_chars_per_word:
            return [self.special_tokens["unk_token"]]
            
        # Kelime doğrudan sözlükte varsa
        if word in self.vocab:
            return [word]
            
        # Kelimeyi karakter karakter böl ve tüm alt dizileri kontrol et
        tokens = []
        start = 0
        max_end = len(word)
        
        # İlk karakter için önek yok
        curr_substr = ""
        
        while start < max_end:
            # En uzun eşleşen alt dizeyi bul
            end = max_end
            found_piece = False
            
            while start < end:
                # Alt dizeyi oluştur
                if start == 0:
                    substr = word[start:end]
                else:
                    substr = self.wordpiece_prefix + word[start:end]
                
                # Sözlükte ara
                if substr in self.vocab:
                    found_piece = True
                    tokens.append(substr)
                    curr_substr = substr
                    start = end
                    break
                    
                end -= 1
            
            # Eşleşme bulunamadıysa, bilinmeyen token döndür
            if not found_piece:
                return [self.special_tokens["unk_token"]]
                
        # Sonucu önbelleğe al
        self.cache[word] = tokens
        return tokens
    
    def _count_subwords(self, texts: List[str]) -> Dict[str, int]:
        """
        Alt kelime (subword) frekanslarını hesaplar.
        
        Args:
            texts: Eğitim metinleri
            
        Returns:
            Dict[str, int]: Alt kelime frekansları
        """
        # Kelime frekanslarını hesapla
        word_freqs = Counter()
        
        for text in texts:
            # Metni ön işle
            text = self._preprocess_text(text)
            
            # Kelimelere böl
            words = self._tokenize_to_words(text)
            word_freqs.update(words)
        
        logger.info(f"Toplam benzersiz kelime: {len(word_freqs)}")
        
        # Alt kelime adaylarını oluştur
        subword_freqs = defaultdict(int)
        
        # Tekli karakterleri ekle
        char_set = set()
        for word, freq in word_freqs.items():
            for char in word:
                char_set.add(char)
                subword_freqs[char] += freq
        
        logger.info(f"Toplam benzersiz karakter: {len(char_set)}")
        
        # Tüm alt kelimeleri oluştur (ön ek ile)
        for word, freq in word_freqs.items():
            word_len = len(word)
            
            # Her olası alt kelimeyi değerlendir
            for start in range(word_len):
                # İlk karakter için önek yok
                is_first = (start == 0)
                
                for end in range(start + 1, word_len + 1):
                    # Alt kelimeyi oluştur
                    subword = word[start:end]
                    
                    if not is_first:
                        subword = self.wordpiece_prefix + subword
                        
                    subword_freqs[subword] += freq
        
        logger.info(f"Toplam alt kelime adayı: {len(subword_freqs)}")
        return subword_freqs
    
    def _compute_scores(
        self,
        subword_freqs: Dict[str, int],
        subword_indices: Dict[str, int],
        word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], float]:
        """
        Alt kelime çiftleri için birleştirme puanlarını hesaplar.
        
        Args:
            subword_freqs: Alt kelime frekansları
            subword_indices: Alt kelime indeksleri
            word_freqs: Kelime frekansları
            
        Returns:
            Dict[Tuple[str, str], float]: Birleştirme puanları
        """
        # Kelime içindeki alt kelime çiftlerini bul
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            if len(word) <= 1:
                continue
                
            # Kelimedeki karakter çiftlerini değerlendir
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freqs[pair] += freq
                
        # Puanları hesapla
        pair_scores = {}
        
        for pair, freq in pair_freqs.items():
            symbol = pair[0] + pair[1]
            score = freq / (subword_freqs[pair[0]] * subword_freqs[pair[1]])
            pair_scores[pair] = score
            
        return pair_scores
    
    def train(
        self,
        texts: List[str],
        trainer: Optional[TokenizerTrainer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        WordPiece modelini verilen metinler üzerinde eğitir.
        
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
        
        # Alt kelime frekanslarını hesapla
        subword_freqs = self._count_subwords(texts)
        
        # Özel tokenları ekle
        for token in self.special_tokens.values():
            if token not in subword_freqs:
                subword_freqs[token] = self.min_frequency
        
        # Alt kelimeleri sıralama
        sorted_subwords = sorted(
            subword_freqs.items(),
            key=lambda x: (-x[1], len(x[0]), x[0])  # -frekans, uzunluk, token (yüksek frekans önce)
        )
        
        # Sözlük boyutunu sınırla
        final_vocab_size = min(self.vocab_size, len(sorted_subwords))
        selected_subwords = [token for token, _ in sorted_subwords[:final_vocab_size]]
        
        # Token -> ID eşlemesi oluştur
        self.vocab = {token: idx for idx, token in enumerate(selected_subwords)}
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
        self.unk_token_id = self.vocab.get(self.special_tokens["unk_token"], 0)
        
        # Önbelleği temizle
        self.cache = {}
        
        # Eğitimi tamamla
        self.is_trained = True
        final_metrics = {
            "vocab_size": len(self.vocab),
            "num_candidates": len(subword_freqs),
            "final_vocab_size": final_vocab_size
        }
        
        # Eğitim meta verilerini güncelle
        self.metadata.update({
            "training_size": len(texts),
            "word_tokenizer_pattern": self.word_tokenizer_pattern,
            "wordpiece_prefix": self.wordpiece_prefix,
            "strip_accents": self.strip_accents,
            "lowercase": self.lowercase
        })
        
        trainer.on_iteration_end(self, 0, final_metrics)
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
                bos_id = self.vocab.get(self.special_tokens.get("bos_token"))
                eos_id = self.vocab.get(self.special_tokens.get("eos_token"))
                
                if bos_id is not None:
                    result.append(bos_id)
                if eos_id is not None:
                    result.append(eos_id)
            
            self.stats["num_encode_calls"] += 1
            self.stats["total_encode_time"] += time.time() - start_time
            return result
            
        # Metni ön işle
        text = self._preprocess_text(text)
        
        # Metin -> Kelimeler -> WordPieces
        words = self._tokenize_to_words(text)
        tokens = []
        
        # Başlangıç tokeni ekle
        if add_special_tokens and "bos_token" in self.special_tokens:
            bos_id = self.vocab.get(self.special_tokens["bos_token"])
            if bos_id is not None:
                tokens.append(bos_id)
        
        # Her kelimeyi tokenize et
        for word in words:
            wordpieces = self._split_word_to_wordpieces(word)
            
            for piece in wordpieces:
                token_id = self.vocab.get(piece, self.unk_token_id)
                tokens.append(token_id)
        
        # Bitiş tokeni ekle
        if add_special_tokens and "eos_token" in self.special_tokens:
            eos_id = self.vocab.get(self.special_tokens["eos_token"])
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
            
        # Özel token ID'lerini belirle
        special_token_ids = set()
        if skip_special_tokens:
            special_token_ids = {
                self.vocab.get(token) for token in self.special_tokens.values()
                if token in self.vocab
            }
        
        # Token ID'lerini tokenlerine dönüştür
        tokens = []
        for token_id in ids:
            if token_id in special_token_ids:
                continue
                
            if token_id in self.ids_to_tokens:
                tokens.append(self.ids_to_tokens[token_id])
            else:
                tokens.append(self.special_tokens["unk_token"])
        
        # WordPiece tokenlarını birleştir
        text = ""
        current_word = ""
        
        for token in tokens:
            if token.startswith(self.wordpiece_prefix):
                current_word += token[len(self.wordpiece_prefix):]
            else:
                if current_word:
                    text += current_word + " "
                    current_word = ""
                text += token + " "
        
        if current_word:
            text += current_word
            
        # Sondaki boşluğu temizle
        text = text.rstrip()
        
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
        return dict(self.vocab)
    
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
            
        # Metni ön işle
        text = self._preprocess_text(text)
        
        # Kelimelere böl ve her kelimeyi tokenize et
        words = self._tokenize_to_words(text)
        tokens = []
        
        for word in words:
            wordpieces = self._split_word_to_wordpieces(word)
            tokens.extend(wordpieces)
            
        return tokens
    
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
            "type": "WordPieceTokenizer",
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "word_tokenizer_pattern": self.word_tokenizer_pattern,
            "wordpiece_prefix": self.wordpiece_prefix,
            "strip_accents": self.strip_accents,
            "lowercase": self.lowercase,
            "max_input_chars_per_word": self.max_input_chars_per_word,
            "metadata": self.metadata
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Tokenizer {save_path} konumuna kaydedildi")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "WordPieceTokenizer":
        """
        Tokenizer modelini yükler.
        
        Args:
            path: Yükleme yolu
            **kwargs: Yükleme için ek parametreler
            
        Returns:
            WordPieceTokenizer: Yüklenen tokenizer modeli
            
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
        if data.get("type") != "WordPieceTokenizer":
            logger.warning(f"Yüklenen tokenizer türü uyumsuz. Beklenen: WordPieceTokenizer, Alınan: {data.get('type')}")
            
        # Tokenizer'ı yapılandır
        tokenizer = cls(
            vocab_size=data.get("vocab_size", 30000),
            min_frequency=data.get("min_frequency", 2),
            special_tokens=data.get("special_tokens"),
            word_tokenizer_pattern=data.get("word_tokenizer_pattern", r"[^\s]+"),
            wordpiece_prefix=data.get("wordpiece_prefix", "##"),
            strip_accents=data.get("strip_accents", False),
            lowercase=data.get("lowercase", False)
        )
        
        # Sözlük verilerini yükle
        tokenizer.vocab = data.get("vocab", {})
        tokenizer.ids_to_tokens = {int(idx): token for token, idx in tokenizer.vocab.items()}
        tokenizer.unk_token_id = tokenizer.vocab.get(tokenizer.special_tokens["unk_token"], 0)
        
        if "max_input_chars_per_word" in data:
            tokenizer.max_input_chars_per_word = data["max_input_chars_per_word"]
        
        # Meta verileri yükle
        if "metadata" in data:
            tokenizer.metadata.update(data["metadata"])
            
        # Eğitilmiş olarak işaretle
        tokenizer.is_trained = True
        
        logger.info(f"Tokenizer {load_path} konumundan yüklendi")
        return tokenizer