# aiquantr_tokenizer/tokenizers/unigram.py
"""
Unigram tokenizer uygulaması.

Bu modül, SentencePiece'in Unigram modelinde kullanılan
istatistiksel tokenizer'ın bir implementasyonunu sağlar.
"""

import os
import json
import logging
import re
import math
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Iterator, Callable

import numpy as np

from .base import BaseTokenizer, TokenizerTrainer

# Logger oluştur
logger = logging.getLogger(__name__)


class UnigramTokenizer(BaseTokenizer):
    """
    Unigram tokenizer implementasyonu.
    
    Bu sınıf, SentencePiece'in Unigram modelini temel alır
    ve dil modelleri için alt kelime tokenizasyonu sağlar.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None,
        character_coverage: float = 0.9995,
        split_pattern: str = r"\s+",
        unk_piece: str = "[UNK]",
        split_by_whitespace: bool = True,
        normalization_rule: str = "nmt_nfkc",
        treat_whitespace_as_suffix: bool = False,
        name: Optional[str] = None
    ):
        """
        UnigramTokenizer sınıfı başlatıcısı.
        
        Args:
            vocab_size: Sözlük boyutu (varsayılan: 30000)
            min_frequency: Minimum token frekansı (varsayılan: 2)
            special_tokens: Özel token eşlemeleri (varsayılan: None)
            character_coverage: Karakter kapsama oranı (varsayılan: 0.9995)
            split_pattern: İlk ayrım için kullanılacak desen (varsayılan: r"\s+")
            unk_piece: Bilinmeyen token (varsayılan: "[UNK]")
            split_by_whitespace: Boşlukla ayır (varsayılan: True)
            normalization_rule: Normalizasyon kuralı (varsayılan: "nmt_nfkc")
            treat_whitespace_as_suffix: Boşlukları sonek olarak kabul et (varsayılan: False)
            name: Tokenizer adı (varsayılan: None)
        """
        # Özel tokenları yapılandır
        custom_special_tokens = special_tokens or {}
        if "unk_token" not in custom_special_tokens:
            custom_special_tokens["unk_token"] = unk_piece
            
        super().__init__(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=custom_special_tokens,
            name=name or "UnigramTokenizer"
        )
        
        self.character_coverage = character_coverage
        self.split_pattern = split_pattern
        self.split_by_whitespace = split_by_whitespace
        self.normalization_rule = normalization_rule
        self.treat_whitespace_as_suffix = treat_whitespace_as_suffix
        
        # Unigram modeli için gerekli veri yapıları
        self.pieces = []  # Token listesi
        self.vocab = {}  # Token -> ID
        self.scores = {}  # Token -> log olasılık puanı
        self.ids_to_pieces = {}  # ID -> Token
        
        # Normalizasyon için derlenen regex
        self.whitespace_regex = re.compile(r"\s+")
        self.split_regex = re.compile(split_pattern) if split_pattern else None
        
        # Önbellek
        self.cache = {}
    
    def _normalize_text(self, text: str) -> str:
        """
        Metni normalleştirir.
        
        Args:
            text: Normalleştirilecek metin
            
        Returns:
            str: Normalleştirilmiş metin
        """
        if not text:
            return ""
            
        if self.normalization_rule == "nmt_nfkc":
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
            
        elif self.normalization_rule == "nfkc":
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
            
        elif self.normalization_rule == "nmt":
            # NMT kuralları uygula (elmas karakteri vb. kaldır)
            # TODO: NMT normalizasyon kuralları ekle
            pass
            
        return text
    
    def _split_text(self, text: str) -> List[str]:
        """
        Metni segmentlere böler.
        
        Args:
            text: Bölünecek metin
            
        Returns:
            List[str]: Metin segmentleri
        """
        if not text:
            return []
        
        if self.split_regex:
            segments = self.split_regex.split(text)
            segments = [segment for segment in segments if segment]
            return segments
            
        if self.split_by_whitespace:
            segments = self.whitespace_regex.split(text)
            segments = [segment for segment in segments if segment]
            return segments
            
        # Hiçbir ayrım yöntemi tanımlanmadıysa, karakterlere böl
        return list(text)
    
    def _gather_piece_candidates(self, texts: List[str]) -> Dict[str, int]:
        """
        Veri kümesinden token adaylarını toplar.
        
        Args:
            texts: Eğitim metinleri
            
        Returns:
            Dict[str, int]: Token adayları ve frekansları
        """
        # Karakter ve n-gram frekansları
        char_freqs = Counter()
        ngram_freqs = defaultdict(int)
        
        total_chars = 0
        covered_chars = set()
        
        for text in texts:
            # Metni normalleştir
            text = self._normalize_text(text)
            
            # Segment frekanslarını hesapla
            segments = self._split_text(text)
            
            for segment in segments:
                # Karakter frekanslarını güncelle
                for char in segment:
                    char_freqs[char] += 1
                    total_chars += 1
                    covered_chars.add(char)
                    
                # Her uzunlukta n-gramları değerlendir (maksimum 16 karakter)
                seg_len = min(len(segment), 16)
                for i in range(seg_len):
                    for j in range(i + 1, seg_len + 1):
                        ngram = segment[i:j]
                        if ngram:
                            ngram_freqs[ngram] += 1
                            
        logger.info(f"Toplam {len(char_freqs)} benzersiz karakter, {len(covered_chars)}/{total_chars} karakter kapsamı")
        
        # Karakter kapsamını hesapla
        current_coverage = len(covered_chars) / total_chars if total_chars > 0 else 0
        
        # Yeterli kapsama yoksa, daha fazla karakter ekle
        if current_coverage < self.character_coverage and total_chars > 0:
            # En yaygın karakterleri ekle
            most_common_chars = sorted(char_freqs.items(), key=lambda x: -x[1])
            
            chars_to_add = []
            for char, freq in most_common_chars:
                if char not in covered_chars:
                    chars_to_add.append(char)
                    covered_chars.add(char)
                    
                # Kapsam hedefimize ulaştık mı kontrol et
                new_coverage = len(covered_chars) / total_chars
                if new_coverage >= self.character_coverage:
                    break
                    
            logger.info(f"Karakter kapsamı: {new_coverage:.4f} ({len(covered_chars)}/{total_chars})")
            
        # Tüm adayları birleştir
        all_pieces = ngram_freqs.copy()
        
        # Tekli karakterleri ekle
        for char in covered_chars:
            if char not in all_pieces:
                all_pieces[char] = char_freqs[char]
                
        return all_pieces
    
    def _prune_piece_candidates(
        self, 
        piece_freqs: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Alt kelime adaylarını budama.
        
        Args:
            piece_freqs: Alt kelime frekansları
            
        Returns:
            Dict[str, int]: Budanmış alt kelime frekansları
        """
        # Min frekans filtrelemesi
        pruned_pieces = {
            piece: freq for piece, freq in piece_freqs.items()
            if freq >= self.min_frequency
        }
        
        # En yaygın alt kelimeleri seç
        sorted_pieces = sorted(
            pruned_pieces.items(),
            key=lambda x: (-x[1], len(x[0]), x[0])  # -frekans, uzunluk, kelime (yüksek frekans önce)
        )
        
        # Sözlük boyutunu sınırla
        vocab_size = min(self.vocab_size, len(sorted_pieces))
        final_pieces = {piece: freq for piece, freq in sorted_pieces[:vocab_size]}
        
        logger.info(f"{len(piece_freqs)} adaydan {len(final_pieces)} alt kelime seçildi")
        return final_pieces
    
    def _calculate_pi(self, freqs: Dict[str, int], total: int) -> Dict[str, float]:
        """
        Her token için olasılık hesaplar.
        
        Args:
            freqs: Token frekansları
            total: Toplam frekans
            
        Returns:
            Dict[str, float]: Token olasılıkları
        """
        # Basit olasılık hesabı (frekans / toplam)
        return {piece: freq / total for piece, freq in freqs.items()}
    
    def _viterbi_segmentation(self, text: str) -> List[str]:
        """
        Viterbi algoritması ile parçalama yapar.
        
        Args:
            text: Parçalanacak metin
            
        Returns:
            List[str]: En olası parçalama
        """
        if not text:
            return []
            
        # Önbellekte varsa kullan
        if text in self.cache:
            return self.cache[text]
            
        n = len(text)
        
        # Best_score[i]: text[0:i]'nin en iyi puanı
        best_score = [float('-inf')] * (n + 1)
        best_score[0] = 0
        
        # Best_edge[i]: text[0:i]'nin en iyi ayrım noktası
        best_edge = [0] * (n + 1)
        
        # Viterbi algoritması
        for i in range(1, n + 1):
            # En iyi ayrım noktasını bul
            for j in range(max(0, i - 16), i):  # En fazla 16 karakter geriye git
                piece = text[j:i]
                
                if piece in self.scores:
                    score = best_score[j] + self.scores[piece]
                    
                    if score > best_score[i]:
                        best_score[i] = score
                        best_edge[i] = j
        
        # En iyi parçalamayı oluştur
        pieces = []
        i = n
        
        while i > 0:
            j = best_edge[i]
            piece = text[j:i]
            pieces.append(piece)
            i = j
            
        pieces.reverse()
        
        # Sonucu önbelleğe al
        self.cache[text] = pieces
        return pieces
    
    def train(
        self,
        texts: List[str],
        trainer: Optional[TokenizerTrainer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unigram modelini verilen metinler üzerinde eğitir.
        
        Args:
            texts: Eğitim metinleri
            trainer: Eğitimi yönetecek TokenizerTrainer nesnesi (varsayılan: None)
            **kwargs: Eğitim için ek parametreler
            
        Returns:
            Dict[str, Any]: Eğitim istatistikleri
        """
        # Varsayılan eğitim yapılandırması
        num_iterations = kwargs.get("num_iterations", 10)
        shrinking_factor = kwargs.get("shrinking_factor", 0.75)
        
        # Eğitim yöneticisini yapılandır
        if trainer is None:
            trainer = TokenizerTrainer(
                batch_size=kwargs.get("batch_size", 1000),
                num_iterations=num_iterations,
                show_progress=kwargs.get("show_progress", True)
            )
        
        # Eğitim başlangıcı
        trainer.on_training_begin(self, texts)
        
        # Alt kelime adaylarını topla
        piece_freqs = self._gather_piece_candidates(texts)
        initial_pieces = len(piece_freqs)
        
        # Özel tokenları ekle
        for token in self.special_tokens.values():
            if token not in piece_freqs:
                piece_freqs[token] = self.min_frequency
        
        # Toplam alt kelime sayısını hesapla
        total_freq = sum(piece_freqs.values())
        
        # Hedef sözlük boyutu
        target_vocab_size = self.vocab_size
        current_vocab_size = len(piece_freqs)
        
        logger.info(f"Başlangıç alt kelime sayısı: {current_vocab_size}")
        
        # Unigram modelinin iteratif eğitimi
        for iteration in range(num_iterations):
            trainer.on_iteration_begin(self, iteration)
            
            # Alt kelime olasılıklarını hesapla
            pi = self._calculate_pi(piece_freqs, total_freq)
            
            # Log olasılıkları hesapla ve sözlüğü oluştur
            self.scores = {piece: math.log(prob) for piece, prob in pi.items()}
            self.pieces = list(self.scores.keys())
            
            # Token -> ID eşleştirmelerini oluştur
            self.vocab = {piece: idx for idx, piece in enumerate(self.pieces)}
            self.ids_to_pieces = {idx: piece for piece, idx in self.vocab.items()}
            
            # Önbelleği temizle
            self.cache = {}
            
            # Eğer son iterasyonsa işlemi sonlandır
            if iteration >= num_iterations - 1:
                break
                
            # Parça frekanslarını yeniden hesapla
            new_piece_freqs = defaultdict(int)
            
            # Her örnek için en olası parçalamayı bul
            for text in texts:
                text = self._normalize_text(text)
                segments = self._split_text(text)
                
                for segment in segments:
                    # Viterbi algoritması ile parçalara ayır
                    pieces = self._viterbi_segmentation(segment)
                    
                    # Frekans tablosunu güncelle
                    for piece in pieces:
                        new_piece_freqs[piece] += 1
            
            # En az kullanılan alt kelimeleri kaldır
            if len(new_piece_freqs) > target_vocab_size:
                # Azaltılacak sözcük sayısını hesapla
                n_to_remove = int((1.0 - shrinking_factor) * len(new_piece_freqs))
                n_to_remove = max(1, min(n_to_remove, len(new_piece_freqs) - target_vocab_size))
                
                # Yardımcı fonksiyon: alt kelimenin şu anki puanını hesapla
                def get_piece_score(piece, freq):
                    logp = self.scores.get(piece, float('-inf'))
                    return logp * freq
                
                # Puanlarına göre alt kelimeleri sırala
                pieces_with_scores = [
                    (piece, freq, get_piece_score(piece, freq))
                    for piece, freq in new_piece_freqs.items()
                ]
                
                # Özel tokenlar hariç tüm alt kelimeleri puanına göre sırala
                protected_pieces = set(self.special_tokens.values())
                
                filtered_pieces = [
                    (piece, freq, score) for piece, freq, score in pieces_with_scores
                    if piece not in protected_pieces
                ]
                
                filtered_pieces.sort(key=lambda x: x[2])  # Puanına göre sırala
                
                # En düşük puanlı alt kelimeleri kaldır
                for piece, _, _ in filtered_pieces[:n_to_remove]:
                    del new_piece_freqs[piece]
                    
            # Frekans tablosunu güncelle
            piece_freqs = new_piece_freqs
            total_freq = sum(piece_freqs.values())
            current_vocab_size = len(piece_freqs)
            
            # Özel tokenları geri ekle
            for token in self.special_tokens.values():
                if token not in piece_freqs:
                    piece_freqs[token] = self.min_frequency
            
            # İlerleme istatistikleri
            metrics = {
                "vocab_size": current_vocab_size,
                "shrink_factor": shrinking_factor,
                "target_vocab_size": target_vocab_size
            }
            
            trainer.on_iteration_end(self, iteration, metrics)
        
        # Son olasılıkları hesapla
        pi = self._calculate_pi(piece_freqs, total_freq)
        
        # Log olasılıkları hesapla ve sözlüğü oluştur
        self.scores = {piece: math.log(prob) for piece, prob in pi.items()}
        self.pieces = list(self.scores.keys())
        
        # Token -> ID eşleştirmelerini oluştur
        self.vocab = {piece: idx for idx, piece in enumerate(self.pieces)}
        self.ids_to_pieces = {idx: piece for piece, idx in self.vocab.items()}
        
        # Eğitimi tamamla
        self.is_trained = True
        final_metrics = {
            "vocab_size": len(self.vocab),
            "initial_pieces": initial_pieces,
            "final_pieces": len(self.pieces),
            "num_iterations": num_iterations
        }
        
        # Eğitim meta verilerini güncelle
        self.metadata.update({
            "training_size": len(texts),
            "character_coverage": self.character_coverage,
            "split_by_whitespace": self.split_by_whitespace,
            "normalization_rule": self.normalization_rule
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
                bos_token = self.special_tokens.get("bos_token")
                eos_token = self.special_tokens.get("eos_token")
                
                if bos_token and bos_token in self.vocab:
                    result.append(self.vocab[bos_token])
                if eos_token and eos_token in self.vocab:
                    result.append(self.vocab[eos_token])
            
            self.stats["num_encode_calls"] += 1
            self.stats["total_encode_time"] += time.time() - start_time
            return result
            
        # Metni normalleştir
        text = self._normalize_text(text)
        
        # Boşluklarla ayır
        segments = self._split_text(text)
        token_ids = []
        
        # Başlangıç tokeni ekle
        if add_special_tokens and "bos_token" in self.special_tokens:
            bos_token = self.special_tokens["bos_token"]
            if bos_token in self.vocab:
                token_ids.append(self.vocab[bos_token])
        
        # Her segmenti tokenize et
        unk_id = self.vocab.get(self.special_tokens["unk_token"], 0)
        
        for segment in segments:
            if not segment:
                continue
                
            # Viterbi algoritması ile parçalama
            pieces = self._viterbi_segmentation(segment)
            
            # Her parça için ID'yi bul
            for piece in pieces:
                token_ids.append(self.vocab.get(piece, unk_id))
                
        # Bitiş tokeni ekle
        if add_special_tokens and "eos_token" in self.special_tokens:
            eos_token = self.special_tokens["eos_token"]
            if eos_token in self.vocab:
                token_ids.append(self.vocab[eos_token])
        
        # İstatistikleri güncelle
        self.stats["num_encode_calls"] += 1
        self.stats["total_encode_time"] += time.time() - start_time
        
        return token_ids
    
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
        special_tokens_set = set()
        if skip_special_tokens:
            for token_type, token in self.special_tokens.items():
                if token in self.vocab:
                    special_tokens_set.add(self.vocab[token])
        
        # Token ID'lerini tokenlere dönüştür
        pieces = []
        
        for token_id in ids:
            if token_id in special_tokens_set:
                continue
                
            if token_id in self.ids_to_pieces:
                pieces.append(self.ids_to_pieces[token_id])
            else:
                pieces.append(self.special_tokens["unk_token"])
        
        # Tokenleri birleştir
        text = "".join(pieces)
        
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
            
        # Metni normalleştir
        text = self._normalize_text(text)
        
        # Boşluklarla ayır
        segments = self._split_text(text)
        tokens = []
        
        for segment in segments:
            if not segment:
                continue
                
            # Viterbi algoritması ile parçalama
            pieces = self._viterbi_segmentation(segment)
            tokens.extend(pieces)
            
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
            "type": "UnigramTokenizer",
            "pieces": self.pieces,
            "scores": self.scores,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "character_coverage": self.character_coverage,
            "split_pattern": self.split_pattern,
            "split_by_whitespace": self.split_by_whitespace,
            "normalization_rule": self.normalization_rule,
            "treat_whitespace_as_suffix": self.treat_whitespace_as_suffix,
            "metadata": self.metadata
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Tokenizer {save_path} konumuna kaydedildi")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "UnigramTokenizer":
        """
        Tokenizer modelini yükler.
        
        Args:
            path: Yükleme yolu
            **kwargs: Yükleme için ek parametreler
            
        Returns:
            UnigramTokenizer: Yüklenen tokenizer modeli
            
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
        if data.get("type") != "UnigramTokenizer":
            logger.warning(f"Yüklenen tokenizer türü uyumsuz. Beklenen: UnigramTokenizer, Alınan: {data.get('type')}")
            
        # Tokenizer'ı yapılandır
        tokenizer = cls(
            vocab_size=data.get("vocab_size", 30000),
            min_frequency=data.get("min_frequency", 2),
            special_tokens=data.get("special_tokens"),
            character_coverage=data.get("character_coverage", 0.9995),
            split_pattern=data.get("split_pattern", r"\s+"),
            split_by_whitespace=data.get("split_by_whitespace", True),
            normalization_rule=data.get("normalization_rule", "nmt_nfkc"),
            treat_whitespace_as_suffix=data.get("treat_whitespace_as_suffix", False)
        )
        
        # Model verilerini yükle
        tokenizer.pieces = data.get("pieces", [])
        tokenizer.scores = data.get("scores", {})
        
        # Sözlüğü oluştur
        tokenizer.vocab = {piece: idx for idx, piece in enumerate(tokenizer.pieces)}
        tokenizer.ids_to_pieces = {idx: piece for piece, idx in tokenizer.vocab.items()}
        
        # Meta verileri yükle
        if "metadata" in data:
            tokenizer.metadata.update(data["metadata"])
            
        # Eğitilmiş olarak işaretle
        tokenizer.is_trained = True
        
        logger.info(f"Tokenizer {load_path} konumundan yüklendi")
        return tokenizer