# aiquantr_tokenizer/tokenizers/sentencepiece.py
"""
SentencePiece tokenizer sarmalayıcı uygulaması.

Bu modül, Google SentencePiece kütüphanesinin
Unigram ve BPE modellerini kullanan bir sarmalayıcı sınıf sağlar.
"""

import os
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Iterator, Callable

from .base import BaseTokenizer, TokenizerTrainer

# Logger oluştur
logger = logging.getLogger(__name__)


class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece tokenizer için sarmalayıcı sınıf.
    
    Bu sınıf, Google'ın SentencePiece kütüphanesini kullanarak
    alt kelime tokenizasyonu için bir arabirim sağlar.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        split_by_whitespace: bool = True,
        add_dummy_prefix: bool = True,
        remove_extra_whitespaces: bool = True,
        normalization_rule_name: str = "nmt_nfkc",
        user_defined_symbols: Optional[List[str]] = None,
        byte_fallback: bool = False,
        pretrained_model_path: Optional[Union[str, Path]] = None,
        name: Optional[str] = None
    ):
        """
        SentencePieceTokenizer sınıfı başlatıcısı.
        
        Args:
            vocab_size: Sözlük boyutu (varsayılan: 30000)
            min_frequency: Minimum token frekansı (varsayılan: 2)
            special_tokens: Özel token eşlemeleri (varsayılan: None)
            model_type: "unigram" veya "bpe" (varsayılan: "unigram")
            character_coverage: Karakter kapsama oranı (varsayılan: 0.9995)
            split_by_whitespace: Boşlukla ayır (varsayılan: True)
            add_dummy_prefix: Dummy önek ekle (varsayılan: True)
            remove_extra_whitespaces: Fazla boşlukları kaldır (varsayılan: True)
            normalization_rule_name: Normalizasyon kuralı (varsayılan: "nmt_nfkc")
            user_defined_symbols: Kullanıcı tanımlı semboller (varsayılan: None)
            byte_fallback: Bilinmeyen karakterler için byte geri dönüşü kullan (varsayılan: False)
            pretrained_model_path: Önceden eğitilmiş model yolu (varsayılan: None)
            name: Tokenizer adı (varsayılan: None)
            
        Raises:
            ImportError: sentencepiece paketi kurulu değilse
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "SentencePieceTokenizer kullanmak için 'sentencepiece' paketi gereklidir. "
                "Lütfen 'pip install sentencepiece' komutuyla yükleyin."
            )
            
        super().__init__(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            name=name or "SentencePieceTokenizer"
        )
        
        # SentencePiece yapılandırması
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.split_by_whitespace = split_by_whitespace
        self.add_dummy_prefix = add_dummy_prefix
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.normalization_rule_name = normalization_rule_name
        self.user_defined_symbols = user_defined_symbols or []
        self.byte_fallback = byte_fallback
        
        # SentencePiece modeli
        self.sp_model = None
        
        # Önceden eğitilmiş model yükleme
        if pretrained_model_path:
            self.load(pretrained_model_path)
            
    def _map_special_tokens(self) -> Dict[str, int]:
        """
        Özel tokenları SentencePiece ID'lerine eşler.
        
        Returns:
            Dict[str, int]: Özel token -> ID eşlemesi
        """
        result = {}
        
        if not self.sp_model:
            return result
            
        # Özel token -> ID eşleştirmelerini bul
        for token_type, token in self.special_tokens.items():
            sp_id = self.sp_model.piece_to_id(token)
            if sp_id != self.sp_model.unk_id():
                result[token_type] = sp_id
            else:
                logger.warning(f"'{token}' özel tokeni SentencePiece modelinde bulunamadı")
                
        return result
    
    def train(
        self,
        texts: List[str],
        trainer: Optional[TokenizerTrainer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        SentencePiece modelini verilen metinler üzerinde eğitir.
        
        Args:
            texts: Eğitim metinleri
            trainer: Eğitimi yönetecek TokenizerTrainer nesnesi (varsayılan: None)
            **kwargs: Eğitim için ek parametreler
            
        Returns:
            Dict[str, Any]: Eğitim istatistikleri
            
        Raises:
            RuntimeError: Eğitim sırasında hata oluşursa
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Bu işlev için 'sentencepiece' paketi gereklidir.")
            
        # Eğitim yöneticisini yapılandır
        if trainer is None:
            trainer = TokenizerTrainer(
                batch_size=kwargs.get("batch_size", 1000),
                num_iterations=1,  # SentencePiece tek adımda eğitilir
                show_progress=kwargs.get("show_progress", True)
            )
            
        # Özel bir eğitim ekstra parametreleri
        extra_options = kwargs.get("extra_options", "")
        model_prefix = kwargs.get("model_prefix", "tokenizer_model")
        
        # Eğitim başlangıcı
        trainer.on_training_begin(self, texts)
        trainer.on_iteration_begin(self, 0)
        
        # Geçici eğitim dosyası oluştur
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
            corpus_file = f.name
            for text in texts:
                f.write(f"{text}\n")
                
        # Geçici model dosyaları
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / model_prefix
            model_prefix_str = str(model_path)
            
            # Özel token yolları yapılandırması
            special_tokens_str = ""
            if self.user_defined_symbols or self.special_tokens:
                all_special_tokens = list(self.user_defined_symbols)
                all_special_tokens.extend(self.special_tokens.values())
                special_tokens_str = f"--user_defined_symbols={','.join(all_special_tokens)}"
                
            # SentencePiece eğitim komutu
            cmd = (
                f"--input={corpus_file} "
                f"--model_prefix={model_prefix_str} "
                f"--vocab_size={self.vocab_size} "
                f"--model_type={self.model_type} "
                f"--character_coverage={self.character_coverage} "
                f"--input_sentence_size=0 "  # tüm veri setini kullan
                f"--shuffle_input_sentence=true "
                f"--input_format=text "
                f"--minloglevel=1 "  # INFO seviyesi
                f"--hard_vocab_limit=false "  # sözlük boyutunu biraz aşmaya izin ver
            )
            
            # Ek seçenekleri ekle
            if self.min_frequency > 1:
                cmd += f"--min_frequency={self.min_frequency} "
                
            if special_tokens_str:
                cmd += f"{special_tokens_str} "
                
            cmd += (
                f"--split_by_whitespace={str(self.split_by_whitespace).lower()} "
                f"--add_dummy_prefix={str(self.add_dummy_prefix).lower()} "
                f"--remove_extra_whitespaces={str(self.remove_extra_whitespaces).lower()} "
                f"--normalization_rule_name={self.normalization_rule_name} "
                f"--byte_fallback={str(self.byte_fallback).lower()} "
            )
            
            # Ek kullanıcı seçenekleri
            if extra_options:
                cmd += f"{extra_options} "
                
            try:
                # SentencePiece modelini eğit
                logger.info(f"SentencePiece eğitimi başlatılıyor: {cmd}")
                spm.SentencePieceTrainer.train(cmd)
                
                # Eğitilmiş modeli yükle
                model_file = f"{model_prefix_str}.model"
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.load(model_file)
                
                # Model meta verilerini çıkar
                vocab_size = self.sp_model.get_piece_size()
                
                # Eğitim meta verilerini güncelle
                self.metadata.update({
                    "training_size": len(texts),
                    "model_type": self.model_type,
                    "character_coverage": self.character_coverage,
                    "split_by_whitespace": self.split_by_whitespace,
                    "normalization_rule_name": self.normalization_rule_name
                })
                
                # İlerleme istatistikleri
                metrics = {
                    "vocab_size": vocab_size,
                    "unk_id": self.sp_model.unk_id(),
                    "bos_id": self.sp_model.bos_id(),
                    "eos_id": self.sp_model.eos_id(),
                    "pad_id": self.sp_model.pad_id(),
                }
                
                trainer.on_iteration_end(self, 0, metrics)
                
                # Eğitilmiş modeli kaydet
                self.is_trained = True
                
                # Geçici dosyaları temizle
                os.unlink(corpus_file)
                
            except Exception as e:
                # Geçici dosyaları temizle
                os.unlink(corpus_file)
                raise RuntimeError(f"SentencePiece eğitimi başarısız: {e}")
        
        # Eğitimi tamamla
        final_metrics = metrics.copy()
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
            
        Raises:
            RuntimeError: Model eğitilmemişse
        """
        if not self.sp_model:
            raise RuntimeError("Kodlama yapmadan önce bir model eğitin veya yükleyin")
            
        start_time = time.time()
        
        # Özel parametreler
        enable_sampling = kwargs.get("enable_sampling", False)
        alpha = kwargs.get("alpha", 0.1)
        nbest_size = kwargs.get("nbest_size", -1)
        
        # Boş metin kontrolü
        if not text:
            result = []
            
            if add_special_tokens:
                if self.sp_model.bos_id() > 0:
                    result.append(self.sp_model.bos_id())
                if self.sp_model.eos_id() > 0:
                    result.append(self.sp_model.eos_id())
            
            self.stats["num_encode_calls"] += 1
            self.stats["total_encode_time"] += time.time() - start_time
            return result
            
        # Kodlama seçeneklerini belirle
        encode_options = {}
        
        if enable_sampling:
            encode_options["enable_sampling"] = True
            encode_options["alpha"] = alpha
            encode_options["nbest_size"] = nbest_size
            
        # SentencePiece ile kodlama
        if add_special_tokens:
            token_ids = self.sp_model.encode(text, add_bos=True, add_eos=True, **encode_options)
        else:
            token_ids = self.sp_model.encode(text, add_bos=False, add_eos=False, **encode_options)
            
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
            
        Raises:
            RuntimeError: Model eğitilmemişse
        """
        if not self.sp_model:
            raise RuntimeError("Kod çözme yapmadan önce bir model eğitin veya yükleyin")
            
        start_time = time.time()
        
        if not ids:
            self.stats["num_decode_calls"] += 1
            self.stats["total_decode_time"] += time.time() - start_time
            return ""
            
        # Özel tokenları atla
        filtered_ids = ids
        if skip_special_tokens:
            special_ids = {
                self.sp_model.unk_id(),
                self.sp_model.bos_id(),
                self.sp_model.eos_id(),
                self.sp_model.pad_id()
            }
            filtered_ids = [id_ for id_ in ids if id_ not in special_ids and id_ >= 0]
            
        # Kod çözme
        text = self.sp_model.decode(filtered_ids)
        
        # İstatistikleri güncelle
        self.stats["num_decode_calls"] += 1
        self.stats["total_decode_time"] += time.time() - start_time
        
        return text
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Tokenizer'ın sözlüğünü döndürür.
        
        Returns:
            Dict[str, int]: Token - ID eşleşmelerini içeren sözlük
            
        Raises:
            RuntimeError: Model eğitilmemişse
        """
        if not self.sp_model:
            raise RuntimeError("Sözlük almadan önce bir model eğitin veya yükleyin")
            
        vocab = {}
        for i in range(self.sp_model.get_piece_size()):
            piece = self.sp_model.id_to_piece(i)
            vocab[piece] = i
            
        return vocab
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Metni token dizilerine dönüştürür (ID'lere kodlamadan).
        
        Args:
            text: Tokenize edilecek metin
            **kwargs: Tokenize için ek parametreler
            
        Returns:
            List[str]: Token dizileri
            
        Raises:
            RuntimeError: Model eğitilmemişse
        """
        if not self.sp_model:
            raise RuntimeError("Tokenize yapmadan önce bir model eğitin veya yükleyin")
            
        if not text:
            return []
            
        # Özel parametreler
        add_bos = kwargs.get("add_bos", False)
        add_eos = kwargs.get("add_eos", False)
        
        return self.sp_model.encode_as_pieces(text, add_bos=add_bos, add_eos=add_eos)
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Tokenizer modelini kaydeder.
        
        Args:
            path: Kaydetme yolu
            **kwargs: Kaydetme için ek parametreler
            
        Raises:
            RuntimeError: Model eğitilmemişse
        """
        if not self.sp_model:
            raise RuntimeError("Kaydetmeden önce bir model eğitin veya yükleyin")
            
        path = Path(path)
        
        # Ana dizini oluştur
        if path.is_dir():
            model_path = path / "tokenizer.model"
            config_path = path / "tokenizer.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            model_path = path.with_suffix(".model")
            config_path = path.with_suffix(".json")
            
        # SentencePiece model dosyasını kaydet
        self.sp_model.save(str(model_path))
        
        # Yapılandırma bilgilerini kaydet
        config = {
            "type": "SentencePieceTokenizer",
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "character_coverage": self.character_coverage,
            "split_by_whitespace": self.split_by_whitespace,
            "add_dummy_prefix": self.add_dummy_prefix,
            "remove_extra_whitespaces": self.remove_extra_whitespaces,
            "normalization_rule_name": self.normalization_rule_name,
            "user_defined_symbols": self.user_defined_symbols,
            "byte_fallback": self.byte_fallback,
            "metadata": self.metadata,
            "model_path": str(model_path.name)
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Tokenizer {model_path} ve {config_path} konumlarına kaydedildi")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "SentencePieceTokenizer":
        """
        Tokenizer modelini yükler.
        
        Args:
            path: Yükleme yolu
            **kwargs: Yükleme için ek parametreler
            
        Returns:
            SentencePieceTokenizer: Yüklenen tokenizer modeli
            
        Raises:
            ValueError: Model yüklenemezse
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Bu işlev için 'sentencepiece' paketi gereklidir.")
            
        path = Path(path)
        
        # Dosya türünü belirle
        if path.is_dir():
            config_path = path / "tokenizer.json"
            model_path = path / "tokenizer.model"
        else:
            if path.suffix == ".model":
                model_path = path
                config_path = path.with_suffix(".json")
            else:
                config_path = path
                model_path = path.with_suffix(".model")
                
        # Yapılandırma dosyasını kontrol et
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            # Model türünü doğrula
            if config.get("type") != "SentencePieceTokenizer":
                logger.warning(
                    f"Yüklenen tokenizer türü uyumsuz. Beklenen: SentencePieceTokenizer, "
                    f"Alınan: {config.get('type')}"
                )
                
            # Model yolu ayarlanmışsa, güncelle
            if "model_path" in config and not model_path.exists():
                model_dir = path if path.is_dir() else path.parent
                model_path = model_dir / config["model_path"]
        
        # Model dosyasını kontrol et
        if not model_path.exists():
            raise ValueError(f"SentencePiece model dosyası bulunamadı: {model_path}")
            
        # Yapılandırma parametrelerini çıkar
        model_type = config.get("model_type", "unigram")
        vocab_size = config.get("vocab_size", 30000)
        min_frequency = config.get("min_frequency", 2)
        special_tokens = config.get("special_tokens")
        character_coverage = config.get("character_coverage", 0.9995)
        split_by_whitespace = config.get("split_by_whitespace", True)
        add_dummy_prefix = config.get("add_dummy_prefix", True)
        remove_extra_whitespaces = config.get("remove_extra_whitespaces", True)
        normalization_rule_name = config.get("normalization_rule_name", "nmt_nfkc")
        user_defined_symbols = config.get("user_defined_symbols")
        byte_fallback = config.get("byte_fallback", False)
        
        # Tokenizer'ı oluştur
        tokenizer = cls(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            model_type=model_type,
            character_coverage=character_coverage,
            split_by_whitespace=split_by_whitespace,
            add_dummy_prefix=add_dummy_prefix,
            remove_extra_whitespaces=remove_extra_whitespaces,
            normalization_rule_name=normalization_rule_name,
            user_defined_symbols=user_defined_symbols,
            byte_fallback=byte_fallback
        )
        
        # SentencePiece modelini yükle
        tokenizer.sp_model = spm.SentencePieceProcessor()
        try:
            tokenizer.sp_model.load(str(model_path))
        except Exception as e:
            raise ValueError(f"SentencePiece model yüklenirken hata: {e}")
            
        # Meta verileri yükle
        if "metadata" in config:
            tokenizer.metadata.update(config["metadata"])
            
        # Eğitilmiş olarak işaretle
        tokenizer.is_trained = True
        
        logger.info(f"Tokenizer {path} konumundan yüklendi")
        return tokenizer