# aiquantr_tokenizer/tokenizers/mixed.py
"""
Karma tokenizer uygulaması.

Bu modül, birden fazla tokenizer'ı bir araya getirerek
çeşitli veri türleri için özelleştirilmiş karma bir tokenizer sağlar.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Iterator, Callable

from .base import BaseTokenizer, TokenizerTrainer

# Logger oluştur
logger = logging.getLogger(__name__)


class MixedTokenizer(BaseTokenizer):
    """
    Karma tokenizer implementasyonu.
    
    Bu sınıf, farklı veri türleri veya görevler için
    birden fazla tokenizer'ı birleştirmeye olanak tanır.
    """
    
    def __init__(
        self,
        tokenizers: Dict[str, BaseTokenizer],
        default_tokenizer: str,
        router: Optional[Callable[[str], str]] = None,
        merged_vocab: bool = False,
        special_tokens: Optional[Dict[str, str]] = None,
        name: Optional[str] = None
    ):
        """
        MixedTokenizer sınıfı başlatıcısı.
        
        Args:
            tokenizers: Tokenizer adı -> tokenizer eşleştirmesi
            default_tokenizer: Varsayılan tokenizer'ın adı
            router: Metin -> tokenizer adı eşleştirmesi yapan fonksiyon (varsayılan: None)
            merged_vocab: Sözlükleri birleştir (varsayılan: False)
            special_tokens: Özel token eşlemeleri (varsayılan: None)
            name: Tokenizer adı (varsayılan: None)
            
        Raises:
            ValueError: Tokenizer'lar geçersizse
        """
        if not tokenizers:
            raise ValueError("En az bir tokenizer gereklidir")
            
        if default_tokenizer not in tokenizers:
            raise ValueError(f"Varsayılan tokenizer '{default_tokenizer}' tanımlanan tokenizer'lar arasında değil")
            
        # Varsayılan tokenizer'dan sözlük boyutunu ve min frekansı al
        default_t = tokenizers[default_tokenizer]
        vocab_size = default_t.vocab_size
        min_frequency = default_t.min_frequency
            
        super().__init__(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            name=name or "MixedTokenizer"
        )
        
        self.tokenizers = tokenizers
        self.default_tokenizer = default_tokenizer
        self.router = router
        self.merged_vocab = merged_vocab
        
        # ID aralıklarını ayır
        self.id_ranges = {}
        self.id_to_tokenizer = {}  # ID -> tokenizer adı eşlemesi
        
        # Birleştirilmiş sözlük için gerekli değişkenler
        self.merged_id_map = {}  # (tokenizer_name, local_id) -> global_id
        self.merged_reverse_map = {}  # global_id -> (tokenizer_name, local_id)
        
        # Birleştirilmiş sözlüğü oluştur
        if merged_vocab:
            self._build_merged_vocabulary()
    
    def _build_merged_vocabulary(self):
        """
        Tüm tokenizer'ların sözlüklerini birleştirir ve ID eşlemelerini oluşturur.
        """
        # Tüm sözlükleri birleştir
        next_id = 0
        
        # Önce özel tokenları işle
        special_token_set = set()
        for token_type, token in self.special_tokens.items():
            if token not in special_token_set:
                self.merged_id_map[("__special__", token)] = next_id
                self.merged_reverse_map[next_id] = ("__special__", token)
                special_token_set.add(token)
                next_id += 1
        
        # Her tokenizer için
        for tokenizer_name, tokenizer in self.tokenizers.items():
            self.id_ranges[tokenizer_name] = (next_id, None)  # son sınır daha sonra doldurulacak
            
            vocab = tokenizer.get_vocab()
            
            # Yerel ID'leri global ID'lere eşle
            for token, local_id in vocab.items():
                # Özel tokenlar zaten eklenmiş mi kontrol et
                if token in special_token_set:
                    continue
                    
                self.merged_id_map[(tokenizer_name, local_id)] = next_id
                self.merged_reverse_map[next_id] = (tokenizer_name, local_id)
                next_id += 1
                
            # ID aralığının son sınırını güncelle
            self.id_ranges[tokenizer_name] = (self.id_ranges[tokenizer_name][0], next_id - 1)
            
        logger.info(f"Karma sözlük oluşturuldu: {next_id} token")
    
    def _route_text(self, text: str) -> str:
        """
        Metni uygun tokenizer'a yönlendirir.
        
        Args:
            text: Yönlendirilecek metin
            
        Returns:
            str: Seçilen tokenizer'ın adı
        """
        if self.router is not None:
            try:
                tokenizer_name = self.router(text)
                if tokenizer_name in self.tokenizers:
                    return tokenizer_name
            except Exception as e:
                logger.warning(f"Yönlendirici hata verdi: {e}")
                
        # Varsayılan tokenizer'a geri dön
        return self.default_tokenizer
    
    def train(
        self,
        texts: List[str],
        trainer: Optional[TokenizerTrainer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tüm tokenizer'ları verilen metinler üzerinde eğitir.
        
        Args:
            texts: Eğitim metinleri
            trainer: Eğitimi yönetecek TokenizerTrainer nesnesi (varsayılan: None)
            **kwargs: Eğitim için ek parametreler
            
        Returns:
            Dict[str, Any]: Eğitim istatistikleri
        """
        # Eğitim yapılandırması
        tokenizer_datasets = kwargs.get("tokenizer_datasets", {})
        
        # Varsayılan eğitim yöneticisi
        if trainer is None:
            trainer = TokenizerTrainer(
                batch_size=kwargs.get("batch_size", 1000),
                num_iterations=1,
                show_progress=kwargs.get("show_progress", True)
            )
            
        # Eğitim başlangıcı
        trainer.on_training_begin(self, texts)
        trainer.on_iteration_begin(self, 0)
        
        training_results = {}
        
        # Her tokenizer'ı eğit
        for tokenizer_name, tokenizer in self.tokenizers.items():
            # Tokenizer'a özel veri kümesi var mı kontrol et
            tokenizer_texts = tokenizer_datasets.get(tokenizer_name, texts)
            
            logger.info(f"'{tokenizer_name}' tokenizer'ı eğitiliyor: {len(tokenizer_texts)} örnek")
            result = tokenizer.train(tokenizer_texts, trainer=None, **kwargs)
            training_results[tokenizer_name] = result
            
        # Birleştirilmiş sözlüğü güncelle
        if self.merged_vocab:
            self._build_merged_vocabulary()
            
        # İlerleme metrikleri
        total_vocab_size = 0
        for tokenizer_name, tokenizer in self.tokenizers.items():
            total_vocab_size += tokenizer.get_vocab_size()
            
        metrics = {
            "total_vocab_size": total_vocab_size,
            "num_tokenizers": len(self.tokenizers)
        }
        
        # Eğitim meta verilerini güncelle
        self.metadata.update({
            "training_size": len(texts),
            "tokenizers": list(self.tokenizers.keys()),
            "default_tokenizer": self.default_tokenizer,
            "merged_vocab": self.merged_vocab
        })
        
        trainer.on_iteration_end(self, 0, metrics)
        
        # Eğitimi tamamla
        self.is_trained = True
        final_metrics = metrics.copy()
        final_metrics["training_results"] = training_results
        
        trainer.on_training_end(self, final_metrics)
        return final_metrics
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        tokenizer_name: Optional[str] = None,
        **kwargs
    ) -> List[int]:
        """
        Metni token ID'lerine dönüştürür.
        
        Args:
            text: Encode edilecek metin
            add_special_tokens: Başlangıç/bitiş tokenlarını ekle (varsayılan: True)
            tokenizer_name: Kullanılacak tokenizer (varsayılan: None - otomatik seçim)
            **kwargs: Encode için ek parametreler
            
        Returns:
            List[int]: Token ID'leri
        """
        start_time = time.time()
        
        # Uygun tokenizer'ı seç
        if tokenizer_name is None:
            tokenizer_name = self._route_text(text)
            
        if tokenizer_name not in self.tokenizers:
            tokenizer_name = self.default_tokenizer
            
        tokenizer = self.tokenizers[tokenizer_name]
        
        # Seçilen tokenizer ile metni kodla
        local_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        
        # Yerel ID'leri global ID'lere dönüştür (eğer gerekiyorsa)
        if self.merged_vocab:
            global_ids = [
                self.merged_id_map.get((tokenizer_name, local_id), 0)
                for local_id in local_ids
            ]
            token_ids = global_ids
        else:
            token_ids = local_ids
            
        # İstatistikleri güncelle
        self.stats["num_encode_calls"] += 1
        self.stats["total_encode_time"] += time.time() - start_time
        
        return token_ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
        tokenizer_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Token ID'lerini metne dönüştürür.
        
        Args:
            ids: Decode edilecek token ID'leri
            skip_special_tokens: Özel tokenları atla (varsayılan: True)
            tokenizer_name: Kullanılacak tokenizer (varsayılan: None - otomatik tespit)
            **kwargs: Decode için ek parametreler
            
        Returns:
            str: Elde edilen metin
        """
        start_time = time.time()
        
        if not ids:
            self.stats["num_decode_calls"] += 1
            self.stats["total_decode_time"] += time.time() - start_time
            return ""
            
        # Birleştirilmiş sözlük kullanılıyorsa
        if self.merged_vocab:
            # Token ID'lerini doğru tokenizer'a ayır
            tokenizer_to_local_ids = {}
            
            for global_id in ids:
                if global_id in self.merged_reverse_map:
                    t_name, local_id = self.merged_reverse_map[global_id]
                    
                    if t_name == "__special__":
                        # Özel tokenlar için varsayılan tokenizer'ı kullan
                        t_name = self.default_tokenizer
                        
                    if t_name not in tokenizer_to_local_ids:
                        tokenizer_to_local_ids[t_name] = []
                        
                    tokenizer_to_local_ids[t_name].append(local_id)
            
            # Her tokenizer için ilgili ID'lerin kod çözümünü yap
            text_parts = []
            
            for t_name, local_ids in tokenizer_to_local_ids.items():
                tokenizer = self.tokenizers[t_name]
                text_parts.append(tokenizer.decode(local_ids, skip_special_tokens, **kwargs))
                
            # Metinleri birleştir
            text = " ".join(text_parts)
            
        else:
            # Tokenizer adı belirtilmişse, kullan
            if tokenizer_name and tokenizer_name in self.tokenizers:
                tokenizer = self.tokenizers[tokenizer_name]
            else:
                tokenizer = self.tokenizers[self.default_tokenizer]
                
            # Doğrudan kod çözümü yap
            text = tokenizer.decode(ids, skip_special_tokens, **kwargs)
            
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
        if self.merged_vocab:
            # Birleştirilmiş sözlük
            vocab = {}
            
            # Tüm token ID eşlemelerini yeniden düzenle
            for global_id, (tokenizer_name, local_id) in self.merged_reverse_map.items():
                if tokenizer_name == "__special__":
                    # Özel tokenlar için doğrudan token adını kullan
                    token = local_id
                else:
                    # Yerel ID'yi token'a dönüştür
                    tokenizer = self.tokenizers[tokenizer_name]
                    token = tokenizer.id_to_token(local_id)
                    
                    # Tokenizer adını ekle
                    token = f"{tokenizer_name}:{token}"
                    
                vocab[token] = global_id
                
            return vocab
        else:
            # Varsayılan tokenizer sözlüğü
            return self.tokenizers[self.default_tokenizer].get_vocab()
    
    def tokenize(self, text: str, tokenizer_name: Optional[str] = None, **kwargs) -> List[str]:
        """
        Metni token dizilerine dönüştürür (ID'lere kodlamadan).
        
        Args:
            text: Tokenize edilecek metin
            tokenizer_name: Kullanılacak tokenizer (varsayılan: None - otomatik seçim)
            **kwargs: Tokenize için ek parametreler
            
        Returns:
            List[str]: Token dizileri
        """
        # Uygun tokenizer'ı seç
        if tokenizer_name is None:
            tokenizer_name = self._route_text(text)
            
        if tokenizer_name not in self.tokenizers:
            tokenizer_name = self.default_tokenizer
            
        tokenizer = self.tokenizers[tokenizer_name]
        
        # Seçilen tokenizer ile tokenize et
        return tokenizer.tokenize(text, **kwargs)
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Tokenizer modelini kaydeder.
        
        Args:
            path: Kaydetme yolu
            **kwargs: Kaydetme için ek parametreler
        """
        path = Path(path)
        
        # Ana dizini oluştur
        path.mkdir(parents=True, exist_ok=True)
        
        # Üst seviye yapılandırma
        config = {
            "type": "MixedTokenizer",
            "tokenizers": {},
            "default_tokenizer": self.default_tokenizer,
            "merged_vocab": self.merged_vocab,
            "special_tokens": self.special_tokens,
            "metadata": self.metadata
        }
        
        # Her tokenizer'ı kaydet
        for tokenizer_name, tokenizer in self.tokenizers.items():
            tokenizer_dir = path / tokenizer_name
            tokenizer.save(tokenizer_dir)
            
            # Tokenizer bilgilerini yapılandırmaya ekle - tip bilgisini doğru şekilde kaydet
            class_name = tokenizer.__class__.__name__
            tokenizer_type = class_name.lower().replace("tokenizer", "")  # örn: ByteLevelTokenizer -> bytelevel
            
            config["tokenizers"][tokenizer_name] = {
                "class": class_name,
                "type": tokenizer_type,  # type anahtarını ekle (factory.py'daki TOKENIZER_TYPES ile uyumlu olacak şekilde)
                "path": tokenizer_name
            }
            
        # ID eşlemelerini kaydet (birleştirilmiş sözlük için)
        if self.merged_vocab:
            merged_vocab_config = {
                "id_ranges": self.id_ranges,
                "merged_id_map": {f"{t}:{i}": g for (t, i), g in self.merged_id_map.items()},
                "merged_reverse_map": {str(g): [t, i] for g, (t, i) in self.merged_reverse_map.items()}
            }
            
            with open(path / "merged_vocab.json", "w", encoding="utf-8") as f:
                json.dump(merged_vocab_config, f, ensure_ascii=False, indent=2)
                
        # Üst seviye yapılandırmayı kaydet
        with open(path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Karma tokenizer {path} konumuna kaydedildi")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "MixedTokenizer":
        """
        Tokenizer modelini yükler.
        
        Args:
            path: Yükleme yolu
            **kwargs: Yükleme için ek parametreler
            
        Returns:
            MixedTokenizer: Yüklenen tokenizer modeli
            
        Raises:
            ValueError: Model yüklenemezse
        """
        from .factory import load_tokenizer_from_path, TOKENIZER_TYPES
        
        path = Path(path)
        
        # Yapılandırma dosyasını kontrol et
        config_path = path / "tokenizer.json"
        if not config_path.exists():
            raise ValueError(f"Tokenizer yapılandırma dosyası bulunamadı: {config_path}")
            
        # Yapılandırmayı yükle
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Model türünü doğrula
        if config.get("type") != "MixedTokenizer":
            logger.warning(f"Yüklenen tokenizer türü uyumsuz. Beklenen: MixedTokenizer, Alınan: {config.get('type')}")
            
        # Özel tokenları yükle
        special_tokens = config.get("special_tokens")
        
        # Alt tokenizer'ları yükle
        tokenizers = {}
        tokenizer_configs = config.get("tokenizers", {})
        
        for tokenizer_name, tokenizer_info in tokenizer_configs.items():
            tokenizer_path = path / tokenizer_info["path"]
            
            try:
                # İlk olarak direkt sınıf adını kullan
                class_name = tokenizer_info.get("class", "").lower()
                tokenizer_type = tokenizer_info.get("type", "").lower()
                
                # Factory'deki TOKENIZER_TYPES anahtarlarıyla eşleşmesi için düzenle
                if tokenizer_type.endswith("tokenizer"):
                    tokenizer_type = tokenizer_type[:-9]  # "tokenizer" son ekini kaldır
                elif class_name.endswith("tokenizer"):
                    class_name = class_name[:-9]  # "tokenizer" son ekini kaldır
                
                # Önce tip ile dene, sonra sınıf adı ile
                if tokenizer_type in TOKENIZER_TYPES:
                    tokenizers[tokenizer_name] = TOKENIZER_TYPES[tokenizer_type].load(tokenizer_path)
                elif class_name in TOKENIZER_TYPES:
                    tokenizers[tokenizer_name] = TOKENIZER_TYPES[class_name].load(tokenizer_path)
                else:
                    # Genel yükleyici ile dene
                    tokenizers[tokenizer_name] = load_tokenizer_from_path(tokenizer_path)
                    
            except Exception as e:
                logger.warning(f"'{tokenizer_name}' tokenizer'ı yüklenirken hata: {e}")
                
        if not tokenizers:
            raise ValueError("Hiç alt tokenizer yüklenemedi")
            
        # Varsayılan tokenizer'ı kontrol et
        default_tokenizer = config.get("default_tokenizer")
        if default_tokenizer not in tokenizers:
            # İlk yüklenen tokenizer'ı varsayılan olarak kullan
            default_tokenizer = next(iter(tokenizers.keys()))
            
        # Birleştirilmiş sözlük yapılandırmasını yükle
        merged_vocab = config.get("merged_vocab", False)
        merged_vocab_config = {}
        
        if merged_vocab:
            merged_vocab_path = path / "merged_vocab.json"
            
            if merged_vocab_path.exists():
                with open(merged_vocab_path, "r", encoding="utf-8") as f:
                    merged_vocab_config = json.load(f)
                    
        # Karma tokenizer'ı oluştur
        mixed_tokenizer = cls(
            tokenizers=tokenizers,
            default_tokenizer=default_tokenizer,
            merged_vocab=merged_vocab,
            special_tokens=special_tokens
        )
        
        # Birleştirilmiş sözlük eşlemelerini yükle
        if merged_vocab and merged_vocab_config:
            mixed_tokenizer.id_ranges = merged_vocab_config.get("id_ranges", {})
            
            # Eşleme sözlüklerini yükle
            if "merged_id_map" in merged_vocab_config:
                mixed_id_map = {}
                for key, value in merged_vocab_config["merged_id_map"].items():
                    t, i = key.split(":", 1)
                    try:
                        i = int(i)
                    except ValueError:
                        pass  # String olarak sakla
                    mixed_id_map[(t, i)] = value
                mixed_tokenizer.merged_id_map = mixed_id_map
                
            if "merged_reverse_map" in merged_vocab_config:
                mixed_reverse_map = {}
                for key, value in merged_vocab_config["merged_reverse_map"].items():
                    mixed_reverse_map[int(key)] = tuple(value)
                mixed_tokenizer.merged_reverse_map = mixed_reverse_map
                
        # Meta verileri yükle
        if "metadata" in config:
            mixed_tokenizer.metadata.update(config["metadata"])
            
        # Eğitilmiş olarak işaretle
        mixed_tokenizer.is_trained = True
        
        logger.info(f"Karma tokenizer {path} konumundan yüklendi")
        return mixed_tokenizer