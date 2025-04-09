# aiquantr_tokenizer/core/pipeline_manager.py
"""
İşlem hattı yönetimi.

Bu modül, tokenizer hazırlama sürecindeki
farklı aşamaları yöneten işlem hattı sistemini içerir.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union, Callable

from ..config.config_manager import ConfigManager
from .data_collector import DataCollector
from .registry import Registry
from ..tokenizers.base import BaseTokenizer
from ..tokenizers.factory import create_tokenizer_from_config

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Tokenizer hazırlama işlem hatlarını yöneten sınıf.
    
    Bu sınıf, veri toplama, işleme, tokenizer eğitimi ve
    değerlendirme süreçlerini koordine eder.
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None,
        data_collector: Optional[DataCollector] = None,
        tokenizer: Optional[BaseTokenizer] = None
    ):
        """
        PipelineManager sınıfı başlatıcısı.
        
        Args:
            config: İşlem hattı yapılandırması
            data_collector: Veri toplama bileşeni
            tokenizer: Tokenizer nesnesi
        """
        # Yapılandırma yönetimi
        if config is None:
            self.config = {}
            self.config_manager = None
        elif isinstance(config, ConfigManager):
            self.config_manager = config
            self.config = config.to_dict()
        else:
            self.config = config
            self.config_manager = None
            
        # Veri toplama bileşeni
        self.data_collector = data_collector
        
        # Tokenizer nesnesi
        self.tokenizer = tokenizer
        
        # İşlemci ve metriklerin kaydı
        self.registry = Registry()
        
        # İşlem hattı aşamaları
        self.pipeline_steps = []
        
        # İstatistikler
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_duration": None,
            "steps": {}
        }
        
    def add_step(
        self,
        step_name: str,
        step_func: Callable,
        enabled: bool = True,
        description: str = ""
    ):
        """
        İşlem hattına bir adım ekler.
        
        Args:
            step_name: Adım adı
            step_func: Adım fonksiyonu
            enabled: Adım etkin mi? (varsayılan: True)
            description: Adım açıklaması (varsayılan: "")
        """
        self.pipeline_steps.append({
            "name": step_name,
            "func": step_func,
            "enabled": enabled,
            "description": description
        })
        logger.debug(f"İşlem hattına adım eklendi: {step_name}")
        
    def setup_default_pipeline(self):
        """
        Varsayılan işlem hattını yapılandırır.
        """
        # 1. Veri toplama adımı
        self.add_step(
            "collect_data",
            self._step_collect_data,
            enabled=True,
            description="Veri kaynaklarından metin toplama"
        )
        
        # 2. İşlemcileri uygulama adımı
        self.add_step(
            "apply_processors",
            self._step_apply_processors,
            enabled=True,
            description="Metin işlemcilerini uygulama"
        )
        
        # 3. Tokenizer oluşturma adımı
        self.add_step(
            "create_tokenizer",
            self._step_create_tokenizer,
            enabled=True,
            description="Tokenizer'ı yapılandırma"
        )
        
        # 4. Tokenizer eğitimi adımı
        self.add_step(
            "train_tokenizer",
            self._step_train_tokenizer,
            enabled=True,
            description="Tokenizer'ı eğitme"
        )
        
        # 5. Değerlendirme adımı
        self.add_step(
            "evaluate_tokenizer",
            self._step_evaluate_tokenizer,
            enabled=True,
            description="Tokenizer'ı değerlendirme"
        )
        
        # 6. Dışa aktarma adımı
        self.add_step(
            "export_tokenizer",
            self._step_export_tokenizer,
            enabled=True,
            description="Tokenizer'ı dışa aktarma"
        )
        
    def run(self) -> Dict[str, Any]:
        """
        İşlem hattını çalıştırır.
        
        Returns:
            Dict[str, Any]: İşlem hattı sonuçları
        """
        # Başlangıç zamanını kaydet
        self.stats["start_time"] = time.time()
        logger.info("İşlem hattı başlatılıyor...")
        
        result = {
            "success": True,
            "steps": {},
            "artifacts": {}
        }
        
        # Tüm adımları çalıştır
        for step in self.pipeline_steps:
            if not step["enabled"]:
                logger.info(f"Adım atlanıyor: {step['name']} (devre dışı)")
                continue
                
            step_name = step["name"]
            logger.info(f"Adım başlatılıyor: {step_name}")
            
            step_start = time.time()
            
            try:
                step_result = step["func"]()
                
                # Adım istatistiklerini kaydet
                step_duration = time.time() - step_start
                self.stats["steps"][step_name] = {
                    "duration": step_duration,
                    "success": True
                }
                
                # Adım sonucunu kaydet
                result["steps"][step_name] = {
                    "success": True,
                    "duration": step_duration
                }
                
                if step_result:
                    result["artifacts"][step_name] = step_result
                    
                logger.info(f"Adım tamamlandı: {step_name} ({step_duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"Adım başarısız: {step_name} - {str(e)}")
                
                # Hata istatistiklerini kaydet
                self.stats["steps"][step_name] = {
                    "duration": time.time() - step_start,
                    "success": False,
                    "error": str(e)
                }
                
                # Hata sonucunu kaydet
                result["steps"][step_name] = {
                    "success": False,
                    "error": str(e)
                }
                
                result["success"] = False
                
                # Kritik adım başarısızsa işlemi durdur
                if step_name in ["collect_data", "create_tokenizer", "train_tokenizer"]:
                    logger.error("Kritik adım başarısız, işlem hattı durduruluyor")
                    break
        
        # Bitiş zamanını ve toplam süreyi kaydet
        self.stats["end_time"] = time.time()
        self.stats["total_duration"] = self.stats["end_time"] - self.stats["start_time"]
        
        result["total_duration"] = self.stats["total_duration"]
        result["tokenizer"] = self.tokenizer
        
        logger.info(f"İşlem hattı tamamlandı ({self.stats['total_duration']:.2f}s)")
        return result
    
    def _step_collect_data(self) -> Dict[str, Any]:
        """
        Veri toplama adımını gerçekleştirir.
        
        Returns:
            Dict[str, Any]: Toplanan veriler ve istatistikler
        """
        # Veri toplayıcı kontrol et
        if self.data_collector is None:
            # Yapılandırmadan veri toplayıcı oluştur
            from ..data.sources.local_source import LocalDataSource
            from ..data.sources.huggingface_source import HuggingFaceDataSource
            
            data_config = self.config.get("data", {})
            
            # Veri toplayıcı oluştur
            self.data_collector = DataCollector(data_config)
            
            # Yapılandırmadan veri kaynaklarını ekle
            sources_config = data_config.get("sources", [])
            
            for source_config in sources_config:
                source_type = source_config.get("type")
                
                if source_type == "local":
                    source = LocalDataSource(
                        path=source_config.get("path"),
                        name=source_config.get("name", "local"),
                        config=source_config
                    )
                    self.data_collector.add_source(source)
                    
                elif source_type == "huggingface":
                    source = HuggingFaceDataSource(
                        dataset_name=source_config.get("dataset"),
                        name=source_config.get("name", "huggingface"),
                        config=source_config
                    )
                    self.data_collector.add_source(source)
                    
        # Veri topla
        limit = self.config.get("data", {}).get("limit")
        texts = self.data_collector.collect_data(limit=limit)
        
        # Veri bölme
        if self.config.get("data", {}).get("split_validation", False):
            from ..utils.io_utils import split_data
            
            train_ratio = self.config.get("data", {}).get("train_ratio", 0.9)
            val_ratio = 1.0 - train_ratio
            
            train_texts, val_texts = split_data(
                texts=texts,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                shuffle=True
            )
            
            result = {
                "train_texts": train_texts,
                "val_texts": val_texts,
                "stats": self.data_collector.get_stats()
            }
            
            logger.info(f"Veri bölündü: {len(train_texts)} eğitim, {len(val_texts)} doğrulama")
        else:
            result = {
                "texts": texts,
                "stats": self.data_collector.get_stats()
            }
            
        return result
    
    def _step_apply_processors(self) -> Dict[str, Any]:
        """
        Metin işlemcilerini uygulama adımını gerçekleştirir.
        
        Returns:
            Dict[str, Any]: İşlenmiş veriler ve istatistikler
        """
        # Önceki adım verilerini kontrol et
        if "artifacts" not in vars(self) or "collect_data" not in self.artifacts:
            logger.warning("Veri toplama adımı sonuçları bulunamadı, işlemciler atlanıyor")
            return {}
            
        # Metin verileri al
        if "texts" in self.artifacts["collect_data"]:
            texts = self.artifacts["collect_data"]["texts"]
            has_validation = False
        else:
            texts = self.artifacts["collect_data"]["train_texts"]
            val_texts = self.artifacts["collect_data"]["val_texts"]
            has_validation = True
            
        # İşlemcileri yapılandır ve uygula
        processors_config = self.config.get("processors", {})
        
        # İşlemci kayıtlarını al
        language_processors = self.registry.get_components("processor", "language")
        code_processors = self.registry.get_components("processor", "code")
        
        # İstatistikler
        stats = {"applied_processors": []}
        
        # Dil işlemcileri
        if processors_config.get("language", {}).get("enabled", False):
            lang_config = processors_config.get("language", {})
            
            # Aktif dil işlemcilerini bul
            for lang, processor_cls in language_processors.items():
                if lang_config.get(lang, {}).get("enabled", False):
                    processor = processor_cls(lang_config.get(lang, {}))
                    
                    # İşlemciyi uygula
                    logger.info(f"Dil işlemcisi uygulanıyor: {lang}")
                    texts = processor.process_batch(texts)
                    
                    if has_validation:
                        val_texts = processor.process_batch(val_texts)
                        
                    stats["applied_processors"].append(f"language.{lang}")
        
        # Kod işlemcileri
        if processors_config.get("code", {}).get("enabled", False):
            code_config = processors_config.get("code", {})
            
            # Aktif kod işlemcilerini bul
            for lang, processor_cls in code_processors.items():
                if code_config.get(lang, {}).get("enabled", False):
                    processor = processor_cls(code_config.get(lang, {}))
                    
                    # İşlemciyi uygula
                    logger.info(f"Kod işlemcisi uygulanıyor: {lang}")
                    texts = processor.process_batch(texts)
                    
                    if has_validation:
                        val_texts = processor.process_batch(val_texts)
                        
                    stats["applied_processors"].append(f"code.{lang}")
        
        # Sonuçları döndür
        if has_validation:
            result = {
                "train_texts": texts,
                "val_texts": val_texts,
                "stats": stats
            }
        else:
            result = {
                "texts": texts,
                "stats": stats
            }
            
        return result
    
    def _step_create_tokenizer(self) -> Dict[str, Any]:
        """
        Tokenizer oluşturma adımını gerçekleştirir.
        
        Returns:
            Dict[str, Any]: Oluşturulan tokenizer ve yapılandırma
        """
        # Tokenizer zaten varsa atla
        if self.tokenizer is not None:
            logger.info("Tokenizer zaten var, oluşturma adımı atlanıyor")
            return {"tokenizer": self.tokenizer}
            
        # Tokenizer yapılandırmasını al
        if self.config_manager:
            tokenizer_config = self.config_manager.get_tokenizer_config()
        else:
            tokenizer_config = self.config.get("tokenizer", {})
            
        # Yapılandırmadan tokenizer oluştur
        logger.info(f"Tokenizer oluşturuluyor: {tokenizer_config.get('type', 'bpe')}")
        self.tokenizer = create_tokenizer_from_config(tokenizer_config)
        
        return {
            "tokenizer": self.tokenizer,
            "config": tokenizer_config
        }
    
    def _step_train_tokenizer(self) -> Dict[str, Any]:
        """
        Tokenizer eğitimi adımını gerçekleştirir.
        
        Returns:
            Dict[str, Any]: Eğitim sonuçları ve istatistikler
        """
        # Tokenizer'ı kontrol et
        if self.tokenizer is None:
            logger.error("Eğitim için tokenizer bulunamadı")
            raise ValueError("Tokenizer oluşturulmamış")
            
        # Eğitim verilerini al
        if "artifacts" not in vars(self) or not (
            "collect_data" in self.artifacts or "apply_processors" in self.artifacts
        ):
            logger.error("Eğitim için veri bulunamadı")
            raise ValueError("Veri toplama veya işleme adımları tamamlanmamış")
            
        # En son veri adımını belirle
        data_step = "apply_processors" if "apply_processors" in self.artifacts else "collect_data"
        
        # Metinleri al
        if "texts" in self.artifacts[data_step]:
            texts = self.artifacts[data_step]["texts"]
            has_validation = False
        else:
            texts = self.artifacts[data_step]["train_texts"]
            val_texts = self.artifacts[data_step]["val_texts"]
            has_validation = True
            
        # Eğitim yapılandırmasını al
        training_config = self.config.get("training", {})
        
        # TokenizerTrainer oluştur
        from .tokenizer_trainer import TokenizerTrainer
        
        trainer = TokenizerTrainer(
            batch_size=training_config.get("batch_size", 1000),
            num_iterations=training_config.get("num_iterations"),
            show_progress=training_config.get("show_progress", True),
            validation_split=training_config.get("validation_split", 0.1),
            seed=training_config.get("seed", 42)
        )
        
        # Tokenizer'ı eğit
        logger.info("Tokenizer eğitimi başlıyor...")
        training_results = self.tokenizer.train(texts, trainer=trainer)
        
        # Doğrulama yap
        if has_validation:
            logger.info("Tokenizer doğrulaması yapılıyor...")
            validation_results = trainer.validate(self.tokenizer, val_texts)
        else:
            # Eğitim verilerinin bir kısmını doğrulama için kullan
            val_size = min(1000, len(texts) // 10)
            validation_results = trainer.validate(self.tokenizer, texts[:val_size])
            
        # Sonuçları döndür
        return {
            "training_results": training_results,
            "validation_results": validation_results,
            "stats": trainer.get_stats()
        }
    
    def _step_evaluate_tokenizer(self) -> Dict[str, Any]:
        """
        Tokenizer değerlendirme adımını gerçekleştirir.
        
        Returns:
            Dict[str, Any]: Değerlendirme sonuçları
        """
        # Tokenizer'ı kontrol et
        if self.tokenizer is None:
            logger.error("Değerlendirme için tokenizer bulunamadı")
            raise ValueError("Tokenizer oluşturulmamış veya eğitilmemiş")
            
        # Değerlendirme yapılandırmasını al
        eval_config = self.config.get("evaluation", {})
        
        # Değerlendirme verilerini al
        if "artifacts" not in vars(self) or not (
            "collect_data" in self.artifacts or "apply_processors" in self.artifacts
        ):
            logger.warning("Değerlendirme için veri bulunamadı, dışarıdan veri yükleniyor")
            
            # Dış değerlendirme verisi yükle
            if "external_data_path" in eval_config:
                from ..utils.io_utils import load_data
                texts = load_data(eval_config["external_data_path"])
            else:
                logger.error("Değerlendirme için veri bulunamadı")
                raise ValueError("Değerlendirme verisi bulunamadı")
        else:
            # En son veri adımını belirle
            data_step = "apply_processors" if "apply_processors" in self.artifacts else "collect_data"
            
            # Metinleri al - öncelikle doğrulama verisi kullan
            if "val_texts" in self.artifacts[data_step]:
                texts = self.artifacts[data_step]["val_texts"]
            elif "texts" in self.artifacts[data_step]:
                # Eğitim verilerinin bir kısmını değerlendirme için kullan
                texts = self.artifacts[data_step]["texts"][-1000:]
            else:
                logger.error("Değerlendirme için veri bulunamadı")
                raise ValueError("Değerlendirme verisi bulunamadı")
                
        # Değerlendirilecek metrikleri belirle
        metrics = eval_config.get("metrics", [
            "vocabulary_coverage",
            "compression_ratio",
            "token_frequency",
            "encoding_speed",
            "reconstruction_accuracy"
        ])
        
        # Tokenizer'ı değerlendir
        from ..tokenizers.evaluation import evaluate_tokenizer
        
        logger.info(f"Tokenizer değerlendiriliyor: {len(metrics)} metrik, {len(texts)} metin")
        evaluation_results = evaluate_tokenizer(self.tokenizer, texts, metrics)
        
        return {
            "evaluation_results": evaluation_results,
            "metrics": metrics
        }
    
    def _step_export_tokenizer(self) -> Dict[str, Any]:
        """
        Tokenizer'ı dışa aktarma adımını gerçekleştirir.
        
        Returns:
            Dict[str, Any]: Dışa aktarma sonuçları
        """
        # Tokenizer'ı kontrol et
        if self.tokenizer is None:
            logger.error("Dışa aktarma için tokenizer bulunamadı")
            raise ValueError("Tokenizer oluşturulmamış veya eğitilmemiş")
            
        # Dışa aktarma yapılandırmasını al
        export_config = self.config.get("export", {})
        export_path = export_config.get("path", "./output/tokenizer")
        export_format = export_config.get("format", "json")
        
        # Tokenizer meta verilerini güncelle
        self.tokenizer.metadata["export_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # İstatistikleri ekle
        if export_config.get("include_statistics", True) and "train_tokenizer" in self.artifacts:
            training_stats = self.artifacts["train_tokenizer"]["stats"]
            self.tokenizer.metadata["training_stats"] = training_stats
            
        # Tokenizer'ı kaydet
        logger.info(f"Tokenizer dışa aktarılıyor: {export_path}")
        self.tokenizer.save_pretrained(export_path)
        
        # Diğer formatları dışa aktar
        exported_formats = ["json"]
        
        # HuggingFace formatına dönüştür
        if export_format == "huggingface" or export_config.get("export_huggingface", False):
            try:
                from ..utils.hub_utils import export_to_huggingface
                hf_path = f"{export_path}_hf"
                export_to_huggingface(self.tokenizer, hf_path)
                exported_formats.append("huggingface")
                logger.info(f"Tokenizer HuggingFace formatında dışa aktarıldı: {hf_path}")
            except Exception as e:
                logger.warning(f"HuggingFace formatına dönüştürme başarısız: {e}")
                
        # SentencePiece formatına dönüştür
        if export_format == "sentencepiece" or export_config.get("export_sentencepiece", False):
            try:
                sp_path = f"{export_path}_sp"
                
                # SentencePiece tokenizer ise doğrudan kaydet
                if hasattr(self.tokenizer, "save_sentencepiece"):
                    self.tokenizer.save_sentencepiece(sp_path)
                # Değilse dönüştür
                else:
                    from ..tokenizers.sentencepiece import convert_to_sentencepiece
                    convert_to_sentencepiece(self.tokenizer, sp_path)
                    
                exported_formats.append("sentencepiece")
                logger.info(f"Tokenizer SentencePiece formatında dışa aktarıldı: {sp_path}")
            except Exception as e:
                logger.warning(f"SentencePiece formatına dönüştürme başarısız: {e}")
        
        # Sözlük dışa aktar
        if export_config.get("save_vocab", True):
            try:
                from ..utils.io_utils import save_json
                vocab = self.tokenizer.get_vocab()
                vocab_path = f"{export_path}_vocab.json"
                save_json(vocab, vocab_path)
                logger.info(f"Sözlük dışa aktarıldı: {vocab_path}")
            except Exception as e:
                logger.warning(f"Sözlük dışa aktarma başarısız: {e}")
                
        return {
            "export_path": export_path,
            "exported_formats": exported_formats
        }