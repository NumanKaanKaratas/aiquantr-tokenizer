# aiquantr_tokenizer/config/config_manager.py
"""
Konfigürasyon yönetimi sınıfı ve işlevleri.

Bu modül, tokenizer eğitimi için konfigürasyon dosyalarını yükleme,
doğrulama ve yönetme işlevlerini sağlar.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.yaml")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def get_template_path(template_name: str) -> str:
    """
    Belirtilen şablon adına göre tam dosya yolunu döndürür.
    
    Args:
        template_name: Şablon dosyasının adı (örn. "python_tr.yaml")
        
    Returns:
        str: Şablon dosyasının tam yolu
        
    Raises:
        FileNotFoundError: Belirtilen şablon dosyası bulunamazsa
    """
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    if not os.path.exists(template_path):
        available_templates = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.yaml')]
        raise FileNotFoundError(
            f"'{template_name}' şablonu bulunamadı. Mevcut şablonlar: {', '.join(available_templates)}"
        )
    return template_path


def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    İki konfigürasyon sözlüğünü birleştirir. Override sözlüğü, base sözlüğünü geçersiz kılar.
    
    Args:
        base_config: Temel konfigürasyon
        override_config: Geçersiz kılacak konfigürasyon
        
    Returns:
        Dict[str, Any]: Birleştirilmiş konfigürasyon
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        # Eğer her iki sözlükte de aynı anahtar varsa ve her ikisi de sözlük ise, iç içe birleştir
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            # Aksi takdirde, değeri doğrudan değiştir
            result[key] = value
            
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Konfigürasyonun geçerli olup olmadığını kontrol eder.
    
    Args:
        config: Kontrol edilecek konfigürasyon sözlüğü
        
    Returns:
        bool: Geçerli ise True, değilse False
    
    Note:
        Geçersiz bir yapılandırma bulunursa hata mesajları loglanır
    """
    # Temel konfigürasyon kontrolü
    if not isinstance(config, dict):
        logger.error("Konfigürasyon bir sözlük olmalıdır")
        return False
    
    # Gerekli alanları kontrol et - gerçek ihtiyaçlarınıza göre güncelleyin
    required_sections = ["tokenizer", "data"]
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Konfigürasyonda gerekli '{section}' bölümü eksik")
            return False
    
    # Tokenizer yapılandırmasını kontrol et
    if not isinstance(config.get('tokenizer'), dict):
        logger.error("'tokenizer' bir sözlük olmalıdır")
        return False
    
    # Veri yapılandırmasını kontrol et
    if not isinstance(config.get('data'), dict):
        logger.error("'data' bir sözlük olmalıdır")
        return False
    
    return True


class ConfigManager:
    """
    Konfigürasyon yönetimi sınıfı.
    
    Bu sınıf, tokenizer eğitimi için konfigürasyon dosyalarını yükleme,
    doğrulama ve yönetme işlevlerini sağlar.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None, 
        template_name: Optional[str] = None
    ):
        """
        ConfigManager sınıfı başlatıcısı.
        
        Args:
            config_path: Konfigürasyon dosyasının yolu (varsayılan: None)
            template_name: Kullanılacak şablon adı (varsayılan: None)
            
        Raises:
            FileNotFoundError: Belirtilen konfigürasyon dosyası bulunamazsa
            yaml.YAMLError: YAML dosyası doğru formatlanmamışsa
        """
        self.config_path = config_path
        self.config = {}
        
        # Yapılandırmayı yükle
        self.load_config(config_path, template_name)
    
    def load_config(self, config_path=None, template_name=None):
        """
        Konfigürasyon dosyasını yükler ve birleştirir.
        
        Yükleme sırası:
        1. Varsayılan konfigürasyon
        2. Şablon konfigürasyonu (eğer belirtildiyse)
        3. Kullanıcı konfigürasyonu (eğer belirtildiyse)
        
        Her aşamada yüklenen konfigürasyon önceki konfigürasyonlarla birleştirilir.
        
        Args:
            config_path (str, optional): Kullanıcı konfigürasyon dosyasının yolu.
            template_name (str, optional): Kullanılacak şablon adı.
        
        Returns:
            dict: Birleştirilmiş ve doğrulanmış konfigürasyon sözlüğü.
        
        Raises:
            yaml.YAMLError: YAML dosyası geçersiz formatlanmışsa.
        """
        from pathlib import Path
        
        # Mevcut konfigürasyon yolunu kullan veya parametre olarak gelenini al
        config_path = config_path or self.config_path
        
        # Başlangıç konfigürasyonu boş
        self.config = {}
        
        # Konfigürasyon dosyası yükleme işlemlerini izlemek için
        loaded_files = []
        
        # YAML dosyasını farklı kodlamalarla yüklemeyi deneyen yardımcı fonksiyon
        def _load_yaml_with_fallback_encodings(file_path):
            """
            YAML dosyasını çeşitli karakter kodlamaları deneyerek yüklemeyi dener.
            
            Args:
                file_path (str): Yüklenecek YAML dosyasının yolu.
                
            Returns:
                dict: Yüklenen YAML içeriği veya boş sözlük.
                
            Raises:
                yaml.YAMLError: YAML formatı geçersizse.
            """
            # Desteklenen kodlamalar (öncelik sırasına göre)
            encodings = ['utf-8', 'cp1254', 'latin-1', 'windows-1252']
            last_error = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = yaml.safe_load(f) or {}
                        logger.debug(f"Dosya başarıyla yüklendi ({encoding} kodlaması): {file_path}")
                        return content
                except UnicodeDecodeError as e:
                    last_error = e
                    continue
                except yaml.YAMLError as e:
                    # YAML formatı hatalı - bu ciddi bir hata, yeniden yükleme denemesi olmaz
                    logger.error(f"YAML formatı hatası ({encoding} kodlaması): {file_path}")
                    raise yaml.YAMLError(f"YAML dosyası geçersiz format içeriyor: {e}")
            
            # Hiçbir kodlama ile açılamazsa
            logger.error(f"Dosya hiçbir kodlama ile okunamadı: {file_path}")
            logger.debug(f"Son hata: {last_error}")
            return {}
        
        try:
            # 1. Varsayılan konfigürasyonu yükle
            default_config_path = Path(DEFAULT_CONFIG_PATH)
            if default_config_path.is_file():
                try:
                    default_config = _load_yaml_with_fallback_encodings(default_config_path)
                    self.config = default_config
                    loaded_files.append(str(default_config_path))
                    logger.debug(f"Varsayılan konfigürasyon yüklendi: {default_config_path}")
                except Exception as e:
                    logger.error(f"Varsayılan konfigürasyon yüklenirken hata: {e}")
            else:
                logger.warning(f"Varsayılan konfigürasyon dosyası bulunamadı: {default_config_path}")
            
            # 2. Şablon konfigürasyonunu yükle ve birleştir (belirtildiyse)
            if template_name:
                try:
                    template_path = get_template_path(template_name)
                    template_file = Path(template_path)
                    if template_file.is_file():
                        template_config = _load_yaml_with_fallback_encodings(template_file)
                        self.config = _merge_configs(self.config, template_config)
                        loaded_files.append(str(template_file))
                        logger.debug(f"Şablon konfigürasyonu birleştirildi: {template_file}")
                    else:
                        logger.warning(f"Belirtilen şablon dosyası bulunamadı: {template_file}")
                except FileNotFoundError:
                    logger.error(f"Şablon dosyası bulunamadı: {template_name}")
                except Exception as e:
                    logger.error(f"Şablon konfigürasyonu yüklenirken hata: {e}")
            
            # 3. Kullanıcı konfigürasyonunu yükle ve birleştir (belirtildiyse)
            if config_path:
                user_config_file = Path(config_path)
                if user_config_file.is_file():
                    try:
                        user_config = _load_yaml_with_fallback_encodings(user_config_file)
                        self.config = _merge_configs(self.config, user_config)
                        loaded_files.append(str(user_config_file))
                        logger.debug(f"Kullanıcı konfigürasyonu birleştirildi: {user_config_file}")
                    except Exception as e:
                        logger.error(f"Kullanıcı konfigürasyonu yüklenirken hata: {e}")
                else:
                    logger.warning(f"Belirtilen konfigürasyon dosyası bulunamadı: {user_config_file}")
            
            # Yüklenen dosyaları özetle
            if loaded_files:
                logger.info(f"Konfigürasyon yükleme tamamlandı. Birleştirilen dosyalar: {', '.join(loaded_files)}")
            else:
                logger.warning("Hiçbir konfigürasyon dosyası yüklenemedi. Varsayılan değerler kullanılacak.")
            
            # 4. Konfigürasyonu doğrula ve gerekirse tamamla
            if not validate_config(self.config):
                logger.warning("Geçersiz konfigürasyon. Temel işlevsellik için varsayılan değerler uygulanacak.")
                self._ensure_minimal_config()
            
            return self.config
        
        except Exception as e:
            logger.exception(f"Konfigürasyon yükleme işlemi sırasında beklenmeyen hata: {e}")
            # Temel işlevselliği sağlamak için minimal konfigürasyon oluştur
            self._ensure_minimal_config()
            return self.config
    
    def _ensure_minimal_config(self):
        """
        Geçersiz konfigürasyon durumunda minimum çalışabilir bir konfigürasyon oluşturur.
        Kullanıcının eksik girdileri için varsayılan değerlerle temel işlevselliği sağlar.
        """
        # Tokenizer bölümü yoksa veya geçersizse
        if not isinstance(self.config.get('tokenizer'), dict):
            self.config['tokenizer'] = {
                "name": "default_tokenizer",
                "vocab_size": 30000,
                "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
                "min_frequency": 5
            }
            logger.info("Tokenizer yapılandırması bulunamadı veya geçersiz. Varsayılan değerler kullanılıyor.")
        else:
            # Tokenizer alt alanlarını kontrol et
            if "name" not in self.config['tokenizer']:
                self.config['tokenizer']["name"] = "default_tokenizer"
                logger.info("Tokenizer ismi bulunamadı. Varsayılan 'default_tokenizer' kullanılıyor.")
            
            if "vocab_size" not in self.config['tokenizer']:
                self.config['tokenizer']["vocab_size"] = 30000
                logger.info("Tokenizer sözcük dağarcığı boyutu bulunamadı. Varsayılan 30000 kullanılıyor.")
            
            if "special_tokens" not in self.config['tokenizer']:
                self.config['tokenizer']["special_tokens"] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
                logger.info("Özel tokenler bulunamadı. Varsayılan özel tokenler kullanılıyor.")
        
        # Data bölümü yoksa veya geçersizse
        if not isinstance(self.config.get('data'), dict):
            self.config['data'] = {
                "sources": [],
                "processors": [
                    {
                        "name": "text_cleaner",
                        "type": "text",
                        "remove_urls": True,
                        "lowercase": False
                    }
                ],
                "max_length": 512,
                "min_length": 10
            }
            logger.info("Veri yapılandırması bulunamadı veya geçersiz. Varsayılan değerler kullanılıyor.")
        else:
            # Data alt alanlarını kontrol et
            if "sources" not in self.config['data']:
                self.config['data']["sources"] = []
                logger.info("Veri kaynakları bulunamadı. Boş liste kullanılıyor.")
            
            if "processors" not in self.config['data']:
                self.config['data']["processors"] = [
                    {
                        "name": "text_cleaner",
                        "type": "text",
                        "remove_urls": True,
                        "lowercase": False
                    }
                ]
                logger.info("Veri işleyicileri bulunamadı. Varsayılan metin temizleyici kullanılıyor.")
        
        # Training bölümü yoksa veya geçersizse
        if not isinstance(self.config.get('training'), dict):
            self.config['training'] = {
                "max_samples": 1000000,
                "batch_size": 1000,
                "epochs": 2,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "validation_split": 0.1
            }
            logger.info("Eğitim yapılandırması bulunamadı veya geçersiz. Varsayılan değerler kullanılıyor.")
        else:
            # Training alt alanlarını kontrol et
            if "max_samples" not in self.config['training']:
                self.config['training']["max_samples"] = 1000000
                logger.info("Maksimum örnek sayısı bulunamadı. Varsayılan 1000000 kullanılıyor.")
            
            if "batch_size" not in self.config['training']:
                self.config['training']["batch_size"] = 1000
                logger.info("Batch boyutu bulunamadı. Varsayılan 1000 kullanılıyor.")
            
            if "epochs" not in self.config['training']:
                self.config['training']["epochs"] = 2
                logger.info("Epoch sayısı bulunamadı. Varsayılan 2 kullanılıyor.")
            
            if "learning_rate" not in self.config['training']:
                self.config['training']["learning_rate"] = 0.0001
                logger.info("Öğrenme oranı bulunamadı. Varsayılan 0.0001 kullanılıyor.")
        
        # Output bölümü yoksa veya geçersizse
        if not isinstance(self.config.get('output'), dict):
            self.config['output'] = {
                "path": "./output",
                "save_intermediates": False,
                "save_every_epoch": False,
                "metrics_file": "metrics.json"
            }
            logger.info("Çıktı yapılandırması bulunamadı veya geçersiz. Varsayılan değerler kullanılıyor.")
        else:
            # Output alt alanlarını kontrol et
            if "path" not in self.config['output']:
                self.config['output']["path"] = "./output"
                logger.info("Çıktı yolu bulunamadı. Varsayılan './output' kullanılıyor.")
        
        # Metrics bölümü yoksa veya geçersizse
        if not isinstance(self.config.get('metrics'), dict):
            self.config['metrics'] = {
                "log_every": 100,
                "save_history": True
            }
            logger.info("Metrik yapılandırması bulunamadı veya geçersiz. Varsayılan değerler kullanılıyor.")
        
    def update_config(self, update_dict):
        """
        Konfigürasyonu günceller.
        
        Args:
            update_dict (dict): Güncellenecek konfigürasyon sözlüğü
        
        Returns:
            dict: Güncellenmiş konfigürasyon
        """
        self.config = _merge_configs(self.config, update_dict)
        return self.config

    def save_config(self, output_path=None):
        """
        Konfigürasyonu dosyaya kaydeder.
        
        Args:
            output_path (str, optional): Çıktı dosya yolu
                
        Returns:
            None
        """
        save_path = output_path or self.config_path
        
        if not save_path:
            logger.warning("Kayıt yolu belirtilmedi, konfigürasyon kaydedilmiyor")
            return
            
        try:
            # Dizin yoksa oluştur
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                
            logger.info(f"Konfigürasyon kaydedildi: {save_path}")
        except (PermissionError, IOError) as e:
            logger.error(f"Konfigürasyon kaydedilirken hata: {e}")
            raise
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Konfigürasyon değerini getirir. "a.b.c" formatındaki iç içe anahtarları destekler.
        
        Args:
            key: Alınacak konfigürasyon anahtarı (örn. "tokenizer.vocab_size")
            default: Anahtar bulunamazsa dönülecek varsayılan değer
                
        Returns:
            Any: Konfigürasyon değeri veya varsayılan değer
        """
        if '.' in key:
            # İç içe anahtarlar için özel işleme (örn. "tokenizer.vocab_size")
            parts = key.split('.')
            current = self.config
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
                    
            return current
        else:
            # Basit anahtar
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Konfigürasyon değerini ayarlar. "a.b.c" formatındaki iç içe anahtarları destekler.
        
        Args:
            key: Ayarlanacak konfigürasyon anahtarı (örn. "tokenizer.vocab_size")
            value: Yeni değer
        """
        if '.' in key:
            # İç içe anahtarlar için özel işleme (örn. "tokenizer.vocab_size")
            parts = key.split('.')
            current = self.config
            
            # Son kısımdan önceki tüm kısımları işle
            for part in parts[:-1]:
                # Eğer bu kısım yoksa, yeni bir sözlük oluştur
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            
            # Son kısmı ayarla
            current[parts[-1]] = value
        else:
            # Basit anahtar
            self.config[key] = value
    
    def get_data_sources(self) -> List[Dict[str, Any]]:
        """
        Veri kaynakları konfigürasyonunu getirir.
        
        Returns:
            List[Dict[str, Any]]: Veri kaynakları listesi
        """
        return self.config.get('data', {}).get('sources', [])
    
    def get_processors(self) -> List[Dict[str, Any]]:
        """
        İşlemci konfigürasyonlarını getirir.
        
        Returns:
            List[Dict[str, Any]]: İşlemci konfigürasyonları listesi
        """
        return self.config.get('data', {}).get('processors', [])
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Çıktı konfigürasyonunu getirir.
        
        Returns:
            Dict[str, Any]: Çıktı konfigürasyonu
        """
        return self.config.get('output', {})
    
    def get_metric_config(self) -> Dict[str, Any]:
        """
        Metrik konfigürasyonunu getirir.
        
        Returns:
            Dict[str, Any]: Metrik konfigürasyonu
        """
        return self.config.get('metrics', {})
    
    @staticmethod
    def list_available_templates() -> List[str]:
        """
        Kullanılabilir konfigürasyon şablonlarını listeler.
        
        Returns:
            List[str]: Kullanılabilir şablon dosyaları listesi
        """
        try:
            return [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.yaml')]
        except FileNotFoundError:
            logger.warning(f"Şablon dizini bulunamadı: {TEMPLATE_DIR}")
            return []