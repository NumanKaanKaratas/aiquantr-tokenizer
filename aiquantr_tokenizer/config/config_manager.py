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
        Konfigürasyon dosyasını yükler. Önce varsayılan konfigürasyonu, sonra belirtilen şablonu,
        en son olarak da kullanıcı konfigürasyonunu yükleyerek bunları birleştirir.
        
        Args:
            config_path (str, optional): Konfigürasyon dosyasının yolu
            template_name (str, optional): Şablon adı
        
        Returns:
            dict: Yüklenen birleştirilmiş konfigürasyon
        
        Raises:
            FileNotFoundError: Belirtilen dosya bulunamazsa
        """
        config_path = config_path or self.config_path
        
        # Başlangıç boş konfigürasyon
        self.config = {}
        
        # 1. Varsayılan konfigürasyonu yükle
        try:
            if os.path.exists(DEFAULT_CONFIG_PATH):
                with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    default_config = yaml.safe_load(f) or {}
                    self.config = default_config
                    logger.debug(f"Varsayılan konfigürasyon yüklendi: {DEFAULT_CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Varsayılan konfigürasyon dosyası yüklenemedi: {e}")
        
        # 2. Eğer şablon belirtildiyse, şablonu yükle ve birleştir
        if template_name:
            try:
                template_path = get_template_path(template_name)
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_config = yaml.safe_load(f) or {}
                    self.config = _merge_configs(self.config, template_config)
                    logger.debug(f"Şablon konfigürasyonu yüklendi ve birleştirildi: {template_path}")
            except Exception as e:
                logger.error(f"Şablon konfigürasyonu yüklenemedi: {e}")
        
        # 3. Kullanıcı konfigürasyonunu yükle ve birleştir
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    self.config = _merge_configs(self.config, user_config)
                    logger.debug(f"Kullanıcı konfigürasyonu yüklendi ve birleştirildi: {config_path}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"YAML dosyası doğru formatlanmamış: {e}")
            except Exception as e:
                logger.error(f"Konfigürasyon dosyası yüklenirken hata: {e}")
        
        # 4. Konfigürasyonu doğrula
        if not validate_config(self.config):
            logger.warning("Geçersiz konfigürasyon. Varsayılan değerlerle devam edilecek.")
            # Burada en azından temel gerekli yapılandırmanın olduğundan emin olalım
            self._ensure_minimal_config()
        
        return self.config
    
    def _ensure_minimal_config(self):
        """
        Geçersiz konfigürasyon durumunda minimum çalışabilir bir konfigürasyon oluşturur.
        """
        # Tokenizer bölümü yoksa veya geçersizse
        if not isinstance(self.config.get('tokenizer'), dict):
            self.config['tokenizer'] = {
                "name": "default_tokenizer",
                "vocab_size": 30000,
                "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
            }
        
        # Data bölümü yoksa veya geçersizse
        if not isinstance(self.config.get('data'), dict):
            self.config['data'] = {
                "sources": [],
                "processors": []
            }
    
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