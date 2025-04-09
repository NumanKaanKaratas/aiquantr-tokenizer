# aiquantr_tokenizer/core/registry.py
"""
Bileşen kaydı (plugin sistemi).

Bu modül, tokenizer hazırlama sürecinde kullanılan
çeşitli bileşenlerin kayıt ve keşif mekanizmasını sağlar.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Any, List, Type, Optional, Callable, Tuple, Set

logger = logging.getLogger(__name__)


class Registry:
    """
    Bileşen kayıt ve yönetim sınıfı.
    
    Bu sınıf, çeşitli tokenizer bileşenlerinin (işlemciler, veri kaynakları, vs.)
    kayıt edilmesi ve erişilmesi için merkezi bir mekanizma sağlar.
    """
    
    def __init__(self):
        """
        Registry sınıfı başlatıcısı.
        """
        self._components = {
            "processor": {
                "language": {},
                "code": {},
                "knowledge": {}
            },
            "data_source": {},
            "tokenizer": {},
            "metric": {}
        }
        
        # Plugin keşfedildi mi?
        self._plugins_discovered = False
        
    def register_component(
        self,
        component_type: str,
        component_class: Type,
        component_id: Optional[str] = None,
        component_subtype: Optional[str] = None,
        overwrite: bool = False
    ) -> bool:
        """
        Bir bileşeni kaydeder.
        
        Args:
            component_type: Bileşen türü ("processor", "data_source", "tokenizer", "metric")
            component_class: Bileşen sınıfı
            component_id: Bileşen kimliği (varsayılan: None - sınıf adından)
            component_subtype: Alt tür (yalnızca "processor" için, varsayılan: None)
            overwrite: Mevcut bir bileşenin üzerine yazılsın mı? (varsayılan: False)
            
        Returns:
            bool: Kayıt başarılı mı?
        """
        # Bileşen kimliğini belirle
        if component_id is None:
            component_id = component_class.__name__.lower()
            
        # Bileşen türünü kontrol et
        if component_type not in self._components:
            logger.warning(f"Bilinmeyen bileşen türü: {component_type}")
            return False
            
        # İşlemci alt türü kontrolü
        if component_type == "processor":
            if component_subtype is None:
                # Alt türü belirle
                if hasattr(component_class, "PROCESSOR_TYPE"):
                    component_subtype = component_class.PROCESSOR_TYPE
                else:
                    logger.warning(f"İşlemci alt türü belirtilmemiş: {component_id}")
                    return False
                    
            # Alt türü kontrol et
            if component_subtype not in self._components[component_type]:
                logger.warning(f"Bilinmeyen işlemci alt türü: {component_subtype}")
                return False
                
            # Mevcut kontrolü
            if component_id in self._components[component_type][component_subtype] and not overwrite:
                logger.warning(
                    f"Bu kimlikle bir işlemci zaten kayıtlı: {component_id} "
                    f"(alt tür: {component_subtype})"
                )
                return False
                
            # İşlemciyi kaydet
            self._components[component_type][component_subtype][component_id] = component_class
            logger.debug(f"İşlemci kaydedildi: {component_id} (alt tür: {component_subtype})")
            
        else:
            # Mevcut kontrolü
            if component_id in self._components[component_type] and not overwrite:
                logger.warning(f"Bu kimlikle bir bileşen zaten kayıtlı: {component_id}")
                return False
                
            # Bileşeni kaydet
            self._components[component_type][component_id] = component_class
            logger.debug(f"Bileşen kaydedildi: {component_id} (tür: {component_type})")
            
        return True
        
    def get_component(
        self,
        component_type: str,
        component_id: str,
        component_subtype: Optional[str] = None
    ) -> Optional[Type]:
        """
        Bir bileşeni alır.
        
        Args:
            component_type: Bileşen türü
            component_id: Bileşen kimliği
            component_subtype: Alt tür (yalnızca "processor" için)
            
        Returns:
            Optional[Type]: Bileşen sınıfı veya None
        """
        # Bileşen türünü kontrol et
        if component_type not in self._components:
            logger.warning(f"Bilinmeyen bileşen türü: {component_type}")
            return None
            
        # İşlemci kontrolü
        if component_type == "processor":
            if component_subtype is None:
                logger.warning("İşlemci alt türü belirtilmemiş")
                return None
                
            if component_subtype not in self._components[component_type]:
                logger.warning(f"Bilinmeyen işlemci alt türü: {component_subtype}")
                return None
                
            return self._components[component_type][component_subtype].get(component_id)
            
        # Diğer bileşen türleri
        return self._components[component_type].get(component_id)
        
    def get_components(
        self,
        component_type: str,
        component_subtype: Optional[str] = None
    ) -> Dict[str, Type]:
        """
        Belirli türdeki tüm bileşenleri alır.
        
        Args:
            component_type: Bileşen türü
            component_subtype: Alt tür (yalnızca "processor" için)
            
        Returns:
            Dict[str, Type]: Bileşen kimliği -> sınıf eşlemesi
        """
        # Bileşen türünü kontrol et
        if component_type not in self._components:
            logger.warning(f"Bilinmeyen bileşen türü: {component_type}")
            return {}
            
        # İşlemci kontrolü
        if component_type == "processor":
            if component_subtype is None:
                # Tüm işlemci alt türlerini birleştir
                combined = {}
                for subtype in self._components[component_type]:
                    combined.update(self._components[component_type][subtype])
                return combined
                
            if component_subtype not in self._components[component_type]:
                logger.warning(f"Bilinmeyen işlemci alt türü: {component_subtype}")
                return {}
                
            return self._components[component_type][component_subtype].copy()
            
        # Diğer bileşen türleri
        return self._components[component_type].copy()
        
    def list_components(
        self,
        component_type: Optional[str] = None,
        component_subtype: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Kayıtlı bileşenlerin listesini döndürür.
        
        Args:
            component_type: Bileşen türü (varsayılan: None - tüm türler)
            component_subtype: Alt tür (yalnızca "processor" için)
            
        Returns:
            Dict[str, List[str]]: Bileşen listesi
        """
        result = {}
        
        # Belirli bir tür için
        if component_type:
            if component_type not in self._components:
                logger.warning(f"Bilinmeyen bileşen türü: {component_type}")
                return {}
                
            if component_type == "processor":
                # İşlemciler için alt tür kontrolü
                if component_subtype:
                    if component_subtype not in self._components[component_type]:
                        logger.warning(f"Bilinmeyen işlemci alt türü: {component_subtype}")
                        return {}
                        
                    result[f"{component_type}.{component_subtype}"] = list(
                        self._components[component_type][component_subtype].keys()
                    )
                else:
                    # Tüm işlemci alt türleri
                    for subtype in self._components[component_type]:
                        result[f"{component_type}.{subtype}"] = list(
                            self._components[component_type][subtype].keys()
                        )
            else:
                # Diğer bileşen türleri
                result[component_type] = list(self._components[component_type].keys())
                
        else:
            # Tüm bileşen türleri
            for ctype in self._components:
                if ctype == "processor":
                    for subtype in self._components[ctype]:
                        result[f"{ctype}.{subtype}"] = list(
                            self._components[ctype][subtype].keys()
                        )
                else:
                    result[ctype] = list(self._components[ctype].keys())
                    
        return result
        
    def discover_plugins(self, package_name: str = "aiquantr_tokenizer") -> Set[str]:
        """
        Paket içindeki eklentileri keşfeder.
        
        Bu yöntem, belirtilen paket içindeki tüm modülleri tarar ve 
        ilgili bileşen sınıflarını kaydetmeye çalışır.
        
        Args:
            package_name: Taranacak paket adı
            
        Returns:
            Set[str]: Keşfedilen eklenti adları
        """
        if self._plugins_discovered:
            logger.debug("Eklentiler zaten keşfedildi")
            return set()
            
        discovered = set()
        
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.error(f"Paket bulunamadı: {package_name}")
            return discovered
            
        # Alt paketleri tara
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            if is_pkg:
                # Alt paketleri özyinelemeli olarak tara
                discovered.update(self.discover_plugins(module_name))
            else:
                try:
                    # Modülü içe aktar
                    module = importlib.import_module(module_name)
                    
                    # Modüldeki sınıfları tara
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Temel bileşen sınıflarını kontrol et
                        if hasattr(obj, "COMPONENT_TYPE"):
                            component_type = getattr(obj, "COMPONENT_TYPE")
                            component_subtype = getattr(obj, "COMPONENT_SUBTYPE", None)
                            component_id = getattr(obj, "COMPONENT_ID", name.lower())
                            
                            # Bileşeni kaydet
                            if self.register_component(
                                component_type, obj, component_id, component_subtype
                            ):
                                discovered.add(f"{component_type}.{component_id}")
                                
                except ImportError as e:
                    logger.warning(f"Modül içe aktarılamadı: {module_name} - {e}")
                    
        self._plugins_discovered = True
        return discovered