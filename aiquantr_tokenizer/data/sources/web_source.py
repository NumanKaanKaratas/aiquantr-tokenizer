"""
Web tabanlı veri kaynakları.

Bu modül, web sayfalarından ve API'lerden veri
yüklemek için kullanılan sınıfları içerir.
"""

import re
import json
import time
import logging
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List, Iterator, Union, Set, Tuple

from tokenizer_prep.data.sources.base_source import BaseDataSource

# Logger oluştur
logger = logging.getLogger(__name__)

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests paketi yüklü değil. Web veri kaynakları kullanılamaz.")


class WebSource(BaseDataSource):
    """
    Web'den veri yükleyen temel kaynak.
    """
    
    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        retry_attempts: int = 3,
        retry_backoff: float = 0.5,
        delay_between_requests: float = 0,
        **kwargs
    ):
        """
        WebSource başlatıcısı.
        
        Args:
            headers: HTTP istek başlıkları
            timeout: İstek zaman aşımı (saniye)
            retry_attempts: Yeniden deneme sayısı
            retry_backoff: Yeniden deneme gecikmesi katsayısı
            delay_between_requests: İstekler arası gecikme (saniye)
            **kwargs: BaseDataSource için ek parametreler
        """
        super().__init__(**kwargs)
        
        if not HAS_REQUESTS:
            raise ImportError("Web kaynaklarını kullanmak için 'requests' paketi gereklidir.")
        
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.delay_between_requests = delay_between_requests
        
        # Yeniden deneme stratejisi
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        Yeniden deneme stratejisi ile bir HTTP oturum nesnesi oluşturur.
        
        Returns:
            requests.Session: Oturum nesnesi
        """
        session = requests.Session()
        
        # Yeniden deneme stratejisi tanımla
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        # Adaptörü ayarla
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def fetch_url(self, url: str, method: str = "GET", 
                  data: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[str]]:
        """
        URL'den içerik alır.
        
        Args:
            url: İstek URL'i
            method: HTTP istek yöntemi
            data: POST isteği için veri
            
        Returns:
            Tuple[int, Optional[str]]: (HTTP durum kodu, içerik)
        """
        try:
            # İstek yap
            if method.upper() == "GET":
                response = self.session.get(url, headers=self.headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=self.headers, json=data, timeout=self.timeout)
            else:
                logger.error(f"Desteklenmeyen HTTP yöntemi: {method}")
                return 0, None
            
            # Durum kodunu kontrol et
            response.raise_for_status()
            
            # İstekler arası gecikme
            if self.delay_between_requests > 0:
                time.sleep(self.delay_between_requests)
                
            return response.status_code, response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"{url} adresinden veri alınamadı: {str(e)}")
            return 0, None
    
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        Web'den veri yükleme.
        
        Yields:
            Dict[str, Any]: Yüklenen veri öğeleri
            
        Note:
            Bu temel sınıf doğrudan veri yüklemez. Alt sınıflar bu metodu uygulamalıdır.
        """
        logger.warning("WebSource.load_data() metodu doğrudan çağrılmamalıdır.")
        return []
        yield from []  # Bu satır asla çalışmaz, sadece tip kontrolü için


class URLSource(WebSource):
    """
    Belirli URL'lerden veri yükleyen kaynak.
    """
    
    def __init__(
        self,
        urls: List[str],
        extract_regex: Optional[str] = None,
        json_path: Optional[str] = None,
        **kwargs
    ):
        """
        URLSource başlatıcısı.
        
        Args:
            urls: Yüklenecek URL'lerin listesi
            extract_regex: İçerikten metin çıkarmak için regex deseni
            json_path: JSON yanıtından metin çıkarma yolu (örn: "data.items.text")
            **kwargs: WebSource için ek parametreler
        """
        super().__init__(**kwargs)
        
        self.urls = urls
        self.extract_regex = re.compile(extract_regex) if extract_regex else None
        self.json_path = json_path.split('.') if json_path else None
    
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        URL listesinden veri yükleme.
        
        Yields:
            Dict[str, Any]: Yüklenen veri öğeleri
        """
        count = 0
        for url in self.urls:
            # Limit kontrolü
            if self.limit and count >= self.limit:
                break
            
            # İstatistiği güncelle
            self.stats["total_samples"] += 1
            
            # URL'den içerik al
            status_code, content = self.fetch_url(url)
            
            if status_code != 200 or not content:
                self.stats["skipped_samples"] += 1
                continue
            
            # İçerikten veri çıkar
            processed_item = self._process_content(content, url)
            if processed_item:
                yield processed_item
                count += 1
    
    def _process_content(self, content: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Ham içeriği işler.
        
        Args:
            content: İşlenecek içerik
            url: İçeriğin alındığı URL
            
        Returns:
            Optional[Dict[str, Any]]: İşlenmiş veri öğesi veya None
        """
        try:
            # JSON içerik kontrolü
            if content.strip().startswith('{') and content.strip().endswith('}'):
                try:
                    # JSON olarak parse et
                    json_data = json.loads(content)
                    
                    # JSON yolunu takip et
                    if self.json_path:
                        data = json_data
                        for key in self.json_path:
                            if isinstance(data, dict) and key in data:
                                data = data[key]
                            elif isinstance(data, list) and key.isdigit():
                                index = int(key)
                                if 0 <= index < len(data):
                                    data = data[index]
                                else:
                                    logger.warning(f"Geçersiz JSON dizin: {key}")
                                    return None
                            else:
                                logger.warning(f"JSON yolu izlenemedi: {key}")
                                return None
                        
                        # Text değerini ayarla
                        if isinstance(data, str):
                            text = data
                        else:
                            text = json.dumps(data, ensure_ascii=False)
                    else:
                        # Tüm JSON'u metin olarak kullan
                        text = content
                        
                except json.JSONDecodeError:
                    # JSON parse edilemedi, düz metin olarak devam et
                    text = content
            else:
                # Düz metin
                text = content
            
            # Regex deseni ile metin çıkarma
            if self.extract_regex:
                match = self.extract_regex.search(text)
                if match:
                    if match.groups():
                        text = match.group(1)  # İlk yakalanan grup
                    else:
                        text = match.group(0)  # Tam eşleşme
            
            # Öğeyi oluştur ve işle
            item = {
                self.text_key: text,
                "source_url": url
            }
            
            return self.process_item(item)
            
        except Exception as e:
            logger.error(f"İçerik işleme hatası: {str(e)}")
            return None