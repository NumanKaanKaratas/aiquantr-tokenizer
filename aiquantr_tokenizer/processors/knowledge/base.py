"""
Temel bilgi işleme sınıfı.

Bu modül, çeşitli bilgi türlerini işlemek için temel sınıfı tanımlar.
Tüm bilgi işlemcileri bu temel sınıfı miras almalıdır.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set, Tuple

from processors.base_processor import BaseProcessor

# Logger oluştur
logger = logging.getLogger(__name__)


class BaseKnowledgeProcessor(BaseProcessor):
    """
    Bilgi işleme için temel sınıf.
    
    Bu soyut temel sınıf, farklı bilgi türlerine özel işlemciler için
    ortak arayüzü ve işlevselliği tanımlar.
    """
    
    def __init__(
        self,
        knowledge_type: str,
        domain: Optional[str] = None,
        max_length: Optional[int] = None,
        normalize_text: bool = True,
        extract_entities: bool = False,
        extract_facts: bool = False,
        extract_relations: bool = False,
        **kwargs
    ):
        """
        BaseKnowledgeProcessor başlatıcısı.
        
        Args:
            knowledge_type: Bilgi türü (makale, belge, wiki vb.)
            domain: Bilgi alanı (tıp, hukuk, teknoloji vb.)
            max_length: Maksimum çıktı uzunluğu
            normalize_text: Metni normalleştir (varsayılan: True)
            extract_entities: Varlıkları çıkar (varsayılan: False)
            extract_facts: Gerçekleri çıkar (varsayılan: False)
            extract_relations: İlişkileri çıkar (varsayılan: False)
            **kwargs: BaseProcessor için ek parametreler
        """
        super().__init__(**kwargs)
        
        self.knowledge_type = knowledge_type
        self.domain = domain
        self.max_length = max_length
        self.normalize_text = normalize_text
        self.extract_entities = extract_entities
        self.extract_facts = extract_facts
        self.extract_relations = extract_relations
        
        # İstatistikler
        self.stats.update({
            "entities_extracted": 0,
            "facts_extracted": 0,
            "relations_extracted": 0,
            "paragraphs_processed": 0
        })
    
    def process(self, text: str) -> str:
        """
        Bilgi metni üzerinde işleme yapar.
        
        Args:
            text: İşlenecek bilgi metni
            
        Returns:
            str: İşlenmiş metin
        """
        if not text:
            return ""
        
        # Metni işle
        processed_text = self._preprocess_text(text)
        
        # Varlıkları çıkar (isteğe bağlı)
        if self.extract_entities:
            entities = self.extract_entities_from_text(processed_text)
            self.stats["entities_extracted"] += len(entities)
        
        # Gerçekleri çıkar (isteğe bağlı)
        if self.extract_facts:
            facts = self.extract_facts_from_text(processed_text)
            self.stats["facts_extracted"] += len(facts)
        
        # İlişkileri çıkar (isteğe bağlı)
        if self.extract_relations:
            relations = self.extract_relations_from_text(processed_text)
            self.stats["relations_extracted"] += len(relations)
        
        # Son işleme
        result = self._postprocess_text(processed_text)
        
        # Uzunluk sınırlaması
        if self.max_length and len(result) > self.max_length:
            result = result[:self.max_length]
        
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """
        Ön işleme adımlarını gerçekleştirir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Ön işlenmiş metin
        """
        # Paragrafları sayar
        paragraphs = text.split('\n\n')
        self.stats["paragraphs_processed"] += len(paragraphs)
        
        # Normalleştirme
        if self.normalize_text:
            text = self._normalize_text(text)
            
        return text
    
    def _normalize_text(self, text: str) -> str:
        """
        Metni normalleştirir.
        
        Args:
            text: Normalleştirilecek metin
            
        Returns:
            str: Normalleştirilmiş metin
        """
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Birden fazla yeni satırları tek yeni satıra dönüştür
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Başlangıç ve sondaki boşlukları temizle
        text = text.strip()
        
        return text
    
    def _postprocess_text(self, text: str) -> str:
        """
        Son işleme adımlarını gerçekleştirir.
        
        Alt sınıflar tarafından geçersiz kılınabilir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Son işlenmiş metin
        """
        return text
    
    @abstractmethod
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Metinden varlık bilgilerini çıkarır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[Dict[str, Any]]: Çıkarılan varlık bilgileri
        """
        pass
    
    @abstractmethod
    def extract_facts_from_text(self, text: str) -> List[str]:
        """
        Metinden gerçek bilgilerini çıkarır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[str]: Çıkarılan gerçek ifadeleri
        """
        pass
    
    @abstractmethod
    def extract_relations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Metinden ilişki bilgilerini çıkarır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[Dict[str, Any]]: Çıkarılan ilişki bilgileri
        """
        pass
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Metinden anahtar kelimeler çıkarır.
        
        Args:
            text: İşlenecek metin
            max_keywords: Maksimum anahtar kelime sayısı
            
        Returns:
            List[str]: Çıkarılan anahtar kelimeler
        """
        # Basit anahtar kelime çıkarma (sıklık bazlı)
        words = text.lower().split()
        
        # Durdurma kelimeleri (basit bir liste)
        stop_words = {'ve', 'veya', 'bir', 'bu', 'şu', 'o', 'de', 'da', 'ki', 'mi',
                     'için', 'ile', 'and', 'or', 'the', 'a', 'an', 'of', 'to', 'in',
                     'that', 'this', 'is', 'are', 'was', 'were'}
        
        # Durdurma kelimelerini kaldır
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Kelime frekanslarını hesapla
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Frekansa göre sırala ve en yüksek olanları al
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords