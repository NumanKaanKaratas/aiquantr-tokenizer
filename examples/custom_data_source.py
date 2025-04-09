#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Özel veri kaynağı örneği.

Bu betik, aiquantr_tokenizer kütüphanesi ile özel veri
kaynağı kullanımını gösterir.
"""

import os
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Iterator

# Loglama seviyesini ayarla
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Veri kaynakları import
from aiquantr_tokenizer.data.sources import BaseDataSource, CustomDataSource


# Özel veri kaynağı tanımlama
class QuoteGenerator(BaseDataSource):
    """
    Rastgele alıntı üreten bir veri kaynağı.
    """
    
    def __init__(self, num_quotes: int = 10, **kwargs):
        """
        QuoteGenerator başlatıcısı.
        
        Args:
            num_quotes: Üretilecek alıntı sayısı
            **kwargs: BaseDataSource için ek parametreler
        """
        super().__init__(
            name="AlıntıÜreteci",
            description="Örnek alıntılar üreten bir kaynak",
            **kwargs
        )
        self.num_quotes = num_quotes
        self.quotes = [
            "Hayat, anlayabildiğin kadar basit, yapabildiğin kadar zevkli, üzülebildiğin kadar acı, gülebildiğin kadar tatlıdır.",
            "Başarı, her gün düşünebildiğinizden biraz daha fazlasını yapmaktır.",
            "Öğrenmeyi bıraktığın an, ölmeye başlarsın.",
            "Bir insanın idealleri olmalı ve onlara ulaşmak için çalışmalıdır.",
            "Hayatta en hakiki mürşit ilimdir.",
            "Zor şartlar altında, gerçek karakterler ortaya çıkar.",
            "Zamanın kıymetini bilmek, hayatın kıymetini bilmektir.",
            "Bilgi en büyük hazinedir, çalınmaz ve taşınması kolaydır.",
            "Mutluluğu tatmanın tek çaresi, onu paylaşmaktır.",
            "Dün bilgeydi, bugünü yaşa, yarın ise bir sırdır.",
            "Başarının sırrı, başlamaktadır.",
            "Bildiğini zannettiğin şey, öğrenmenin önündeki en büyük engeldir.",
            "En büyük zafer, insanın kendisine karşı kazandığıdır.",
            "İnsan adaletle yüceltilir ve liyakatle ilerler.",
            "Okumaktan korkmayın, hayal kurmaktan korkmayın.",
            "Geleceği öngörmek için, onu icat etmek gerekir.",
            "Kendini tanımak, tüm bilgeliğin başlangıcıdır.",
            "Eğitim, dünyayı değiştirmek için kullanabileceğiniz en güçlü silahtır.",
            "Yaşamak, düşünmeden eyleme geçmeyi; düşünmek ise eyleme geçmeden yaşamayı sağlar.",
            "Bir işte başarıya ulaşmanın yolu, sabır ve sebattır."
        ]
        
        # Metadata ekle
        self.metadata.update({
            "source": "quote_generator",
            "total_available": len(self.quotes),
            "language": "tr"
        })
    
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """
        Veri kaynağından veri yükleme.
        
        Yields:
            Dict[str, Any]: Üretilen alıntı öğeleri
        """
        # Kaç alıntı üretileceğini belirle
        count = min(self.num_quotes, len(self.quotes))
        
        # Alıntıları karıştır ve seç
        selected_quotes = random.sample(self.quotes, count)
        
        # Her alıntı için
        for i, quote in enumerate(selected_quotes):
            # İstatistiği güncelle
            self.stats["total_samples"] += 1
            
            # Öğe oluştur
            item = {
                self.text_key: quote,
                "id": i + 1,
                "source": "quote_collection"
            }
            
            # Öğeyi işle
            processed_item = self.process_item(item)
            if processed_item:
                yield processed_item


def sample_generator_function() -> Iterator[Dict[str, Any]]:
    """
    Örnek bir üretici fonksiyon.
    
    Yields:
        Dict[str, Any]: Üretilen veri öğeleri
    """
    data = [
        {"text": "Fonksiyondan üretilen örnek metin 1", "source": "function_generator"},
        {"text": "Fonksiyondan üretilen örnek metin 2", "source": "function_generator"},
        {"text": "Fonksiyondan üretilen örnek metin 3", "source": "function_generator"}
    ]
    
    for item in data:
        yield item


def main():
    print("=== Özel Veri Kaynağı Örneği ===\n")
    
    # Özel alıntı üreticisi kullanma
    quote_source = QuoteGenerator(num_quotes=5, min_length=50)
    
    print("Alıntı üretecinden veriler:")
    for i, item in enumerate(quote_source.load_data(), 1):
        print(f"{i}. {item['text']}")
    
    # Üretici istatistiklerini göster
    print(f"\nÜretici istatistikleri: {quote_source.get_stats()}")
    print(f"Üretici metadatası: {quote_source.get_metadata()}")
    
    # Fonksiyondan özel veri kaynağı oluşturma
    function_source = CustomDataSource(
        data_source=sample_generator_function,
        name="FonksiyonKaynağı"
    )
    
    print("\nFonksiyon kaynağından veriler:")
    for i, item in enumerate(function_source.load_data(), 1):
        print(f"{i}. {item['text']}")
    
    # Listeden özel veri kaynağı oluşturma
    list_data = [
        "Listeden gelen veri 1",
        "Listeden gelen veri 2", 
        "Listeden gelen veri 3",
        "Listeden gelen veri 4"
    ]
    
    list_source = CustomDataSource(
        data_source=list_data,
        name="ListeKaynağı"
    )
    
    print("\nListe kaynağından veriler:")
    for i, item in enumerate(list_source.load_data(), 1):
        print(f"{i}. {item['text']}")
    
    print("\nÖrnek tamamlandı.")


if __name__ == "__main__":
    main()