#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temel kullanım örneği.

Bu betik, aiquantr_tokenizer kütüphanesinin temel
kullanımını gösterir.
"""

import os
import logging
from pathlib import Path

# Loglama seviyesini ayarla
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ana paket import
import aiquantr_tokenizer
from aiquantr_tokenizer.data import clean_text, filter_dataset
from aiquantr_tokenizer.processors import BaseProcessor
from aiquantr_tokenizer.processors.language_processor import TextProcessor
from aiquantr_tokenizer.processors.code import CodeProcessor, PythonProcessor
from aiquantr_tokenizer.metrics import calculate_text_diversity


def main():
    print(f"aiquantr_tokenizer sürümü: {aiquantr_tokenizer.__version__}")
    
    # Örnek metin
    sample_text = """
    Tokenizer eğitimi için bir örnek metin.
    Bu metin içerisinde https://example.com gibi URL'ler,
    user@example.com gibi e-postalar, ve 12345 gibi sayılar bulunabilir.
    
    Farklı satırlar ve     fazla boşluklar da olabilir.
    """
    
    # Temel metin işleme
    print("\n=== Temel Metin İşleme ===")
    cleaned_text = clean_text(
        sample_text,
        normalize_whitespace=True,
        replace_urls=True,
        replace_emails=True
    )
    print(cleaned_text)
    
    # Örnek Python kodu
    python_code = """
    # Bu bir örnek Python kodudur
    def hello(name: str) -> str:
        '''Basit bir karşılama fonksiyonu'''
        return f"Merhaba, {name}!"
    
    # Ana program
    if __name__ == "__main__":
        print(hello("Dünya"))
    """
    
    # Python kodu işleme
    print("\n=== Python Kodu İşleme ===")
    python_processor = PythonProcessor(
        remove_comments=True,
        remove_docstrings=True,
        remove_type_hints=True
    )
    processed_code = python_processor(python_code)
    print(processed_code)
    
    # Metin çeşitlilik metrikleri
    print("\n=== Metin Çeşitlilik Metrikleri ===")
    diversity_metrics = calculate_text_diversity(cleaned_text)
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Özel işlemci oluşturma örneği
    class CustomProcessor(BaseProcessor):
        """Örnek özel işlemci"""
        
        def process(self, text):
            # Özel işleme (örnek: tüm kelimeleri büyük harfe çevirme)
            if isinstance(text, str):
                words = text.split()
                return " ".join([word.upper() for word in words])
            return text
    
    # Özel işlemci kullan
    print("\n=== Özel İşlemci ===")
    custom_processor = CustomProcessor(name="BüyükHarfÇevirici")
    result = custom_processor("bu bir özel işlemci örneğidir")
    print(result)
    
    print("\nÖrnek tamamlandı.")


if __name__ == "__main__":
    main()