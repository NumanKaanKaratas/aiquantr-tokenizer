"""
Veri temizleme modülleri.

Bu modül, veri temizleme işlemleri için fonksiyonlar ve sınıflar sağlar.
Ham verileri tokenizer eğitimi için uygun hale getirmek için kullanılır.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterable, Callable

from aiquantr_tokenizer.processors import BaseProcessor
from aiquantr_tokenizer.processors.language_processor import TextProcessor
from aiquantr_tokenizer.processors.code import CodeProcessor

# Logger oluştur
logger = logging.getLogger(__name__)


def clean_text(
    text: str,
    lowercase: bool = False,
    normalize_whitespace: bool = True,
    remove_punct: bool = False,
    max_length: Optional[int] = None,
    **kwargs
) -> str:
    """
    Metin temizleme için hızlı fonksiyon.

    Args:
        text: Temizlenecek metin
        lowercase: Metni küçük harfe çevir (varsayılan: False)
        normalize_whitespace: Boşlukları normalize et (varsayılan: True)
        remove_punct: Noktalama işaretlerini kaldır (varsayılan: False)
        max_length: Maksimum metin uzunluğu (varsayılan: None)
        **kwargs: TextProcessor için ek parametreler

    Returns:
        str: Temizlenmiş metin
    """
    processor = TextProcessor(
        lowercase=lowercase,
        normalize_whitespace=normalize_whitespace,
        remove_punct=remove_punct,
        max_text_length=max_length,
        **kwargs
    )
    return processor(text)


def clean_code(
    code: str, 
    language: str = None,
    remove_comments: bool = False, 
    normalize_whitespace: bool = True,
    **kwargs
) -> str:
    """
    Kod temizleme için hızlı fonksiyon.

    Args:
        code: Temizlenecek kod
        language: Programlama dili (varsayılan: None)
        remove_comments: Yorumları kaldır (varsayılan: False)
        normalize_whitespace: Boşlukları normalize et (varsayılan: True)
        **kwargs: CodeProcessor için ek parametreler

    Returns:
        str: Temizlenmiş kod
    """
    processor = CodeProcessor(
        remove_comments=remove_comments,
        normalize_whitespace=normalize_whitespace,
        **kwargs
    )
    return processor.process(code, language=language)


def clean_file(
    file_path: Union[str, Path],
    processor: Optional[BaseProcessor] = None,
    detect_language: bool = True,
    **kwargs
) -> str:
    """
    Bir dosyayı temizle.

    Args:
        file_path: Temizlenecek dosya yolu
        processor: Kullanılacak işlemci (varsayılan: None - otomatik seçilir)
        detect_language: Dil algılama etkin mi? (varsayılan: True)
        **kwargs: Otomatik oluşturulan işlemciler için ek parametreler

    Returns:
        str: Temizlenmiş içerik
    """
    file_path = Path(file_path)
    
    # Dosya kontrolü
    if not file_path.exists():
        logger.error(f"{file_path} dosyası bulunamadı")
        return ""
        
    if not file_path.is_file():
        logger.error(f"{file_path} bir dosya değil")
        return ""
        
    try:
        # Dosya içeriğini oku
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"{file_path} dosyası okunamadı: {str(e)}")
        return ""
    
    # İşlemci seçimi
    if processor is None:
        # Dosya türüne göre işlemci seç
        suffix = file_path.suffix.lower()
        
        # Kod dosyaları için dil algıla
        language = None
        if detect_language:
            if suffix in ('.py', '.pyw'):
                language = 'python'
            elif suffix in ('.js', '.jsx', '.ts', '.tsx'):
                language = 'javascript'
            elif suffix in ('.c', '.h', '.cpp', '.hpp', '.cc'):
                language = 'c'
            elif suffix in ('.java'):
                language = 'java'
            elif suffix in ('.css'):
                language = 'css'
                
        # Kod dosyası mı?
        is_code = language is not None or suffix in (
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', 
            '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt'
        )
        
        if is_code:
            processor = CodeProcessor(**kwargs)
            return processor.process(content, language=language)
        else:
            # Metin dosyası
            processor = TextProcessor(**kwargs)
            return processor(content)
    else:
        # Verilen işlemciyi kullan
        if isinstance(processor, CodeProcessor):
            # Dil tespiti yap
            language = None
            if detect_language:
                suffix = file_path.suffix.lower()
                if suffix in ('.py', '.pyw'):
                    language = 'python'
                elif suffix in ('.js', '.jsx', '.ts', '.tsx'):
                    language = 'javascript'
                elif suffix in ('.c', '.h', '.cpp', '.hpp', '.cc'):
                    language = 'c'
                elif suffix in ('.java'):
                    language = 'java'
                elif suffix in ('.css'):
                    language = 'css'
            
            return processor.process(content, language=language)
        else:
            return processor(content)


def clean_batch(
    items: Iterable[str],
    processor: BaseProcessor,
    keep_empty: bool = False
) -> List[str]:
    """
    Bir metin dizisini temizle.

    Args:
        items: Temizlenecek metinler
        processor: Kullanılacak işlemci
        keep_empty: Boş sonuçları tut (varsayılan: False)

    Returns:
        List[str]: Temizlenmiş metinler listesi
    """
    results = []
    
    for item in items:
        cleaned = processor(item)
        if keep_empty or cleaned:
            results.append(cleaned)
    
    return results