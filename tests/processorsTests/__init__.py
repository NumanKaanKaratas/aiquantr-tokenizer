"""
aiquantr-tokenizer kod işleme testleri.

Bu paket, kod işleme modüllerinin
test sınıflarını ve test durumlarını içerir.
"""

# Bu modül içindeki testlere yönelik yardımcı fonksiyonlar veya sabitler burada tanımlanabilir

# Test verileri için sabit değerler
SAMPLE_CODE_PYTHON = """
def example():
    # Bir yorum
    return "test"
"""

SAMPLE_CODE_PHP = """<?php
// Bir yorum
function example() {
    return "test";
}
?>"""

# Test yardımcı fonksiyonları
def get_processor_test_file_path(language, filename):
    """
    Belirtilen dil ve dosya adı için test dosya yolunu döndürür.
    
    Args:
        language: Dil adı 
        filename: Dosya adı
    
    Returns:
        str: Test dosyasının tam yolu
    """
    import os
    return os.path.join(
        os.path.dirname(__file__), 
        "test_files",
        language, 
        filename
    )