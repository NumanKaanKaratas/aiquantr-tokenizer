"""
aiquantr-tokenizer test paketleri.

Bu paket, aiquantr-tokenizer projesinin
test modüllerini ve test yardımcı fonksiyonlarını içerir.
"""

# Test yardımcı fonksiyonlarını veya sabitlerini burada tanımlayabilirsiniz
import os
import sys

# Proje kök dizinini ekleme (gerekirse)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)