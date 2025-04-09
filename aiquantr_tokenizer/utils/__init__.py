# aiquantr_tokenizer/utils/__init__.py
"""
Tokenizer Prep Utils Modülü.

Bu modül, tokenizer hazırlama sürecinde kullanılan
çeşitli yardımcı fonksiyonları ve araçları sağlar.
"""

# Loglama yardımcıları
from .logging_utils import setup_logger, get_logger

# Dosya işleme fonksiyonları
from .io_utils import (
    # Temel dosya işlemleri
    ensure_dir,
    file_exists,
    read_file,
    write_file,
    safe_write_file,
    
    # JSON/YAML işlemleri
    load_json,
    save_json,
    load_yaml,
    save_yaml,
    
    # Dosya yönetimi
    copy_file,
    delete_file,
    list_files,
    create_temp_file,
    create_temp_dir,
    get_file_size,
    get_file_modification_time,
    read_chunks,
    count_lines,
    
    # Veri çıkarma işlemleri
    extract_texts,
    extract_texts_from_json,
    extract_texts_from_file,
    extract_texts_from_directory,
    extract_texts_from_csv,
    extract_texts_from_yaml,
    
    # Veri işleme fonksiyonları
    split_data,
    save_texts,
    count_tokens
)

# Paralel işleme
from .parallel_utils import process_in_parallel

# Bağımlılık kontrolü
from .dependency_check import (
    check_python_version,
    check_package_installed,
    check_package_version,
    get_installed_packages,
    check_dependencies,
    install_package,
    install_dependencies,
    get_missing_dependencies,
    check_compatibility,
    get_system_info,
    display_dependencies_status,
    display_system_info,
    DEPENDENCY_GROUPS
)

# HuggingFace Hub entegrasyonu
from .hub_utils import (
    check_hub_available,
    login_to_hub,
    upload_to_hub,
    download_from_hub,
    push_to_hub,
    list_hub_models
)

# Genel yardımcı işlevler
def set_seed(seed: int):
    """
    Rastgele sayı üreticileri için başlangıç değerini ayarlar.
    
    Args:
        seed: Rastgele başlangıç değeri
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


__all__ = [
    # Loglama yardımcıları
    "setup_logger",
    "get_logger",
    
    # Dosya işlemleri
    "ensure_dir",
    "file_exists",
    "read_file",
    "write_file",
    "safe_write_file",
    "load_json",
    "save_json",
    "load_yaml",
    "save_yaml",
    "copy_file",
    "delete_file",
    "list_files",
    "create_temp_file",
    "create_temp_dir",
    "get_file_size",
    "get_file_modification_time",
    "read_chunks",
    "count_lines",
    
    # Veri çıkarma işlemleri
    "extract_texts",
    "extract_texts_from_json",
    "extract_texts_from_file",
    "extract_texts_from_directory",
    "extract_texts_from_csv",
    "extract_texts_from_yaml",
    
    # Veri işleme 
    "split_data",
    "save_texts",
    "count_tokens",
    
    # Paralel işleme
    "process_in_parallel",
    
    # Bağımlılık kontrolü
    "check_dependencies",
    "check_python_version",
    "check_package_installed",
    "check_package_version",
    "get_installed_packages",
    "install_package",
    "install_dependencies",
    "get_missing_dependencies",
    "check_compatibility",
    "get_system_info",
    "display_dependencies_status",
    "display_system_info",
    "DEPENDENCY_GROUPS",
    
    # HuggingFace Hub yardımcıları
    "check_hub_available",
    "login_to_hub",
    "upload_to_hub",
    "download_from_hub",
    "push_to_hub",
    "list_hub_models",
    
    # Genel yardımcı işlevler
    "set_seed"
]