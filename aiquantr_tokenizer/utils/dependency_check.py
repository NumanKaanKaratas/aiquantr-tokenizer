# aiquantr_tokenizer/utils/dependency_check.py
"""
Bağımlılık kontrolü için yardımcı fonksiyonlar.

Bu modül, tokenizer hazırlama sürecinde gerekli
Python bağımlılıklarını kontrol etmek ve yönetmek için
işlevler sağlar.
"""

import sys
import platform
import logging
import importlib
import subprocess
from typing import Dict, List, Set, Tuple, Optional, Union, Any

# Logger oluştur
logger = logging.getLogger(__name__)

# Özel bağımlılık grupları
DEPENDENCY_GROUPS = {
    "core": {
        "numpy": "1.18.0", 
        "tqdm": "4.45.0",
        "pyyaml": "5.1.0",
        "requests": "2.20.0"
    },
    "huggingface": {
        "transformers": "4.0.0",
        "datasets": "1.0.0",
        "huggingface_hub": "0.0.1"
    },
    "sentencepiece": {
        "sentencepiece": "0.1.96"
    },
    "visualization": {
        "matplotlib": "3.0.0",
        "seaborn": "0.9.0"
    },
    "language": {
        "nltk": "3.5",
        "spacy": "3.0.0"
    },
    "code": {
        "pygments": "2.7.0"
    }
}


def check_python_version(min_version: str = "3.7.0") -> bool:
    """
    Python sürümünün yeterli olup olmadığını kontrol eder.
    
    Args:
        min_version: Minimum gerekli Python sürümü (varsayılan: "3.7.0")
        
    Returns:
        bool: Python sürümü yeterli mi?
    """
    current_version = platform.python_version()
    current_parts = [int(x) for x in current_version.split(".")]
    required_parts = [int(x) for x in min_version.split(".")]
    
    for i in range(len(required_parts)):
        if i >= len(current_parts):
            return False
        if current_parts[i] > required_parts[i]:
            return True
        if current_parts[i] < required_parts[i]:
            return False
            
    return True


def check_package_installed(package_name: str) -> bool:
    """
    Python paketinin yüklü olup olmadığını kontrol eder.
    
    Args:
        package_name: Paket adı
        
    Returns:
        bool: Paket yüklü mü?
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def check_package_version(
    package_name: str,
    min_version: Optional[str] = None,
    silent: bool = True
) -> bool:
    """
    Paketin belirli bir sürümünün yüklü olup olmadığını kontrol eder.
    
    Args:
        package_name: Paket adı
        min_version: Minimum gereken sürüm (varsayılan: None)
        silent: Hataları gizle (varsayılan: True)
        
    Returns:
        bool: Paket yeterli sürümde mi?
    """
    if not check_package_installed(package_name):
        return False
        
    # Sürüm kontrolü gerekmiyorsa
    if min_version is None:
        return True
        
    try:
        # Modülü içe aktar
        module = importlib.import_module(package_name)
        
        # Sürümü al
        if hasattr(module, "__version__"):
            version = getattr(module, "__version__")
        elif hasattr(module, "VERSION"):
            version = getattr(module, "VERSION")
        elif hasattr(module, "version"):
            version = getattr(module, "version")
        else:
            # pkg_resources ile dene
            import pkg_resources
            version = pkg_resources.get_distribution(package_name).version

        # Sürümleri karşılaştır
        return compare_versions(version, min_version) >= 0
        
    except Exception as e:
        if not silent:
            logger.warning(f"Paket sürümü kontrol edilemedi ({package_name}): {e}")
        return False


def compare_versions(version1: str, version2: str) -> int:
    """
    İki sürüm numarasını karşılaştırır.
    
    Args:
        version1: Birinci sürüm
        version2: İkinci sürüm
        
    Returns:
        int: version1 > version2 ise 1, version1 < version2 ise -1, eşitse 0
    """
    # Sürüm bileşenlerini al
    parts1 = [int(x) if x.isdigit() else x for x in version1.split(".")]
    parts2 = [int(x) if x.isdigit() else x for x in version2.split(".")]
    
    # Karşılaştır
    for i in range(max(len(parts1), len(parts2))):
        # Eksik parçaları ele al
        if i >= len(parts1):
            return -1  # version1 daha kısa (küçük)
        if i >= len(parts2):
            return 1   # version2 daha kısa (küçük)
            
        # Farklı parçaları karşılaştır
        if parts1[i] != parts2[i]:
            if isinstance(parts1[i], int) and isinstance(parts2[i], int):
                return 1 if parts1[i] > parts2[i] else -1
            else:
                # Sayı olmayan parçaları alfanümerik olarak karşılaştır
                return 1 if str(parts1[i]) > str(parts2[i]) else -1
                
    return 0  # Eşit


def get_installed_packages() -> List[Tuple[str, str]]:
    """
    Yüklü Python paketlerini sürümleri ile birlikte alır.
    
    Returns:
        List[Tuple[str, str]]: [(paket_adı, sürüm), ...]
    """
    try:
        import pkg_resources
        return [(dist.key, dist.version) for dist in pkg_resources.working_set]
    except ImportError:
        logger.warning("pkg_resources bulunamadı. Paket listesi alınamadı.")
        return []


def check_dependencies(
    required_packages: Dict[str, str] = None,
    groups: List[str] = None
) -> Dict[str, bool]:
    """
    Belirtilen bağımlılıkları kontrol eder.
    
    Args:
        required_packages: {paket_adı: min_sürüm} şeklinde sözlük
        groups: Kontrol edilecek bağımlılık grupları
        
    Returns:
        Dict[str, bool]: {paket_adı: yüklü_mü} şeklinde sözlük
    """
    packages_to_check = {}
    
    # Grupları kullan
    if groups is not None:
        for group in groups:
            if group in DEPENDENCY_GROUPS:
                packages_to_check.update(DEPENDENCY_GROUPS[group])
            else:
                logger.warning(f"Bilinmeyen bağımlılık grubu: {group}")
    
    # Doğrudan paket listesi kullan
    if required_packages is not None:
        packages_to_check.update(required_packages)
    
    # Hiçbir paket belirtilmediyse, temel grubu kullan
    if not packages_to_check and "core" in DEPENDENCY_GROUPS:
        packages_to_check = DEPENDENCY_GROUPS["core"]
    
    # Paketleri kontrol et
    results = {}
    for package, version in packages_to_check.items():
        results[package] = check_package_version(package, min_version=version)
        
    return results


def install_package(
    package_name: str,
    version: Optional[str] = None,
    upgrade: bool = False
) -> bool:
    """
    Belirtilen Python paketini pip ile kurar.
    
    Args:
        package_name: Kurulacak paket adı
        version: İstenen sürüm (varsayılan: None - en son sürüm)
        upgrade: Paketi güncelle (varsayılan: False)
        
    Returns:
        bool: Kurulum başarılı mı?
    """
    try:
        # pip komutunu yapılandır
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade:
            cmd.append("-U")
            
        # Versiyon belirtilmişse ekle
        if version:
            cmd.append(f"{package_name}=={version}")
        else:
            cmd.append(package_name)
            
        # Komutu çalıştır
        logger.info(f"Paket kuruluyor: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Paket kuruldu: {package_name}")
            return True
        else:
            logger.error(f"Paket kurulum hatası: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"Paket kurulum hatası ({package_name}): {e}")
        return False


def install_dependencies(
    required_packages: Dict[str, str] = None,
    groups: List[str] = None,
    upgrade: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Eksik bağımlılıkları kurar.
    
    Args:
        required_packages: {paket_adı: min_sürüm} sözlüğü
        groups: Kurulacak bağımlılık grupları
        upgrade: Paketleri yükselt (varsayılan: False)
        
    Returns:
        Tuple[List[str], List[str]]: (Başarılı kurulanlar, başarısız olanlar)
    """
    # Eksik bağımlılıkları belirle
    packages_to_check = {}
    
    # Grupları kullan
    if groups is not None:
        for group in groups:
            if group in DEPENDENCY_GROUPS:
                packages_to_check.update(DEPENDENCY_GROUPS[group])
            else:
                logger.warning(f"Bilinmeyen bağımlılık grubu: {group}")
    
    # Doğrudan paket listesi kullan
    if required_packages is not None:
        packages_to_check.update(required_packages)
        
    # Eksik paketleri filtrele
    missing_packages = {}
    for package, version in packages_to_check.items():
        if not check_package_version(package, min_version=version):
            missing_packages[package] = version
            
    if not missing_packages:
        logger.info("Tüm bağımlılıklar zaten yüklü")
        return [], []
        
    # Eksik paketleri kur
    successful = []
    failed = []
    
    for package, version in missing_packages.items():
        if install_package(package, version=version, upgrade=upgrade):
            successful.append(package)
        else:
            failed.append(package)
            
    # Sonucu bildir
    if successful:
        logger.info(f"Başarıyla kurulan paketler: {', '.join(successful)}")
    if failed:
        logger.warning(f"Kurulamayan paketler: {', '.join(failed)}")
        
    return successful, failed


def get_missing_dependencies(
    required_packages: Dict[str, str] = None,
    groups: List[str] = None
) -> List[str]:
    """
    Eksik bağımlılıkları listeler.
    
    Args:
        required_packages: {paket_adı: min_sürüm} sözlüğü
        groups: Kontrol edilecek bağımlılık grupları
        
    Returns:
        List[str]: Eksik paket listesi
    """
    # Eksik bağımlılıkları belirle
    packages_to_check = {}
    
    # Grupları kullan
    if groups is not None:
        for group in groups:
            if group in DEPENDENCY_GROUPS:
                packages_to_check.update(DEPENDENCY_GROUPS[group])
            else:
                logger.warning(f"Bilinmeyen bağımlılık grubu: {group}")
    
    # Doğrudan paket listesi kullan
    if required_packages is not None:
        packages_to_check.update(required_packages)
        
    # Eksik paketleri filtrele
    results = check_dependencies(packages_to_check)
    missing = [pkg for pkg, installed in results.items() if not installed]
    
    return missing


def get_system_info() -> Dict[str, str]:
    """
    Sistem bilgilerini toplar.
    
    Returns:
        Dict[str, str]: Sistem bilgileri sözlüğü
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "processor": platform.processor()
    }
    
    # CPU bilgisi
    try:
        import multiprocessing
        info["cpu_count"] = str(multiprocessing.cpu_count())
    except:
        pass
        
    # RAM bilgisi
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_ram"] = f"{mem.total / (1024**3):.1f} GB"
    except:
        pass
        
    # GPU bilgisi
    try:
        import torch
        info["has_cuda"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = str(torch.cuda.device_count())
            info["gpu_name"] = torch.cuda.get_device_name(0)  # Birincil GPU
    except:
        info["has_cuda"] = "False"
        
    return info


def check_compatibility() -> Dict[str, str]:
    """
    Bağımlılıkların uyumluluğunu kontrol eder.
    
    Returns:
        Dict[str, str]: Uyumsuz paket uyarıları
    """
    # Yüklü paketleri al
    installed = dict(get_installed_packages())
    warnings = {}
    
    # Bilinen uyumsuzlukları kontrol et
    if "transformers" in installed and "tokenizers" in installed:
        transformers_ver = installed["transformers"]
        tokenizers_ver = installed["tokenizers"]
        
        # Uyumsuzluk kontrolü
        if transformers_ver.startswith("4.") and tokenizers_ver.startswith("0.12"):
            warnings["transformers"] = (
                f"transformers=={transformers_ver} ve tokenizers=={tokenizers_ver} "
                "uyumsuz olabilir. tokenizers==0.11.x sürümüne düşürmeyi deneyin."
            )
                
    # Diğer bilinen uyumsuzluklar
    # ...
        
    return warnings


def display_dependencies_status(
    required_packages: Dict[str, str] = None,
    groups: List[str] = None
):
    """
    Bağımlılık durumunu yazdırır.
    
    Args:
        required_packages: {paket_adı: min_sürüm} sözlüğü
        groups: Kontrol edilecek bağımlılık grupları
    """
    packages_to_check = {}
    
    # Grupları kullan
    if groups is not None:
        for group in groups:
            if group in DEPENDENCY_GROUPS:
                packages_to_check.update(DEPENDENCY_GROUPS[group])
                print(f"\n{group.upper()} grubu:")
            else:
                logger.warning(f"Bilinmeyen bağımlılık grubu: {group}")
    
    # Doğrudan paket listesi kullan
    if required_packages is not None:
        packages_to_check.update(required_packages)
    
    # Paketleri kontrol et ve durumlarını yazdır
    print("\nBağımlılık Durumu:")
    print("-" * 60)
    print(f"{'Paket':<20} {'Gerekli':<15} {'Yüklü':<15} {'Durum'}")
    print("-" * 60)
    
    all_installed = True
    
    # Yüklü paketleri al
    installed_packages = dict(get_installed_packages())
    
    for package, min_version in sorted(packages_to_check.items()):
        if package in installed_packages:
            installed_version = installed_packages[package]
            if min_version and compare_versions(installed_version, min_version) < 0:
                status = "Eski sürüm!"
                all_installed = False
            else:
                status = "OK"
        else:
            installed_version = "-"
            status = "Eksik!"
            all_installed = False
            
        print(f"{package:<20} {min_version:<15} {installed_version:<15} {status}")
    
    print("-" * 60)
    print("Genel Durum:", "Hazır" if all_installed else "Eksik bağımlılıklar var!")
    
    # Uyumluluk kontrolü
    compatibility_warnings = check_compatibility()
    if compatibility_warnings:
        print("\nUyumluluk Uyarıları:")
        for package, warning in compatibility_warnings.items():
            print(f"  - {package}: {warning}")


def display_system_info():
    """
    Sistem bilgilerini yazdırır.
    """
    sys_info = get_system_info()
    
    print("\nSistem Bilgileri:")
    print("-" * 40)
    print(f"  Python: {sys_info['python_version']} ({sys_info['python_implementation']})")
    print(f"  İşletim Sistemi: {sys_info['platform']}")
    
    if "cpu_count" in sys_info:
        print(f"  CPU: {sys_info['processor']} ({sys_info['cpu_count']} çekirdek)")
    if "total_ram" in sys_info:
        print(f"  RAM: {sys_info['total_ram']}")
    if "has_cuda" in sys_info and sys_info["has_cuda"] == "True":
        print(f"  GPU: {sys_info.get('gpu_name', 'Bilinmiyor')} (CUDA {sys_info.get('cuda_version', 'Bilinmiyor')})")
        if "gpu_count" in sys_info and int(sys_info["gpu_count"]) > 1:
            print(f"       {sys_info['gpu_count']} GPU mevcut")
    else:
        print("  GPU: Mevcut değil veya tespit edilemedi")