# aiquantr_tokenizer/utils/hub_utils.py
"""
Hugging Face Hub ile etkileşim için yardımcı işlevler.

Bu modül, modelleri ve tokenizer'ları Hugging Face Hub'a
yükleme ve indirme için işlevler sağlar.
"""

import os
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

try:
    from huggingface_hub import HfApi, Repository, create_repo, upload_folder, login
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Logger oluştur
logger = logging.getLogger(__name__)


def check_hub_available() -> bool:
    """
    Hugging Face Hub modüllerinin kullanılabilir olup olmadığını kontrol eder.
    
    Returns:
        bool: Hub erişilebilir mi?
    """
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub paketi yüklü değil. Hub işlemleri kullanılamaz.")
        return False
    return True


def login_to_hub(token: Optional[str] = None) -> bool:
    """
    Hugging Face Hub'a giriş yapar.
    
    Args:
        token: Hugging Face API token'ı (varsayılan: None - ortam değişkeninden alır)
        
    Returns:
        bool: Giriş başarılı mı?
    """
    if not check_hub_available():
        return False
    
    try:
        # Token yoksa ortam değişkeninden almaya çalış
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            
        if not token:
            logger.error("Hub'a giriş yapılamadı: Token bulunamadı")
            return False
        
        login(token=token)
        logger.info("Hugging Face Hub'a giriş yapıldı")
        return True
        
    except Exception as e:
        logger.error(f"Hub'a giriş yapılamadı: {e}")
        return False


def upload_to_hub(
    local_path: Union[str, Path],
    repo_id: str,
    token: Optional[str] = None,
    commit_message: str = "Model yüklendi",
    private: bool = False,
    readme: Optional[str] = None
) -> Optional[str]:
    """
    Dosyaları ve dizinleri Hugging Face Hub'a yükler.
    
    Args:
        local_path: Yüklenecek dosya veya dizin yolu
        repo_id: Hub'daki repo ID'si (kullanıcı/model_adı)
        token: Hugging Face API token'ı (varsayılan: None - ortam değişkeninden alır)
        commit_message: Commit mesajı (varsayılan: "Model yüklendi")
        private: Özel repo oluştur (varsayılan: False)
        readme: README içeriği (varsayılan: None)
        
    Returns:
        Optional[str]: Başarılıysa repo URL'si, aksi halde None
    """
    if not check_hub_available():
        return None
    
    try:
        # Token kontrolü
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            
        if not token:
            logger.error("Hub'a yükleme yapılamadı: Token bulunamadı")
            return None
            
        local_path = Path(local_path)
        
        # Dosya ve dizin ayrımı
        if local_path.is_file():
            # Tek dosya için geçici dizin oluştur
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                dest_path = tmp_path / local_path.name
                shutil.copy2(local_path, dest_path)
                
                # README ekle
                if readme:
                    readme_path = tmp_path / "README.md"
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(readme)
                
                # Repo oluştur ve yükle
                api = HfApi()
                api.create_repo(repo_id, token=token, private=private, exist_ok=True)
                
                # Klasörü yükle
                uploaded_url = api.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(tmp_path),
                    commit_message=commit_message,
                    token=token
                )
                
                logger.info(f"Dosya başarıyla yüklendi: {uploaded_url}")
                return uploaded_url
                
        else:
            # Dizin için doğrudan yükle
            tmp_path = local_path
            
            # README ekle (yoksa)
            if readme and not (local_path / "README.md").exists():
                readme_path = local_path / "README.md"
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme)
            
            # Repo oluştur ve yükle
            api = HfApi()
            api.create_repo(repo_id, token=token, private=private, exist_ok=True)
            
            # Klasörü yükle
            uploaded_url = api.upload_folder(
                repo_id=repo_id,
                folder_path=str(tmp_path),
                commit_message=commit_message,
                token=token
            )
            
            logger.info(f"Dizin başarıyla yüklendi: {uploaded_url}")
            return uploaded_url
            
    except Exception as e:
        logger.error(f"Hub'a yükleme yapılamadı: {e}")
        return None


def download_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    revision: str = "main",
    token: Optional[str] = None,
    subfolder: str = ""
) -> Optional[Path]:
    """
    Hugging Face Hub'dan dosyaları indirir.
    
    Args:
        repo_id: Hub'daki repo ID'si (kullanıcı/model_adı)
        local_dir: İndirilen dosyaların kaydedileceği dizin
        revision: Repo revizyonu (branch/tag/commit) (varsayılan: "main")
        token: Hugging Face API token'ı (varsayılan: None - ortam değişkeninden alır)
        subfolder: İndirilecek alt dizin (varsayılan: "")
        
    Returns:
        Optional[Path]: Başarılıysa indirilen dizin yolu, aksi halde None
    """
    if not check_hub_available():
        return None
    
    try:
        # Token kontrolü
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Repository sınıfını kullanarak klonla
        repo = Repository(
            local_dir=str(local_dir),
            clone_from=repo_id,
            revision=revision,
            use_auth_token=token,
        )
        repo.git_pull()
        
        # Alt dizin belirtilmişse
        if subfolder:
            subfolder_path = local_dir / subfolder
            if not subfolder_path.exists():
                logger.error(f"Alt dizin bulunamadı: {subfolder}")
                return None
            return subfolder_path
            
        logger.info(f"Repo başarıyla indirildi: {repo_id} -> {local_dir}")
        return local_dir
        
    except RepositoryNotFoundError:
        logger.error(f"Repo bulunamadı: {repo_id}")
        return None
    except RevisionNotFoundError:
        logger.error(f"Revizyon bulunamadı: {revision}")
        return None
    except Exception as e:
        logger.error(f"Hub'dan indirme yapılamadı: {e}")
        return None


def push_to_hub(
    model_or_tokenizer: Any,
    repo_id: str,
    token: Optional[str] = None,
    commit_message: str = "Model yüklendi",
    private: bool = False
) -> Optional[str]:
    """
    Hugging Face model veya tokenizer'ını doğrudan Hub'a yükler.
    
    Args:
        model_or_tokenizer: Hugging Face model veya tokenizer nesnesi
        repo_id: Hub'daki repo ID'si (kullanıcı/model_adı)
        token: Hugging Face API token'ı (varsayılan: None - ortam değişkeninden alır)
        commit_message: Commit mesajı (varsayılan: "Model yüklendi")
        private: Özel repo oluştur (varsayılan: False)
        
    Returns:
        Optional[str]: Başarılıysa repo URL'si, aksi halde None
    """
    if not check_hub_available():
        return None
    
    try:
        # Token kontrolü
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            
        if not token:
            logger.error("Hub'a yükleme yapılamadı: Token bulunamadı")
            return None
        
        # push_to_hub() metodunu kullan
        if hasattr(model_or_tokenizer, "push_to_hub"):
            repo_url = model_or_tokenizer.push_to_hub(
                repo_id=repo_id,
                use_auth_token=token,
                commit_message=commit_message,
                private=private
            )
            logger.info(f"Model/tokenizer başarıyla yüklendi: {repo_url}")
            return repo_url
        else:
            logger.error("Gönderilen nesnenin push_to_hub metodu yok")
            return None
            
    except Exception as e:
        logger.error(f"Hub'a yükleme yapılamadı: {e}")
        return None


def list_hub_models(
    filter_str: str = "",
    author: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Hugging Face Hub'daki modelleri listeler.
    
    Args:
        filter_str: Filtreleme dizgisi (varsayılan: "")
        author: Belirli bir yazara göre filtrele (varsayılan: None)
        token: Hugging Face API token'ı (varsayılan: None - ortam değişkeninden alır)
        limit: Sonuç sayısı limiti (varsayılan: 100)
        
    Returns:
        List[Dict[str, Any]]: Model bilgileri listesi
    """
    if not check_hub_available():
        return []
    
    try:
        api = HfApi()
        
        # Token kontrolü
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # Modelleri listele
        models = api.list_models(
            filter=filter_str,
            author=author,
            use_auth_token=token,
            limit=limit
        )
        
        # API sonucunu daha kullanışlı bir formata dönüştür
        results = []
        for model in models:
            results.append({
                "id": model.modelId,
                "author": model.author,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "last_modified": model.lastModified,
                "private": model.private
            })
            
        return results
        
    except Exception as e:
        logger.error(f"Hub modelleri listelenirken hata: {e}")
        return []