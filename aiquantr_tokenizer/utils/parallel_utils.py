# aiquantr_tokenizer/utils/parallel_utils.py
"""
Paralel işleme için yardımcı fonksiyonlar.

Bu modül, çok çekirdekli işlemcilerde paralel iş yürütme 
için gelişmiş fonksiyonlar sağlar.
"""

import os
import time
import logging
import threading
import multiprocessing
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, TypeVar, Generic, Any, Dict, Iterable, Set, Optional, Union, Tuple, Iterator

# Jenerik tip tanımlamaları için
T = TypeVar('T')  # Girdi nesnesi tipi
R = TypeVar('R')  # Sonuç nesnesi tipi

# Logger oluştur
logger = logging.getLogger(__name__)


def get_optimal_worker_count(
    scale_factor: float = 0.75, 
    min_workers: int = 1, 
    max_workers: Optional[int] = None
) -> int:
    """
    Mevcut sistem için optimal iş parçacığı sayısını belirler.
    
    Args:
        scale_factor: CPU sayısı için çarpan (varsayılan: 0.75)
        min_workers: Minimum iş parçacığı sayısı (varsayılan: 1)
        max_workers: Maksimum iş parçacığı sayısı (varsayılan: None)
        
    Returns:
        int: Optimal iş parçacığı sayısı
    """
    cpu_count = os.cpu_count() or 4
    worker_count = max(min_workers, int(cpu_count * scale_factor))
    
    if max_workers is not None:
        worker_count = min(worker_count, max_workers)
        
    return worker_count


def process_in_parallel(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    chunksize: int = 1,
    timeout: Optional[float] = None,
    show_progress: bool = False,
    progress_interval: int = 100,
    desc: str = "Processing"
) -> List[R]:
    """
    Bir fonksiyonu bir liste öğeleri üzerinde paralel olarak çalıştırır.
    
    Args:
        func: Çağrılacak fonksiyon, bir öğe almalı ve bir sonuç döndürmeli
        items: İşlenecek öğeler
        max_workers: Maksimum iş parçacığı/süreç sayısı (varsayılan: None - optimal sayı kullanılır)
        use_processes: İş parçacıkları yerine süreçler kullanılsın mı (varsayılan: False)
        chunksize: Süreç havuzları için her görevde işlenecek öğe sayısı (varsayılan: 1)
        timeout: Saniye cinsinden zaman aşımı (varsayılan: None - zaman aşımı yok)
        show_progress: İlerleme durumu görüntülensin mi (varsayılan: False)
        progress_interval: İlerleme güncellemesi aralığı (varsayılan: 100 öğe)
        desc: İlerleme açıklaması (varsayılan: "Processing")
        
    Returns:
        List[R]: Sonuçlar listesi, giriş sırasıyla
        
    Raises:
        TimeoutError: İşlem zaman aşımına uğrarsa
        Exception: İşlem sırasında herhangi bir hata oluşursa
    """
    # Optimal iş parçacığı sayısını belirleme
    if max_workers is None:
        max_workers = get_optimal_worker_count()
    
    # Öğeleri listeye dönüştür
    items_list = list(items)
    total_items = len(items_list)
    
    if total_items == 0:
        logger.warning("İşlenecek öğe yok, boş liste döndürülüyor")
        return []
    
    # Sadece bir öğe varsa, paralel işleme yapmadan işle
    if total_items == 1:
        return [func(items_list[0])]
    
    # İlerleme izleme için değişkenler
    if show_progress:
        processed_items = 0
        start_time = time.time()
        last_progress_time = start_time
        mutex = threading.Lock()
        
        def progress_callback(future):
            nonlocal processed_items, last_progress_time
            with mutex:
                processed_items += 1
                current_time = time.time()
                
                # Her progress_interval öğede bir ilerleme durumunu göster
                if (processed_items % progress_interval == 0 or 
                    processed_items == total_items or 
                    current_time - last_progress_time >= 5):  # En az 5 saniyede bir güncelle
                    
                    elapsed = current_time - start_time
                    items_per_sec = processed_items / elapsed if elapsed > 0 else 0
                    eta = (total_items - processed_items) / items_per_sec if items_per_sec > 0 else 0
                    
                    logger.info(f"{desc}: {processed_items}/{total_items} öğe işlendi "
                                f"({processed_items/total_items*100:.1f}%) - "
                                f"{items_per_sec:.1f} öğe/sn - "
                                f"ETA: {eta:.1f} sn")
                    
                    last_progress_time = current_time
    
    # Executor sınıfını seç
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    results = []
    
    try:
        with executor_class(max_workers=max_workers) as executor:
            # Görevleri başlat
            if use_processes:
                future_to_idx = {
                    executor.submit(func, item): i 
                    for i, item in enumerate(items_list)
                }
            else:
                future_to_idx = {
                    executor.submit(func, item): i 
                    for i, item in enumerate(items_list)
                }
            
            # İlerleme izleme için future'lara callback ekle
            if show_progress:
                for future in future_to_idx:
                    future.add_done_callback(progress_callback)
            
            # Sonuçları doğru sırada alabilmek için boş bir liste oluştur
            results = [None] * total_items
            
            # Tamamlanan görevleri izle ve sonuçları topla
            for future in as_completed(future_to_idx, timeout=timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Öğe {idx} işlenirken hata: {e}")
                    raise
    
    except TimeoutError:
        logger.error(f"İşlem zaman aşımına uğradı ({timeout} saniye)")
        raise
    except Exception as e:
        logger.error(f"Paralel işleme sırasında hata: {e}")
        raise
        
    return results


class ParallelTaskProcessor(Generic[T, R]):
    """
    Bir dizi görevi paralel olarak işleyen sınıf.
    
    Bu sınıf, büyük veri kümeleri üzerinde bellek verimli paralel 
    işleme için akış tabanlı bir yaklaşım sağlar.
    """
    
    def __init__(
        self,
        worker_func: Callable[[T], R],
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        queue_size: int = 1000,
        show_progress: bool = False,
        progress_interval: int = 100,
        desc: str = "Processing",
        error_handling: str = "raise"  # Options: "raise", "skip", "return"
    ):
        """
        ParallelTaskProcessor sınıfı başlatıcısı.
        
        Args:
            worker_func: Her öğe için çalıştırılacak işlev
            max_workers: Maksimum paralel iş parçacığı/süreç sayısı
            use_processes: İş parçacıkları yerine süreçler kullanılsın mı
            queue_size: Girdi kuyruğu boyutu
            show_progress: İlerleme durumu görüntülensin mi
            progress_interval: İlerleme güncelleme aralığı
            desc: İlerleme açıklaması
            error_handling: Hata işleme stratejisi ("raise", "skip", "return")
        """
        self.worker_func = worker_func
        self.max_workers = max_workers if max_workers is not None else get_optimal_worker_count()
        self.use_processes = use_processes
        self.queue_size = queue_size
        self.show_progress = show_progress
        self.progress_interval = progress_interval
        self.desc = desc
        self.error_handling = error_handling
        
        self._input_queue = None
        self._output_queue = None
        self._stop_event = None
        self._workers = []
        self._producer_thread = None
        self._processed_count = 0
        self._failed_count = 0
        self._total_count = 0
        self._start_time = 0
        
    def process(self, items: Iterable[T]) -> Iterator[Union[R, Tuple[int, Exception]]]:
        """
        Öğeleri paralel olarak işler ve sonuçları döndürür.
        
        Bu, bir generator fonksiyonudur ve sonuçları kullanılabilir hale
        geldikçe verimli bir şekilde döndürür.
        
        Args:
            items: İşlenecek öğeler
            
        Yields:
            Union[R, Tuple[int, Exception]]: 
                Başarılı sonuçlar veya (index, hata) çiftleri
        """
        # İşlemci havuzu kullanıyorsak ve Python'un ana yorumlayıcı kilidi varsa,
        # input/output kuyrukları için multiprocessing yöntemlerini kullanmalıyız
        if self.use_processes:
            self._input_queue = multiprocessing.Queue(maxsize=self.queue_size)
            self._output_queue = multiprocessing.Queue()
            self._stop_event = multiprocessing.Event()
        else:
            self._input_queue = Queue(maxsize=self.queue_size)
            self._output_queue = Queue()
            self._stop_event = threading.Event()
        
        # İstatistikleri sıfırla
        self._processed_count = 0
        self._failed_count = 0
        self._total_count = 0
        self._start_time = time.time()
        
        # Üretici iş parçacığını başlat
        self._producer_thread = threading.Thread(
            target=self._producer_task,
            args=(items,),
            daemon=True
        )
        self._producer_thread.start()
        
        # Çalışanları başlat
        self._workers = []
        for _ in range(self.max_workers):
            if self.use_processes:
                worker = multiprocessing.Process(
                    target=self._worker_task,
                    daemon=True
                )
            else:
                worker = threading.Thread(
                    target=self._worker_task,
                    daemon=True
                )
            worker.start()
            self._workers.append(worker)
        
        try:
            # Öğelerin işlendiği sırada sonuçları yield et
            completed_count = 0
            while completed_count < self._total_count or not self._total_count:
                try:
                    # Bekleyen sonuçları kontrol et (1 saniye zaman aşımı ile)
                    item_idx, result_or_error, is_error = self._output_queue.get(timeout=1.0)
                    
                    if is_error:
                        self._failed_count += 1
                        if self.error_handling == "raise":
                            self._stop_event.set()
                            raise result_or_error
                        elif self.error_handling == "return":
                            yield (item_idx, result_or_error)
                    else:
                        yield result_or_error
                    
                    completed_count += 1
                    
                    # İlerleme güncellemesi
                    if self.show_progress:
                        self._update_progress(completed_count)
                        
                except Empty:
                    # Kuyruk boşsa ve bütün görevler tamamlanmış veya durmuş ise
                    if not self._producer_thread.is_alive() and self._input_queue.empty():
                        if self._all_workers_done():
                            break
                
                # Durdurma olayı tetiklenmiş mi kontrol et
                if self._stop_event.is_set():
                    break
                    
        finally:
            # Temizlik işlemleri
            self._stop_event.set()
            
            # Üreticiyi bekle
            if self._producer_thread.is_alive():
                self._producer_thread.join(timeout=2.0)
            
            # Çalışanları bekle
            for worker in self._workers:
                if hasattr(worker, 'terminate'):  # ProcessPoolExecutor için
                    worker.terminate()
                if worker.is_alive():
                    worker.join(timeout=2.0)
                    
            # Kuyrukları temizle
            self._clear_queues()
            
            # Son ilerleme güncellemesi
            if self.show_progress:
                self._update_progress(completed_count, final=True)
    
    def _producer_task(self, items: Iterable[T]) -> None:
        """
        Öğeleri giriş kuyruğuna yerleştiren görev.
        
        Args:
            items: İşlenecek öğeler
        """
        try:
            for i, item in enumerate(items):
                if self._stop_event.is_set():
                    break
                    
                self._total_count = i + 1
                self._input_queue.put((i, item))
                
            # Sentinel değerleri ekle (her çalışan için bir tane)
            for _ in range(self.max_workers):
                self._input_queue.put((None, None))
                
        except Exception as e:
            logger.error(f"Üretici görevde hata: {e}")
            self._stop_event.set()
                
    def _worker_task(self) -> None:
        """Her çalışan tarafından çalıştırılan görev."""
        while not self._stop_event.is_set():
            try:
                # Giriş kuyruğundan bir öğe al
                idx, item = self._input_queue.get(timeout=0.5)
                
                # Sentinel değeri kontrol et
                if idx is None:
                    break
                    
                # Öğeyi işle
                try:
                    result = self.worker_func(item)
                    self._output_queue.put((idx, result, False))
                except Exception as e:
                    self._output_queue.put((idx, e, True))
                    if self.error_handling == "raise":
                        self._stop_event.set()
                        
            except Empty:
                pass
            except Exception as e:
                logger.error(f"Çalışan görevde beklenmeyen hata: {e}")
                if self.error_handling == "raise":
                    self._stop_event.set()
    
    def _update_progress(self, completed_count: int, final: bool = False) -> None:
        """
        İlerleme durumunu güncelleyen yardımcı fonksiyon.
        
        Args:
            completed_count: Tamamlanan öğe sayısı
            final: Son güncelleme mi
        """
        if not self.show_progress:
            return
            
        if completed_count % self.progress_interval == 0 or final:
            elapsed = time.time() - self._start_time
            items_per_sec = completed_count / elapsed if elapsed > 0 else 0
            
            if self._total_count > 0:
                eta = (self._total_count - completed_count) / items_per_sec if items_per_sec > 0 else 0
                percentage = completed_count / self._total_count * 100
                
                logger.info(
                    f"{self.desc}: {completed_count}/{self._total_count} öğe işlendi "
                    f"({percentage:.1f}%) - Hata: {self._failed_count} - "
                    f"{items_per_sec:.1f} öğe/sn - "
                    f"ETA: {eta:.1f} sn"
                )
            else:
                # Toplam sayı bilinmiyorsa
                logger.info(
                    f"{self.desc}: {completed_count} öğe işlendi - "
                    f"Hata: {self._failed_count} - "
                    f"{items_per_sec:.1f} öğe/sn"
                )
    
    def _all_workers_done(self) -> bool:
        """
        Bütün çalışanların tamamlanıp tamamlanmadığını kontrol eder.
        
        Returns:
            bool: Bütün çalışanlar tamamlanmış mı
        """
        for worker in self._workers:
            if worker.is_alive():
                return False
        return True
    
    def _clear_queues(self) -> None:
        """Giriş ve çıkış kuyruklarını temizler."""
        try:
            # Giriş kuyruğunu temizle
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except Empty:
                    break
                    
            # Çıkış kuyruğunu temizle
            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    break
        except:
            pass


class BatchProcessor:
    """
    Büyük veri kümeleri için toplu işleme yapan sınıf.
    
    Bu sınıf, bellek sınırlamaları olan ortamlarda büyük veri kümelerini
    etkili bir şekilde işlemek için tasarlanmıştır.
    """
    
    def __init__(
        self,
        process_batch_func: Callable[[List[T]], List[R]],
        batch_size: int = 1000,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        show_progress: bool = False,
        desc: str = "Processing batches"
    ):
        """
        BatchProcessor sınıfı başlatıcısı.
        
        Args:
            process_batch_func: Bir öğe listesi alan ve sonuç listesi döndüren işlev
            batch_size: Her toplu işlemdeki öğe sayısı
            max_workers: Maksimum paralel iş parçacığı/süreç sayısı
            use_processes: İş parçacıkları yerine süreçler kullanılsın mı
            show_progress: İlerleme durumu görüntülensin mi
            desc: İlerleme açıklaması
        """
        self.process_batch_func = process_batch_func
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.show_progress = show_progress
        self.desc = desc
    
    def process(self, items: Iterable[T]) -> List[R]:
        """
        Öğeleri toplu olarak işler ve sonuçları birleştirir.
        
        Args:
            items: İşlenecek öğeler
            
        Returns:
            List[R]: Tüm toplu işlemlerden birleştirilmiş sonuçlar
        """
        # Öğeleri toplu işlemlere böl
        batches = []
        current_batch = []
        
        for item in items:
            current_batch.append(item)
            
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Son toplu işlemi ekle (eğer boş değilse)
        if current_batch:
            batches.append(current_batch)
        
        if not batches:
            logger.warning("İşlenecek öğe yok, boş liste döndürülüyor")
            return []
        
        # Her toplu işlemi paralel olarak işle
        batch_results = process_in_parallel(
            func=self.process_batch_func,
            items=batches,
            max_workers=self.max_workers,
            use_processes=self.use_processes,
            show_progress=self.show_progress,
            desc=self.desc
        )
        
        # Sonuçları bir listede birleştir
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
            
        return all_results


def parallel_map(
    func: Callable[[T], R], 
    items: Iterable[T], 
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    ordered: bool = True,
    buffer_size: int = 1000,
    show_progress: bool = False
) -> Iterator[R]:
    """
    Öğelerin üzerinde bir işlevi paralel olarak eşler ve sonuçları verimli bir şekilde döndürür.
    
    Bu fonksiyon, bellek verimliliği için bir akış yaklaşımı kullanarak Python'un yerleşik map()
    fonksiyonunun paralel versiyonudur.
    
    Args:
        func: Eşlenecek fonksiyon
        items: İşlenecek öğeler
        max_workers: Maksimum paralel iş parçacığı/süreç sayısı
        use_processes: İş parçacıkları yerine süreçler kullanılsın mı
        ordered: Sonuçlar giriş sırasıyla döndürülsün mü
        buffer_size: Tampon boyutu
        show_progress: İlerleme durumu görüntülensin mi
        
    Yields:
        R: İşlenen sonuçlar
    """
    if ordered:
        # Sıralı sonuçlar için, ParallelTaskProcessor sınıfını kullan
        processor = ParallelTaskProcessor(
            worker_func=func,
            max_workers=max_workers,
            use_processes=use_processes,
            queue_size=buffer_size,
            show_progress=show_progress,
            desc="Parallel mapping"
        )
        
        yield from processor.process(items)
    else:
        # Sırasız sonuçlar için, ThreadPoolExecutor/ProcessPoolExecutor kullan
        if max_workers is None:
            max_workers = get_optimal_worker_count()
            
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            # Metrikler için değişkenler
            processed = 0
            total_seen = 0
            start_time = time.time()
            
            # Öğeleri işleme
            futures = {}
            for item in items:
                future = executor.submit(func, item)
                futures[future] = None
                total_seen += 1
                
                # İşlenen sonuçları kontrol et
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        try:
                            result = future.result()
                            yield result
                            processed += 1
                            done_futures.append(future)
                            
                            # İlerleme güncellemesi
                            if show_progress and processed % 100 == 0:
                                elapsed = time.time() - start_time
                                items_per_sec = processed / elapsed if elapsed > 0 else 0
                                logger.info(f"İşlendi: {processed} - "
                                           f"{items_per_sec:.1f} öğe/sn")
                        except Exception as e:
                            logger.error(f"Öğe işlenirken hata: {e}")
                            done_futures.append(future)
                            raise
                
                # Tamamlanmış görevleri kaldır
                for future in done_futures:
                    futures.pop(future)
            
            # Kalan sonuçları işle
            for future in as_completed(futures):
                try:
                    result = future.result()
                    yield result
                    processed += 1
                    
                    # İlerleme güncellemesi
                    if show_progress and processed % 100 == 0:
                        elapsed = time.time() - start_time
                        items_per_sec = processed / elapsed if elapsed > 0 else 0
                        logger.info(f"İşlendi: {processed}/{total_seen} - "
                                   f"{items_per_sec:.1f} öğe/sn")
                except Exception as e:
                    logger.error(f"Öğe işlenirken hata: {e}")
                    raise
            
            # Son ilerleme güncellemesi
            if show_progress:
                elapsed = time.time() - start_time
                items_per_sec = processed / elapsed if elapsed > 0 else 0
                logger.info(f"Tamamlandı: {processed}/{total_seen} öğe işlendi - "
                           f"{items_per_sec:.1f} öğe/sn")