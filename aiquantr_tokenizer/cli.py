# aiquantr_tokenizer/cli.py
"""
Tokenizer hazırlama araçları için komut satırı arabirimi.

Bu modül, tokenizer'ları eğitmek, değerlendirmek ve kullanmak için
komut satırı arabirimini sağlar.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from .tokenizers import (
    BaseTokenizer,
    TokenizerTrainer,
    ByteLevelTokenizer,
    BPETokenizer,
    WordPieceTokenizer,
    UnigramTokenizer,
    SentencePieceTokenizer,
    MixedTokenizer,
    create_tokenizer_from_config,
    load_tokenizer_from_path,
    evaluate_tokenizer
)
from .utils import load_data, setup_logging, save_json, load_json

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Argüman ayrıştırıcıyı oluşturur.
    
    Returns:
        argparse.ArgumentParser: Argüman ayrıştırıcı
    """
    parser = argparse.ArgumentParser(
        description="Tokenizer hazırlama araçları",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Komut")
    
    # Eğitim komutu
    train_parser = subparsers.add_parser("train", help="Tokenizer eğit")
    train_parser.add_argument("--config", type=str, required=True, help="Yapılandırma dosyası yolu")
    train_parser.add_argument("--data", type=str, required=True, help="Eğitim veri seti yolu")
    train_parser.add_argument("--output-dir", type=str, required=True, help="Çıktı dizini")
    train_parser.add_argument("--vocab-size", type=int, help="Sözlük boyutu")
    train_parser.add_argument("--min-frequency", type=int, help="Minimum token frekansı")
    train_parser.add_argument("--limit", type=int, help="Kullanılacak maksimum örnek sayısı")
    train_parser.add_argument("--seed", type=int, default=42, help="Rastgele başlangıç değeri")
    train_parser.add_argument("--verbose", action="store_true", help="Ayrıntılı günlük kaydı")
    
    # Değerlendirme komutu
    eval_parser = subparsers.add_parser("evaluate", help="Tokenizer değerlendir")
    eval_parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer yolu")
    eval_parser.add_argument("--data", type=str, required=True, help="Değerlendirme veri seti yolu")
    eval_parser.add_argument("--output-file", type=str, help="Değerlendirme sonuçları çıktı dosyası")
    eval_parser.add_argument("--metrics", type=str, nargs="+", help="Hesaplanacak metrikler")
    eval_parser.add_argument("--limit", type=int, help="Kullanılacak maksimum örnek sayısı")
    eval_parser.add_argument("--verbose", action="store_true", help="Ayrıntılı günlük kaydı")
    
    # Kodlama komutu
    encode_parser = subparsers.add_parser("encode", help="Metni tokenize et ve kodla")
    encode_parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer yolu")
    encode_parser.add_argument("--input", type=str, required=True, help="Giriş metni veya dosyası")
    encode_parser.add_argument("--output", type=str, help="Çıktı dosyası")
    encode_parser.add_argument("--special-tokens", action="store_true", help="Özel tokenları ekle")
    encode_parser.add_argument("--verbose", action="store_true", help="Ayrıntılı günlük kaydı")
    
    # Kod çözme komutu
    decode_parser = subparsers.add_parser("decode", help="Token ID'lerinden metin oluştur")
    decode_parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer yolu")
    decode_parser.add_argument("--input", type=str, required=True, help="Token ID'leri içeren dosya")
    decode_parser.add_argument("--output", type=str, help="Çıktı dosyası")
    decode_parser.add_argument("--skip-special", action="store_true", help="Özel tokenları atla")
    decode_parser.add_argument("--verbose", action="store_true", help="Ayrıntılı günlük kaydı")
    
    # Analiz komutu
    analyze_parser = subparsers.add_parser("analyze", help="Tokenizer analiz et")
    analyze_parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer yolu")
    analyze_parser.add_argument("--output", type=str, help="Çıktı dosyası")
    analyze_parser.add_argument("--verbose", action="store_true", help="Ayrıntılı günlük kaydı")
    
    # Dönüştürme komutu
    convert_parser = subparsers.add_parser("convert", help="Tokenizer formatını dönüştür")
    convert_parser.add_argument("--input", type=str, required=True, help="Giriş tokenizer yolu")
    convert_parser.add_argument("--output", type=str, required=True, help="Çıktı tokenizer yolu")
    convert_parser.add_argument("--format", type=str, required=True, 
                              choices=["huggingface", "sentencepiece", "custom"],
                              help="Hedef format")
    convert_parser.add_argument("--verbose", action="store_true", help="Ayrıntılı günlük kaydı")
    
    return parser


def handle_train_command(args: argparse.Namespace) -> None:
    """
    Eğitim komutunu işler.
    
    Args:
        args: Komut satırı argümanları
    """
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Eğitim başlıyor")
    
    # Yapılandırmayı yükle
    config = load_json(args.config)
    
    # Komut satırı parametreleri ile yapılandırmayı güncelle
    if args.vocab_size:
        config["vocab_size"] = args.vocab_size
    if args.min_frequency:
        config["min_frequency"] = args.min_frequency
    
    # Veriyi yükle
    logger.info(f"Veri seti yükleniyor: {args.data}")
    texts = load_data(args.data, limit=args.limit)
    logger.info(f"{len(texts)} örnek yüklendi")
    
    # Tokenizer oluştur
    logger.info("Tokenizer oluşturuluyor")
    tokenizer = create_tokenizer_from_config(config)
    
    # Eğitim yöneticisi oluştur
    trainer_config = config.get("trainer", {})
    trainer = TokenizerTrainer(
        batch_size=trainer_config.get("batch_size", 1000),
        num_iterations=trainer_config.get("num_iterations"),
        show_progress=True,
        seed=args.seed
    )
    
    # Eğitim başlangıç zamanı
    start_time = time.time()
    
    # Tokenizer'ı eğit
    logger.info("Eğitim başlıyor...")
    training_results = tokenizer.train(texts, trainer=trainer)
    
    # Eğitim süresi
    training_time = time.time() - start_time
    logger.info(f"Eğitim tamamlandı ({training_time:.2f}s)")
    
    # Çıktı dizinini oluştur
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tokenizer'ı kaydet
    logger.info(f"Tokenizer kaydediliyor: {output_dir}")
    tokenizer.save_pretrained(output_dir)
    
    # Eğitim sonuçlarını kaydet
    results_file = output_dir / "training_results.json"
    save_json(training_results, results_file)
    logger.info(f"Eğitim sonuçları kaydedildi: {results_file}")
    
    logger.info("İşlem tamamlandı")


def handle_evaluate_command(args: argparse.Namespace) -> None:
    """
    Değerlendirme komutunu işler.
    
    Args:
        args: Komut satırı argümanları
    """
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Değerlendirme başlıyor")
    
    # Tokenizer'ı yükle
    logger.info(f"Tokenizer yükleniyor: {args.tokenizer}")
    tokenizer = load_tokenizer_from_path(args.tokenizer)
    
    # Veriyi yükle
    logger.info(f"Değerlendirme veri seti yükleniyor: {args.data}")
    texts = load_data(args.data, limit=args.limit)
    logger.info(f"{len(texts)} örnek yüklendi")
    
    # Değerlendirme başlangıç zamanı
    start_time = time.time()
    
    # Tokenizer'ı değerlendir
    logger.info("Değerlendirme başlıyor...")
    metrics = args.metrics if args.metrics else None
    evaluation_results = evaluate_tokenizer(tokenizer, texts, metrics)
    
    # Değerlendirme süresi
    evaluation_time = time.time() - start_time
    logger.info(f"Değerlendirme tamamlandı ({evaluation_time:.2f}s)")
    
    # Sonuçları görüntüle
    for metric, value in evaluation_results.items():
        if not isinstance(value, dict) and not isinstance(value, list):
            logger.info(f"{metric}: {value}")
    
    # Sonuçları kaydet
    if args.output_file:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(evaluation_results, output_file)
        logger.info(f"Değerlendirme sonuçları kaydedildi: {output_file}")
    
    logger.info("İşlem tamamlandı")


def handle_encode_command(args: argparse.Namespace) -> None:
    """
    Kodlama komutunu işler.
    
    Args:
        args: Komut satırı argümanları
    """
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Kodlama başlıyor")
    
    # Tokenizer'ı yükle
    logger.info(f"Tokenizer yükleniyor: {args.tokenizer}")
    tokenizer = load_tokenizer_from_path(args.tokenizer)
    
    # Giriş metnini yükle
    input_path = Path(args.input)
    if input_path.exists() and input_path.is_file():
        logger.info(f"Giriş dosyası okunuyor: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        # Doğrudan metin olarak kullan
        text = args.input
        
    logger.info(f"Metin uzunluğu: {len(text)} karakter")
    
    # Metni kodla
    logger.info("Metin kodlanıyor...")
    token_ids = tokenizer.encode(text, add_special_tokens=args.special_tokens)
    
    # Tokenize edilmiş metni görüntüle
    tokens = tokenizer.tokenize(text)
    logger.info(f"Tokenize edilmiş metin: {tokens[:10]}... (toplam: {len(tokens)})")
    logger.info(f"Token ID'leri: {token_ids[:10]}... (toplam: {len(token_ids)})")
    
    # Sonuçları kaydet
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(token_ids, f)
            
        logger.info(f"Token ID'leri kaydedildi: {output_path}")
    
    logger.info("İşlem tamamlandı")


def handle_decode_command(args: argparse.Namespace) -> None:
    """
    Kod çözme komutunu işler.
    
    Args:
        args: Komut satırı argümanları
    """
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Kod çözme başlıyor")
    
    # Tokenizer'ı yükle
    logger.info(f"Tokenizer yükleniyor: {args.tokenizer}")
    tokenizer = load_tokenizer_from_path(args.tokenizer)
    
    # Token ID'lerini yükle
    input_path = Path(args.input)
    logger.info(f"Token ID'leri yükleniyor: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        token_ids = json.load(f)
        
    logger.info(f"Yüklenen token ID'leri: {token_ids[:10]}... (toplam: {len(token_ids)})")
    
    # Token ID'lerini çöz
    logger.info("Token ID'leri çözülüyor...")
    text = tokenizer.decode(token_ids, skip_special_tokens=args.skip_special)
    
    logger.info(f"Çözülen metin: {text[:100]}... (toplam: {len(text)} karakter)")
    
    # Sonuçları kaydet
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        logger.info(f"Çözülen metin kaydedildi: {output_path}")
    
    logger.info("İşlem tamamlandı")


def handle_analyze_command(args: argparse.Namespace) -> None:
    """
    Analiz komutunu işler.
    
    Args:
        args: Komut satırı argümanları
    """
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Tokenizer analizi başlıyor")
    
    # Tokenizer'ı yükle
    logger.info(f"Tokenizer yükleniyor: {args.tokenizer}")
    tokenizer = load_tokenizer_from_path(args.tokenizer)
    
    # Tokenizer sınıfı ve istatistikleri
    logger.info(f"Tokenizer türü: {tokenizer.__class__.__name__}")
    logger.info(f"Sözlük boyutu: {tokenizer.get_vocab_size()}")
    
    # Tokenizer istatistiklerini al
    stats = tokenizer.get_statistics()
    
    # İstatistikleri görüntüle
    logger.info("Tokenizer istatistikleri:")
    for key, value in stats.items():
        if not isinstance(value, dict) and not isinstance(value, list):
            logger.info(f"  {key}: {value}")
    
    # Özel tokenları görüntüle
    logger.info("Özel tokenlar:")
    for token_type, token in tokenizer.special_tokens.items():
        logger.info(f"  {token_type}: {token}")
    
    # En yaygın tokenları görüntüle
    vocab = tokenizer.get_vocab()
    logger.info("Sözlük örneği (ilk 10 token):")
    for i, (token, id_) in enumerate(list(vocab.items())[:10]):
        logger.info(f"  {token}: {id_}")
    
    # Analiz sonuçlarını kaydet
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        analysis_results = {
            "tokenizer_type": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.get_vocab_size(),
            "statistics": stats,
            "special_tokens": tokenizer.special_tokens,
            "metadata": tokenizer.metadata
        }
        
        save_json(analysis_results, output_path)
        logger.info(f"Analiz sonuçları kaydedildi: {output_path}")
    
    logger.info("İşlem tamamlandı")


def handle_convert_command(args: argparse.Namespace) -> None:
    """
    Dönüştürme komutunu işler.
    
    Args:
        args: Komut satırı argümanları
    """
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Tokenizer dönüştürme başlıyor")
    
    # Tokenizer'ı yükle
    logger.info(f"Kaynak tokenizer yükleniyor: {args.input}")
    tokenizer = load_tokenizer_from_path(args.input)
    
    # Hedef formatı belirle
    target_format = args.format
    logger.info(f"Hedef format: {target_format}")
    
    # Dönüştürme işlemini gerçekleştir
    output_path = Path(args.output)
    
    if target_format == "huggingface":
        try:
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            logger.error("HuggingFace dönüşümü için 'transformers' paketi gereklidir.")
            return
            
        # Sözlük ve özel token bilgilerini çıkar
        vocab = tokenizer.get_vocab()
        special_tokens_map = {k: v for k, v in tokenizer.special_tokens.items()}
        
        # JSON formatında sözlük dosyası oluştur
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_file = output_path / "vocab.json"
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
        # HuggingFace tokenizer'ını oluştur
        hf_tokenizer = PreTrainedTokenizerFast(
            vocab_file=str(vocab_file),
            tokenizer_class=tokenizer.__class__.__name__,
            unk_token=special_tokens_map.get("unk_token"),
            bos_token=special_tokens_map.get("bos_token"),
            eos_token=special_tokens_map.get("eos_token"),
            pad_token=special_tokens_map.get("pad_token")
        )
        
        # Kaydet
        hf_tokenizer.save_pretrained(output_path)
        logger.info(f"HuggingFace tokenizer formatına dönüştürüldü: {output_path}")
        
    elif target_format == "sentencepiece":
        if not isinstance(tokenizer, SentencePieceTokenizer):
            # SentencePiece tokenizer'a dönüştür
            vocab = tokenizer.get_vocab()
            
            # Geçici bir eğitim dosyası oluştur
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
                temp_file = f.name
                for token in vocab.keys():
                    f.write(f"{token}\n")
                    
            # SentencePiece tokenizer oluştur ve kaydet
            sentencepiece_tokenizer = SentencePieceTokenizer(
                vocab_size=min(tokenizer.vocab_size, 30000),
                special_tokens=tokenizer.special_tokens
            )
            
            # Eğit
            sentencepiece_tokenizer.train([temp_file])
            
            # Kaydet
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sentencepiece_tokenizer.save(output_path)
            
            # Geçici dosyayı temizle
            os.unlink(temp_file)
            
            logger.info(f"SentencePiece formatına dönüştürüldü: {output_path}")
        else:
            # Zaten SentencePiece tokenizer, doğrudan kaydet
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(output_path)
            logger.info(f"SentencePiece tokenizer kaydedildi: {output_path}")
            
    elif target_format == "custom":
        # Özel format - doğrudan kaydet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Özel formatta tokenizer kaydedildi: {output_path}")
    
    logger.info("İşlem tamamlandı")


def main() -> None:
    """
    Ana program girişi.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "train":
        handle_train_command(args)
    elif args.command == "evaluate":
        handle_evaluate_command(args)
    elif args.command == "encode":
        handle_encode_command(args)
    elif args.command == "decode":
        handle_decode_command(args)
    elif args.command == "analyze":
        handle_analyze_command(args)
    elif args.command == "convert":
        handle_convert_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()