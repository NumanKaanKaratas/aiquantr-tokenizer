# Tokenizer Prep

Tokenizer hazırlama araçları paketi. Bu paket, doğal dil işleme ve makine öğrenimi uygulamaları için çeşitli tokenizer modellerini eğitmek, değerlendirmek ve kullanmak için kapsamlı araçlar sağlar.

## Desteklenen Tokenizer Türleri

- **Byte-Level Tokenizer**: Metni byte seviyesinde tokenize eden basit ve hızlı bir tokenizer.
- **Byte-Pair Encoding (BPE) Tokenizer**: Metin verisindeki en yaygın karakter çiftlerini yinelemeli olarak birleştiren alt kelime tokenizasyon algoritması.
- **WordPiece Tokenizer**: BERT gibi modellerde kullanılan kelime parçalama algoritması.
- **Unigram Tokenizer**: SentencePiece'in Unigram modelini temel alan istatistiksel tokenizer.
- **SentencePiece**: Google SentencePiece kütüphanesi için sarmalayıcı.
- **Mixed (Karma) Tokenizer**: Birden fazla tokenizer'ı bir araya getiren meta tokenizer.

## Kurulum

```bash
pip install aiquantr_tokenizer

# Ek bağımlılıklar için:

pip install aiquantr_tokenizer[sentencepiece]  # SentencePiece desteği
pip install aiquantr_tokenizer[huggingface]    # Hugging Face dönüşümleri
pip install aiquantr_tokenizer[full]           # Tüm ek bağımlılıklar


Hızlı Başlangıç
Komut Satırı Kullanımı
Tokenizer Eğitimi:

bash
tokenizer-prep train --config config.json --data texts.txt --output-dir ./my_tokenizer
Tokenizer Değerlendirmesi:

bash
tokenizer-prep evaluate --tokenizer ./my_tokenizer --data test_texts.txt --output-file eval_results.json
Metin Kodlama:

bash
tokenizer-prep encode --tokenizer ./my_tokenizer --input "Merhaba dünya!" --output encoded.json
Token ID'lerinden Metin Çözme:

bash
tokenizer-prep decode --tokenizer ./my_tokenizer --input encoded.json --output decoded.txt
Python API Kullanımı
Python
from aiquantr_tokenizer import BPETokenizer, load_data, evaluate_tokenizer

# Veri yükleme
texts = load_data("texts.txt")

# Tokenizer oluşturma ve eğitme
tokenizer = BPETokenizer(
    vocab_size=10000,
    min_frequency=2,
    special_tokens={"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}
)
tokenizer.train(texts)

# Tokenizer'ı değerlendirme
results = evaluate_tokenizer(tokenizer, texts[:1000])
print(f"Compression ratio: {results['compression_ratio']['compression_ratio']:.2f}")

# Metin kodlama ve çözme
text = "Merhaba dünya!"
token_ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(token_ids)
print(f"Original: {text}")
print(f"Token IDs: {token_ids}")
print(f"Decoded: {decoded_text}")

# Tokenizer'ı kaydetme ve yükleme
tokenizer.save("./my_tokenizer")

from aiquantr_tokenizer import load_tokenizer_from_path
loaded_tokenizer = load_tokenizer_from_path("./my_tokenizer")