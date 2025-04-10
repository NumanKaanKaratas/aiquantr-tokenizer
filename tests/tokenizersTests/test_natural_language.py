"""
Doğal dil metinleri için tokenizer test modülü.

Bu test modülü, projede bulunan tüm tokenizer tiplerini
farklı doğal dil metinleri üzerinde test eder.
"""

import unittest
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, Any, List, Type, Optional

from aiquantr_tokenizer.tokenizers.base import BaseTokenizer, TokenizerTrainer


class TestNaturalLanguage(unittest.TestCase):
    """
    Farklı doğal dil metinleri üzerinde tokenizer'ları test eder.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        # Gerekli modülleri import et
        try:
            # Tüm tokenizer sınıfları
            from aiquantr_tokenizer.tokenizers.bpe import BPETokenizer
            from aiquantr_tokenizer.tokenizers.wordpiece import WordPieceTokenizer
            from aiquantr_tokenizer.tokenizers.byte_level import ByteLevelTokenizer
            from aiquantr_tokenizer.tokenizers.unigram import UnigramTokenizer
            from aiquantr_tokenizer.tokenizers.mixed import MixedTokenizer
            from aiquantr_tokenizer.tokenizers.factory import create_tokenizer_from_config, register_tokenizer_type
            
            self.BPETokenizer = BPETokenizer
            self.WordPieceTokenizer = WordPieceTokenizer
            self.ByteLevelTokenizer = ByteLevelTokenizer
            self.UnigramTokenizer = UnigramTokenizer
            self.MixedTokenizer = MixedTokenizer
            self.create_tokenizer_from_config = create_tokenizer_from_config
            self.register_tokenizer_type = register_tokenizer_type
            
            self.all_tokenizer_classes = {
                "BPE": BPETokenizer,
                "WordPiece": WordPieceTokenizer,
                "ByteLevel": ByteLevelTokenizer, 
                "Unigram": UnigramTokenizer,
            }
            
        except ImportError as e:
            self.skipTest(f"Gerekli tokenizer modülleri bulunamadı: {e}")
        
        # Geçici dizin oluştur
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Dil örnekleri
        self.language_samples = {
            "turkish": self._get_turkish_sample(),
            "english": self._get_english_sample(),
            "mixed": self._get_mixed_language_sample(),
            "technical": self._get_technical_sample(),
            "social_media": self._get_social_media_sample()
        }
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        self.temp_dir.cleanup()
    
    def _get_turkish_sample(self):
        """
        Türkçe metin örneği oluşturur.
        """
        return """Doğal dil işleme, bilgisayar biliminin ve yapay zekanın önemli bir alt dalıdır. 
        İnsan dilinin anlaşılması ve üretilmesi amacıyla çeşitli algoritmalar ve yöntemler içerir. 
        Günümüzde, sesli asistanlar, otomatik çeviri sistemleri, metin özetleme araçları ve duygu analizi uygulamaları gibi birçok teknolojik üründe kullanılmaktadır.
        
        Tokenizasyon, doğal dil işlemede temel bir ön işleme adımıdır. Metin verilerini daha küçük birimlere ayırmayı amaçlar. 
        Bu birimler kelimeler, alt kelimeler veya karakterler olabilir. Tokenizasyon yönteminin seçimi, dilin yapısına ve çözülmeye çalışılan probleme bağlıdır.
        
        Türkçe gibi sondan eklemeli dillerde, kelime kökleri ve ekler arasındaki ilişkiyi modellemek önemlidir. 
        Örneğin, "kitaplarımızdakiler" kelimesi "kitap-lar-ımız-da-ki-ler" şeklinde morfolojik birimlerine ayrılabilir. 
        Bu tür dillerde alt kelime tabanlı tokenizasyon yöntemleri (BPE, WordPiece, Unigram vb.) daha etkili sonuçlar verebilir.
        
        Tokenizasyon algoritmaları, büyük ölçekli veri kümelerinde eğitilir ve sözlük boyutuna göre optimize edilir. 
        Sözlük boyutunun artması, daha spesifik tokenlerin öğrenilmesine olanak tanır ancak modelin boyutu ve işlem süresi de artar.
        
        Türkçe'nin morfolojik yapısı nedeniyle, etkili bir tokenizasyon stratejisi geliştirmek önemlidir. 
        Aksi takdirde, seyrek görülen kelimeler ve uzun bileşik yapılar, modelin performansını olumsuz etkileyebilir."""
    
    def _get_english_sample(self):
        """
        İngilizce metin örneği oluşturur.
        """
        return """Natural Language Processing (NLP) is a subfield of computer science and artificial intelligence concerned with the interactions between computers and human language.
        It involves enabling computers to understand, interpret, and generate human language in a valuable way.
        
        Tokenization is a fundamental preprocessing step in natural language processing. It aims to break down text data into smaller units.
        These units can be words, subwords, or characters. The choice of tokenization method depends on the structure of the language and the problem being solved.
        
        For languages like English, word-based tokenization might seem straightforward, splitting on spaces and punctuation. However, complications arise with
        contractions (e.g., "don't"), hyphenated words (e.g., "state-of-the-art"), and compound words (e.g., "database").
        
        Subword tokenization approaches such as Byte Pair Encoding (BPE), WordPiece, and Unigram have gained popularity in recent years.
        These methods balance the trade-off between vocabulary size and token frequency by learning common subword patterns from training data.
        
        BPE, for instance, starts with individual characters and iteratively merges the most frequent pairs to form new tokens.
        WordPiece is similar but uses a likelihood-based approach to determine which pairs to merge.
        Unigram uses a probabilistic model to find the most likely segmentation of words into subwords.
        
        The choice of tokenizer can significantly impact model performance, especially for tasks involving rare words or morphologically complex languages.
        Recent developments in NLP have moved towards character-level or byte-level tokenization to handle multilingual scenarios and reduce out-of-vocabulary issues."""
    
    def _get_mixed_language_sample(self):
        """
        Karışık dil içeren metin örneği oluşturur.
        """
        return """In software development, writing clean code is essential. "Temiz kod yazmak" (writing clean code in Turkish) involves several principles:
        
        1. Simplicity (Basitlik): "Keep it simple, stupid!" The code should be easy to understand.
        2. Clarity (Netlik): Variable and function names should clearly describe their purpose.
        3. DRY (Don't Repeat Yourself): "Kendini tekrarlama" - Avoid duplicating code.
        4. Testing (Test Etme): "İyi test edilmiş kod, iyi koddur" (Well-tested code is good code).
        
        Consider this Python example with Turkish comments:
        
        ```python
        # Bir listedeki sayıların ortalamasını hesaplar
        def calculate_average(numbers):
            if not numbers:
                return 0  # Boş liste için 0 döndür
            total = sum(numbers)
            return total / len(numbers)
        ```
        
        Multilingual documentation is becoming more common in global projects. Documentation in both English and the local language helps diverse development teams collaborate effectively.
        
        "Çok dilli belgelendirme, küresel projelerde daha yaygın hale geliyor. İngilizce ve yerel dildeki belgeler, farklı geliştirme ekiplerinin etkili bir şekilde işbirliği yapmasına yardımcı olur."
        
        Using the right tokenization strategy for multilingual text is crucial for NLP applications. A good tokenizer should handle different languages seamlessly, including characters like 'ş', 'ç', 'ğ', 'ü', 'ö', 'ı' in Turkish."""
    
    def _get_technical_sample(self):
        """
        Teknik içerikli metin örneği oluşturur.
        """
        return """# Transformers ve Self-Attention Mekanizması

        ## 1. Transformers Mimarisi
        
        Transformers mimarisi, 2017 yılında "Attention is All You Need" makalesiyle tanıtılmıştır. Bu mimari, tekrarlayan sinir ağları (RNN) ve evrişimli sinir ağlarının (CNN) sınırlılıklarını aşmak için tasarlanmıştır. Temel yenilikçi bileşeni, self-attention mekanizmasıdır.
        
        Mimari, encoder ve decoder olmak üzere iki ana bölümden oluşur:
        
        - **Encoder**: Giriş dizisini işler ve her token için bağlamsal temsiller oluşturur.
        - **Decoder**: Encoder çıktısını ve önceki çıktı tokenlerini kullanarak yeni tokenler üretir.
        
        ## 2. Self-Attention Mekanizması
        
        Self-attention, bir dizinin her öğesinin, dizinin diğer tüm öğeleriyle ilişkisini hesaplar. Matematiksel olarak:
        
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
        
        Burada:
        - Q (Query): Sorgu matrisi
        - K (Key): Anahtar matrisi
        - V (Value): Değer matrisi
        - d_k: Anahtarların boyutu
        
        ## 3. Çok Başlı Dikkat (Multi-Head Attention)
        
        Çok başlı dikkat, farklı temsil alt uzaylarından bilgileri yakalamak için birden fazla self-attention katmanını paralel olarak çalıştırır:
        
        $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
        
        $$\text{where head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$
        
        ## 4. Pozisyon Kodlama (Positional Encoding)
        
        Transformers, dizilerdeki sıra bilgisini kodlamak için pozisyon kodlaması kullanır:
        
        $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
        $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$
        
        ## 5. Uygulamalar
        
        Transformer mimarisi, çeşitli doğal dil işleme görevlerinde kullanılır:
        
        - Metin sınıflandırma
        - Dil çevirisi
        - Soru cevaplama
        - Metin özetleme
        - Dil modelleme (GPT, BERT, T5)
        
        ## 6. İleri Teknolojiler
        
        En son transformer tabanlı modeller şunları içerir:
        
        - BERT (Bidirectional Encoder Representations from Transformers)
        - GPT (Generative Pre-trained Transformer)
        - T5 (Text-to-Text Transfer Transformer)
        - XLNet, RoBERTa, ALBERT, ELECTRA
        
        ## 7. Verimlilik İyileştirmeleri
        
        Büyük transformerların eğitimini ve çıkarımını iyileştirmek için çeşitli teknikler geliştirilmiştir:
        
        - Sparse Attention
        - Gradient Checkpointing
        - Mixed Precision Training
        - Model Parallelism
        - Pipeline Parallelism
        """
    
    def _get_social_media_sample(self):
        """
        Sosyal medya tarzı metin örneği oluşturur.
        """
        return """Bugün #AITokenizer projemizi açık kaynak olarak paylaştık! 🎉 @TurkishNLP ekibine büyük teşekkürler! 👏

        Bu tokenizer farklı dilleri destekliyor:
        - Türkçe 🇹🇷
        - İngilizce 🇬🇧
        - Almanca 🇩🇪
        
        Projemiz hakkında daha fazla bilgiyi 👉 https://github.com/example/ai-tokenizer adresinden bulabilirsiniz.
        
        #NLP #MachineLearning #OpenSource
        
        ---
        
        @user1 hızlı bir soru - 5B parametreli modeli 4xA100 GPU'da nasıl eğitebilirim? Bellek hatası alıyorum 😫
        
        @user2 DeepSpeed ya da FSDP kullan, gradient checkpointing açmayı unutma! ben önceki projemde ZeRO-3 kullandım, GPU başına 20GB bellek tüketimim vardı.
        
        @user3 Proje için PR yolladım, inceleyebilir misin? Unicode normalization sorununu çözdüm.
        
        OMG! Yeni model 42% daha iyi çalışıyor! SOTA sonuçları geçtik! 🚀🔥
        
        ---
        
        Bu gönderide bugün için to-do listemi paylaşıyorum:
        ✅ Tokenizer eğitimi (3 saat sürdü...)
        ✅ Hata analizi
        ❌ Dokümanları güncelleme
        ❌ Test coverage artırma
        
        En sevdiğim meme: "bir tokenizer öğrendiğinde, aslında öğrendiğin dildir" 🤣
        
        RT @ML_News: 2023 yılında tokenizer'lar üzerine yayınlanan 10 önemli çalışma: [LINK]
        
        /cc @researcher1 @researcher2
        """
    
    def test_individual_tokenizers_on_languages(self):
        """
        Her tokenizer'ı farklı doğal dil örnekleri üzerinde test eder.
        """
        # Test edilecek tokenizer'lar
        tokenizer_instances = {
            "BPE": self.BPETokenizer(vocab_size=1000),
            "WordPiece": self.WordPieceTokenizer(vocab_size=1000),
            "ByteLevel": self.ByteLevelTokenizer(vocab_size=1000),
            "Unigram": self.UnigramTokenizer(vocab_size=1000)
        }
        
        # Tüm dil örneklerini birleştir ve eğitim için kullan
        all_samples = list(self.language_samples.values())
        
        # Her tokenizer'ı test et
        for tokenizer_name, tokenizer in tokenizer_instances.items():
            with self.subTest(tokenizer=tokenizer_name):
                # Tokenizer'ı eğit
                print(f"\n{tokenizer_name} tokenizer doğal dil üzerinde eğitiliyor...")
                train_result = tokenizer.train(all_samples)
                
                self.assertTrue(tokenizer.is_trained, f"{tokenizer_name} eğitimi başarısız oldu")
                self.assertGreater(tokenizer.get_vocab_size(), 0, f"{tokenizer_name} boş sözlük oluşturdu")
                
                # Farklı dil örneklerini test et
                for lang_name, text in self.language_samples.items():
                    sample_text = text[:500]  # İlk 500 karakteri test et
                    
                    # Encode ve decode işlemleri
                    encoded = tokenizer.encode(sample_text)
                    decoded = tokenizer.decode(encoded)
                    
                    # Sonuçları yazdır
                    print(f"{tokenizer_name} - {lang_name} encode sonucu: {len(encoded)} token")
                    print(f"İlk 10 token ID: {encoded[:10]}")
                    
                    # Token yoğunluğunu hesapla (token sayısı / metin uzunluğu)
                    density = len(encoded) / len(sample_text)
                    print(f"Token yoğunluğu: {density:.4f} token/karakter")
                    
                    # Benzerlik skoru hesapla
                    similarity = self._text_similarity(sample_text, decoded)
                    print(f"Decode benzerliği: {similarity:.2f}%")
                    
                    # Minimal doğrulama
                    self.assertGreater(len(encoded), 0, f"{tokenizer_name} hiç token üretmedi")
                
                # Tokenizer'ı kaydet ve yükle
                save_path = self.temp_path / f"{tokenizer_name}_nlp"
                tokenizer.save(save_path)
                
                try:
                    loaded_tokenizer = tokenizer.__class__.load(save_path)
                    self.assertEqual(
                        tokenizer.get_vocab_size(), 
                        loaded_tokenizer.get_vocab_size(), 
                        f"{tokenizer_name} yükleme sonrası sözlük boyutu değişti"
                    )
                except Exception as e:
                    print(f"{tokenizer_name} yüklenirken hata oluştu: {e}")
    
    def test_multilingual_tokenizer(self):
        """
        MixedTokenizer'ı farklı diller için test eder.
        """
        # Alt tokenizer'ları oluştur
        tr_tokenizer = self.BPETokenizer(vocab_size=500, name="TurkishTokenizer")
        en_tokenizer = self.BPETokenizer(vocab_size=500, name="EnglishTokenizer")
        tech_tokenizer = self.WordPieceTokenizer(vocab_size=300, name="TechnicalTokenizer")
        social_tokenizer = self.ByteLevelTokenizer(vocab_size=300, name="SocialTokenizer")
        
        # Örnekleri ayrı ayrı eğit
        tr_tokenizer.train([self.language_samples["turkish"]])
        en_tokenizer.train([self.language_samples["english"]])
        tech_tokenizer.train([self.language_samples["technical"]])
        social_tokenizer.train([self.language_samples["social_media"]])
        
        # MixedTokenizer oluştur
        mixed_tokenizer = self.MixedTokenizer(
            tokenizers={
                "tr": tr_tokenizer,
                "en": en_tokenizer, 
                "tech": tech_tokenizer,
                "social": social_tokenizer
            },
            default_tokenizer="tr",
            merged_vocab=True,
            name="MultilingualMixed"
        )
        
        # Router fonksiyonu tanımla
        def router(text):
            # Basit bir dil/içerik tespiti
            tr_chars = sum(1 for c in text if c in "ğüşıöçĞÜŞİÖÇ")
            hashtags = text.count("#")
            mentions = text.count("@")
            equations = sum(1 for c in text if c in "∫∑∏√∂∇∆∉∈")
            
            if hashtags > 0 or mentions > 0:
                return "social"
            elif equations > 0 or "```" in text:
                return "tech"
            elif tr_chars > 10:
                return "tr"
            else:
                return "en"
        
        mixed_tokenizer.router = router
        
        # Test et
        for lang_name, text in self.language_samples.items():
            sample_text = text[:300]  # İlk 300 karakteri test et
            
            # Encode ve decode işlemleri
            encoded = mixed_tokenizer.encode(sample_text)
            decoded = mixed_tokenizer.decode(encoded)
            
            # Sonuçları yazdır
            print(f"\nMixedTokenizer - {lang_name} encode sonucu: {len(encoded)} token")
            print(f"İlk 10 token ID: {encoded[:10]}")
            
            # Token yoğunluğunu hesapla
            density = len(encoded) / len(sample_text)
            print(f"Token yoğunluğu: {density:.4f} token/karakter")
            
            # Benzerlik skoru hesapla
            similarity = self._text_similarity(sample_text, decoded)
            print(f"Decode benzerliği: {similarity:.2f}%")
            
            # Dil tespitini kontrol et
            detected = router(sample_text)
            print(f"Tespit edilen tokenizer: {detected}")
            
            # Minimal doğrulama
            self.assertGreater(len(encoded), 0, f"MixedTokenizer {lang_name} için hiç token üretmedi")
        
        # Kaydet ve yükle
        save_path = self.temp_path / "mixed_multilingual"
        mixed_tokenizer.save(save_path)
        
        try:
            loaded_tokenizer = self.MixedTokenizer.load(save_path)
            self.assertEqual(
                mixed_tokenizer.get_vocab_size(), 
                loaded_tokenizer.get_vocab_size(), 
                "MixedTokenizer yükleme sonrası sözlük boyutu değişti"
            )
        except Exception as e:
            print(f"MixedTokenizer yüklenirken hata oluştu: {e}")
    
    def test_special_chars_and_emojis(self):
        """
        Özel karakterleri ve emojileri test eder.
        """
        # Özel karakterler ve emojiler içeren metin
        special_text = """
        Emojiler: 😀 😃 😄 😁 😆 😅 😂 🤣 😊 😇 🙂 🙃 😉 😌 😍 🥰 😘
        Matematiksel semboller: ∫∑∏√∂∇∆∉∈ ≤≥±×÷ℝℤℚℕℂ
        Müzik sembolleri: ♩♪♫♬♭♮♯
        Ok işaretleri: ←→↑↓↔↕↖↗↘↙
        Kutu çizim: ┌┬┐├┼┤└┴┘│─
        Kalp ve diğer semboller: ❤️💕💔💖 ⭐✨⚡️ ✅❌⛔
        Takım sembolleri: ♠️♥️♦️♣️
        Telefon sembolleri: ☎️📱📲📞📟📠
        Standart semboller: ©®™§¶†‡
        Para birimleri: $€£¥₺₽₹
        """
        
        # ByteLevel tokenizer kullan (emoji ve özel karakterler için en uygun)
        tokenizer = self.ByteLevelTokenizer(vocab_size=500)
        tokenizer.train([special_text])
        
        # Encode ve decode
        encoded = tokenizer.encode(special_text)
        decoded = tokenizer.decode(encoded)
        
        # Sonuçları yazdır
        print("\nÖzel karakterler ve emojiler testi:")
        print(f"Encode sonucu: {len(encoded)} token")
        print(f"İlk 20 token ID: {encoded[:20]}")
        
        # Bazı özel karakterlerin doğru şekilde decode edildiğini kontrol et
        for char_set in ["😀", "∫", "♫", "→", "❤️", "€"]:
            self.assertIn(char_set, decoded, f"'{char_set}' karakteri decode edilmedi")
    
    def _text_similarity(self, original: str, decoded: str) -> float:
        """
        İki metin arasındaki benzerliği hesaplar.
        
        Args:
            original: Orijinal metin
            decoded: Decode edilmiş metin
            
        Returns:
            float: Benzerlik yüzdesi (0-100)
        """
        # Basitleştirilmiş benzerlik: boşlukları temizle ve karakterleri karşılaştır
        original_clean = ''.join(original.split())
        decoded_clean = ''.join(decoded.split())
        
        # Minimum uzunluk üzerinden karakterleri karşılaştır
        min_len = min(len(original_clean), len(decoded_clean))
        if min_len == 0:
            return 0.0
            
        matches = sum(1 for i in range(min_len) if original_clean[i] == decoded_clean[i])
        return (matches / min_len) * 100


if __name__ == "__main__":
    unittest.main()