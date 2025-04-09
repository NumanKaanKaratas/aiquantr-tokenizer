"""
Genel bilgi işleme sınıfı.

Bu modül, çeşitli bilgi formatlarını işlemek için
genel amaçlı işlemcileri içerir.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple, Pattern, Set

from processors.knowledge.base import BaseKnowledgeProcessor

# Logger oluştur
logger = logging.getLogger(__name__)

# NLP paketlerini isteğe bağlı olarak yükle
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    logger.warning("nltk paketi bulunamadı. Bazı NLP özellikleri sınırlı olacak.")


class GeneralKnowledgeProcessor(BaseKnowledgeProcessor):
    """
    Genel bilgi metinlerini işlemek için sınıf.
    
    Bu sınıf, yapılandırılmamış metinleri, makaleleri, belgeleri 
    ve diğer bilgi formatlarını işlemek için kullanılır.
    """
    
    def __init__(
        self,
        summarize: bool = False,
        max_summary_sentences: int = 3,
        remove_citations: bool = False,
        remove_urls: bool = False,
        simplify_text: bool = False,
        **kwargs
    ):
        """
        GeneralKnowledgeProcessor başlatıcısı.
        
        Args:
            summarize: Metni özetleme (varsayılan: False)
            max_summary_sentences: Özetteki maksimum cümle sayısı
            remove_citations: Alıntıları kaldırma (varsayılan: False)
            remove_urls: URL'leri kaldırma (varsayılan: False)
            simplify_text: Metni basitleştirme (varsayılan: False)
            **kwargs: BaseKnowledgeProcessor için ek parametreler
        """
        super().__init__(
            knowledge_type="general",
            **kwargs
        )
        
        self.summarize = summarize
        self.max_summary_sentences = max_summary_sentences
        self.remove_citations = remove_citations
        self.remove_urls = remove_urls
        self.simplify_text = simplify_text
        
        # Regex desenleri
        self.citation_pattern = re.compile(r'\[\d+\]|\(\d+\)|\[citation needed\]')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # NLTK kontrolü
        if self.summarize and not HAS_NLTK:
            logger.warning("Özetleme için nltk gereklidir. Özetleme devre dışı bırakıldı.")
            self.summarize = False
            
        # NLTK kaynakları yükle
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.warning("NLTK punkt verileri bulunamadı. İndirmek için nltk.download('punkt') çalıştırın.")
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.warning("NLTK stopwords verileri bulunamadı. İndirmek için nltk.download('stopwords') çalıştırın.")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Ön işleme adımlarını gerçekleştirir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Ön işlenmiş metin
        """
        text = super()._preprocess_text(text)
        
        # Alıntıları kaldır
        if self.remove_citations:
            text = self.citation_pattern.sub('', text)
        
        # URL'leri kaldır
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
            
        return text
    
    def _postprocess_text(self, text: str) -> str:
        """
        Son işleme adımlarını gerçekleştirir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Son işlenmiş metin
        """
        # Metni özetle (eğer isteniyorsa)
        if self.summarize and HAS_NLTK:
            text = self._summarize_text(text)
        
        # Metni basitleştir (eğer isteniyorsa)
        if self.simplify_text:
            text = self._simplify_text(text)
            
        return text
    
    def _summarize_text(self, text: str) -> str:
        """
        Metni özetler (basit bir özetleme algoritması).
        
        Args:
            text: Özetlenecek metin
            
        Returns:
            str: Özetlenmiş metin
        """
        if not HAS_NLTK:
            return text
            
        # Cümlelere ayır
        sentences = sent_tokenize(text)
        
        if len(sentences) <= self.max_summary_sentences:
            return text
            
        # Basit bir özet için cümle skorlama
        word_frequencies = {}
        
        # Kelimeleri tokenize et ve durdurma kelimelerini kaldır
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word.isalnum():
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        # En yüksek frekans
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        
        # Normalize et
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
            
        # Cümle skorları
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if i not in sentence_scores:
                        sentence_scores[i] = word_frequencies[word]
                    else:
                        sentence_scores[i] += word_frequencies[word]
        
        # En yüksek skorlu cümleleri seç
        selected_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:self.max_summary_sentences]
        selected_sentences = sorted(selected_sentences, key=lambda x: x[0])
        
        # Özeti oluştur
        summary = ' '.join([sentences[i] for i, _ in selected_sentences])
        
        return summary
    
    def _simplify_text(self, text: str) -> str:
        """
        Metni basitleştirir.
        
        Uzun cümleleri kısaltır ve karmaşık ifadeleri daha basit hale getirir.
        Bu basit bir uygulama olup, gelişmiş NLP yöntemleri gerekebilir.
        
        Args:
            text: Basitleştirilecek metin
            
        Returns:
            str: Basitleştirilmiş metin
        """
        if not HAS_NLTK:
            return text
            
        # Cümlelere ayır
        sentences = sent_tokenize(text)
        simplified_sentences = []
        
        for sentence in sentences:
            # Uzun cümleleri kısalt (basit bir yaklaşım)
            words = word_tokenize(sentence)
            if len(words) > 20:
                # Parantez içi ifadeleri kaldır
                sentence = re.sub(r'\([^)]*\)', '', sentence)
                # Virgülle ayrılmış yan cümleleri ayır
                parts = sentence.split(', ')
                if len(parts) > 2:
                    # Sadece ilk birkaç parçayı al
                    sentence = ', '.join(parts[:2]) + ('.' if not parts[1].endswith('.') else '')
            
            simplified_sentences.append(sentence)
        
        return ' '.join(simplified_sentences)
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Metinden varlık bilgilerini çıkarır.
        
        Basit bir isim varlığı tanıma yaklaşımı kullanır.
        Tam NER için daha gelişmiş NLP kütüphaneleri gerekebilir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[Dict[str, Any]]: Çıkarılan varlık bilgileri
        """
        entities = []
        
        # Kişi isimleri için basit bir regex (çok basit)
        person_pattern = re.compile(r'[A-Z][a-z]+ [A-Z][a-z]+')
        person_matches = person_pattern.finditer(text)
        
        for match in person_matches:
            entities.append({
                'type': 'PERSON',
                'text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Organizasyon isimleri için basit bir regex
        org_pattern = re.compile(r'(?:[A-Z][a-z]+ )+(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation)')
        org_matches = org_pattern.finditer(text)
        
        for match in org_matches:
            entities.append({
                'type': 'ORGANIZATION',
                'text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Tarihler için basit bir regex
        date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{2,4}')
        date_matches = date_pattern.finditer(text)
        
        for match in date_matches:
            entities.append({
                'type': 'DATE',
                'text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def extract_facts_from_text(self, text: str) -> List[str]:
        """
        Metinden gerçek bilgilerini çıkarır.
        
        Basit cümleleri gerçek olarak kabul eder.
        Daha gelişmiş bir gerçek çıkarma için NLP teknikleri gerekebilir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[str]: Çıkarılan gerçek ifadeleri
        """
        facts = []
        
        if HAS_NLTK:
            # Cümlelere ayır
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # Basit cümleleri gerçek olarak değerlendir
                # Daha gelişmiş bir gerçek çıkarma için 
                # cümle yapısı ve içerik analizi gerekebilir
                
                # Öznel ifadeleri filtrele
                if re.search(r'\bI think\b|\bprobably\b|\bmaybe\b|\bmight\b|\bperhaps\b|\bseems\b|\bappears\b', sentence, re.IGNORECASE):
                    continue
                    
                # Gerçeğe benzeyen ifadeleri ekle
                if re.search(r'\bis\b|\bare\b|\bwas\b|\bwere\b|\bthe\b', sentence, re.IGNORECASE):
                    facts.append(sentence.strip())
        else:
            # NLTK yoksa çok basit bir yaklaşım
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and re.search(r'\bis\b|\bare\b|\bwas\b|\bwere\b|\bthe\b', sentence, re.IGNORECASE):
                    facts.append(sentence)
        
        return facts
    
    def extract_relations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Metinden ilişki bilgilerini çıkarır.
        
        Basit bir kural temelli yaklaşım kullanır.
        Daha gelişmiş ilişki çıkarma için NLP teknikleri gerekebilir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[Dict[str, Any]]: Çıkarılan ilişki bilgileri
        """
        relations = []
        
        # Varlıkları çıkar
        entities = self.extract_entities_from_text(text)
        
        if not entities or len(entities) < 2:
            return relations
            
        if HAS_NLTK:
            # Cümlelere ayır
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # İlişki ifadelerini ara
                for i, entity1 in enumerate(entities):
                    entity1_text = entity1['text']
                    
                    if entity1_text in sentence:
                        for j, entity2 in enumerate(entities):
                            if i != j:
                                entity2_text = entity2['text']
                                
                                if entity2_text in sentence:
                                    # İki varlık arasındaki metni bul
                                    idx1 = sentence.find(entity1_text)
                                    idx2 = sentence.find(entity2_text)
                                    
                                    if idx1 < idx2:
                                        between = sentence[idx1 + len(entity1_text):idx2].strip()
                                    else:
                                        between = sentence[idx2 + len(entity2_text):idx1].strip()
                                    
                                    # İlişki ifadesini tespit et
                                    relation_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'belongs to', 
                                                    'works for', 'located in', 'part of', 'member of']
                                    
                                    for verb in relation_verbs:
                                        if verb in between.lower():
                                            relations.append({
                                                'source': entity1_text,
                                                'relation': verb,
                                                'target': entity2_text,
                                                'sentence': sentence
                                            })
                                            break
        
        return relations
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Metinden anahtar kelimeler çıkarır.
        
        NLTK varsa daha gelişmiş yöntemler kullanır, yoksa basit bir yaklaşım uygular.
        
        Args:
            text: İşlenecek metin
            max_keywords: Maksimum anahtar kelime sayısı
            
        Returns:
            List[str]: Çıkarılan anahtar kelimeler
        """
        if not HAS_NLTK:
            return super().extract_keywords(text, max_keywords)
            
        # NLTK ile anahtar kelime çıkarma
        words = word_tokenize(text.lower())
        
        # Durdurma kelimeleri
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'and', 'or', 'the', 'a', 'an', 'of', 'to', 'in', 'that', 'this', 'is', 'are', 'was', 'were'}
        
        # Durdurma kelimelerini ve noktalama işaretlerini kaldır
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        
        # Kelime frekanslarını hesapla
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Frekansa göre sırala ve en yüksek olanları al
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords