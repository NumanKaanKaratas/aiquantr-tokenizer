"""
Alan-spesifik bilgi işleme sınıfı.

Bu modül, belirli bilgi alanlarına (tıp, hukuk, finans vb.)
özel işlemcileri içerir.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple, Pattern, Set

from processors.knowledge.base import BaseKnowledgeProcessor

# Logger oluştur
logger = logging.getLogger(__name__)


class DomainSpecificProcessor(BaseKnowledgeProcessor):
    """
    Alan-spesifik bilgi metinlerini işlemek için sınıf.
    
    Bu sınıf, belirli bilgi alanlarına özgü metinleri
    işlemek için özelleştirilebilir.
    """
    
    def __init__(
        self,
        domain: str,
        domain_specific_patterns: Optional[Dict[str, Pattern]] = None,
        domain_specific_terms: Optional[Dict[str, List[str]]] = None,
        normalize_terminology: bool = False,
        extract_domain_entities: bool = True,
        **kwargs
    ):
        """
        DomainSpecificProcessor başlatıcısı.
        
        Args:
            domain: Bilgi alanı (tıp, hukuk, teknoloji vb.)
            domain_specific_patterns: Alana özgü regex desenleri
            domain_specific_terms: Alana özgü terimler sözlüğü
            normalize_terminology: Terminolojiyi normalleştir (varsayılan: False)
            extract_domain_entities: Alana özgü varlıkları çıkar (varsayılan: True)
            **kwargs: BaseKnowledgeProcessor için ek parametreler
        """
        super().__init__(
            knowledge_type="domain-specific",
            domain=domain,
            **kwargs
        )
        
        self.domain_specific_patterns = domain_specific_patterns or {}
        self.domain_specific_terms = domain_specific_terms or {}
        self.normalize_terminology = normalize_terminology
        self.extract_domain_entities = extract_domain_entities
        
        # Derleme zamanı terim desenleri
        self._compile_domain_patterns()
        
        # İstatistikler
        self.stats.update({
            "domain_entities_extracted": 0,
            "terms_normalized": 0
        })
        
    def _compile_domain_patterns(self):
        """
        Alana özgü regex desenlerini derler ve terim eşleştirme kalıplarını hazırlar.
        """
        # Zaten derlenmiş regex desenleri
        self.compiled_patterns = {}
        
        # Derlenmemiş desenleri derle
        for key, pattern in self.domain_specific_patterns.items():
            if isinstance(pattern, str):
                self.compiled_patterns[key] = re.compile(pattern)
            else:
                self.compiled_patterns[key] = pattern
        
        # Terim kalıpları
        self.term_patterns = {}
        
        # Her terim kategorisi için desen oluştur
        for category, terms in self.domain_specific_terms.items():
            if terms:
                # Terimleri sırayla birleştirerek bir regex deseni oluştur
                # Uzun terimleri önce yerleştir ki kısa terimlerle çakışmayı engelleyelim
                sorted_terms = sorted(terms, key=len, reverse=True)
                pattern_str = r'\b(?:' + '|'.join(map(re.escape, sorted_terms)) + r')\b'
                self.term_patterns[category] = re.compile(pattern_str, re.IGNORECASE)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Ön işleme adımlarını gerçekleştirir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Ön işlenmiş metin
        """
        text = super()._preprocess_text(text)
        
        # Terminolojiyi normalleştir
        if self.normalize_terminology:
            text = self._normalize_terminology(text)
            
        return text
    
    def _normalize_terminology(self, text: str) -> str:
        """
        Alana özgü terminolojiyi normalleştirir.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Normalleştirilmiş metin
        """
        normalized_text = text
        norm_count = 0
        
        # Terim desenlerini kullanarak normalleştirme yap
        for category, pattern in self.term_patterns.items():
            # Aynı kategorideki terimleri standartlaştır
            matches = pattern.finditer(normalized_text)
            
            # Eşleşmeleri işle (sondan başa doğru, metin kayması olmasın)
            matches = list(matches)
            for match in reversed(matches):
                term = match.group(0)
                # Standart terim kullan (kategorinin ilk terimi)
                if category in self.domain_specific_terms and self.domain_specific_terms[category]:
                    standard_term = self.domain_specific_terms[category][0]
                    
                    # Küçük/büyük harf uyumunu koru
                    if term.isupper():
                        standard_term = standard_term.upper()
                    elif term[0].isupper():
                        standard_term = standard_term.capitalize()
                    
                    # Değiştir
                    if term.lower() != standard_term.lower():
                        normalized_text = normalized_text[:match.start()] + standard_term + normalized_text[match.end():]
                        norm_count += 1
        
        self.stats["terms_normalized"] += norm_count
        return normalized_text
    
    def _initialize_domain_patterns(self) -> Dict[str, Pattern]:
        """
        Alana özgü regex desenlerini başlatır.
        
        Returns:
            Dict[str, Pattern]: Derlenen regex desenleri
        """
        patterns = {}
        
        # Tıp alanı için desenler
        if self.domain.lower() == "medical":
            patterns.update({
                "disease": re.compile(r'\b(?:cancer|diabetes|asthma|alzheimer|hypertension|disorder)\b', re.IGNORECASE),
                "drug": re.compile(r'\b(?:mg|mcg|ml|tablet|injection|dose)\b', re.IGNORECASE),
                "measurement": re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|mmHg|mmol/L)\b'),
                "body_part": re.compile(r'\b(?:heart|lung|liver|kidney|brain|skin)\b', re.IGNORECASE)
            })
            
        # Hukuk alanı için desenler
        elif self.domain.lower() == "legal":
            patterns.update({
                "case_reference": re.compile(r'\b[A-Z][a-z]+ v\. [A-Z][a-z]+\b'),
                "statute": re.compile(r'\b(?:Section|§)\s+\d+(?:\.\d+)*\b'),
                "court": re.compile(r'\b(?:Supreme Court|District Court|Court of Appeals|Circuit Court)\b'),
                "legal_term": re.compile(r'\b(?:plaintiff|defendant|appellant|respondent|jurisdiction|tort|contract)\b', re.IGNORECASE)
            })
            
        # Finans alanı için desenler
        elif self.domain.lower() == "finance":
            patterns.update({
                "money": re.compile(r'\$\d+(?:\.\d+)?(?:M|B|T)?|\d+(?:\.\d+)?\s*(?:dollars|USD|EUR|GBP|JPY)'),
                "percentage": re.compile(r'\d+(?:\.\d+)?\s*%'),
                "company": re.compile(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\s+(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation)\b'),
                "financial_term": re.compile(r'\b(?:stock|bond|dividend|investment|portfolio|asset|liability)\b', re.IGNORECASE)
            })
            
        # Teknoloji alanı için desenler
        elif self.domain.lower() == "technology":
            patterns.update({
                "programming": re.compile(r'\b(?:function|class|method|variable|object|interface|API)\b'),
                "file_format": re.compile(r'\b(?:\.pdf|\.doc|\.txt|\.json|\.xml|\.html|\.csv|\.jpg|\.png)\b', re.IGNORECASE),
                "tech_company": re.compile(r'\b(?:Google|Microsoft|Apple|Amazon|Facebook|IBM|Intel|Oracle|Cisco)\b'),
                "tech_term": re.compile(r'\b(?:cloud|server|database|network|interface|algorithm|software|hardware)\b', re.IGNORECASE)
            })
            
        return patterns
        
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Metinden alana özgü varlık bilgilerini çıkarır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[Dict[str, Any]]: Çıkarılan varlık bilgileri
        """
        entities = []
        
        if not self.extract_domain_entities:
            return entities
        
        # Alana özgü varlıkları çıkar
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    'type': entity_type.upper(),
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Termlerden varlık çıkar
        for category, pattern in self.term_patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    'type': category.upper(),
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        self.stats["domain_entities_extracted"] += len(entities)
        return entities
    
    def extract_facts_from_text(self, text: str) -> List[str]:
        """
        Metinden alana özgü gerçek bilgilerini çıkarır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[str]: Çıkarılan gerçek ifadeleri
        """
        facts = []
        
        # Alana özgü varlıkları içeren cümleleri çıkar
        sentences = re.split(r'[.!?]', text)
        
        entities = self.extract_entities_from_text(text)
        entity_texts = [e['text'].lower() for e in entities]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Bu cümle bir domain varlığı içeriyor mu?
                has_entity = any(entity.lower() in sentence.lower() for entity in entity_texts)
                
                # Alana özgü kelimeler içeriyor mu?
                domain_keywords = self._get_domain_keywords()
                has_keyword = any(keyword.lower() in sentence.lower() for keyword in domain_keywords)
                
                if has_entity or has_keyword:
                    facts.append(sentence)
        
        return facts
    
    def extract_relations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Metinden alana özgü ilişki bilgilerini çıkarır.
        
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
        
        # Cümlelere ayır
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Bu cümlede bulunan varlık çiftlerini incele
            sentence_entities = [e for e in entities if e['text'].lower() in sentence.lower()]
            
            for i, entity1 in enumerate(sentence_entities):
                for j, entity2 in enumerate(sentence_entities):
                    if i != j:
                        # İki varlık arasında ilişki olup olmadığını kontrol et
                        relation_verbs = self._get_domain_relations()
                        
                        for relation in relation_verbs:
                            # İlişki bu cümlede geçiyor mu?
                            if relation.lower() in sentence.lower():
                                relations.append({
                                    'source': entity1['text'],
                                    'source_type': entity1['type'],
                                    'relation': relation,
                                    'target': entity2['text'],
                                    'target_type': entity2['type'],
                                    'sentence': sentence
                                })
        
        return relations
    
    def _get_domain_keywords(self) -> List[str]:
        """
        Alana özgü anahtar kelimeleri döndürür.
        
        Returns:
            List[str]: Alan anahtar kelimeleri
        """
        keywords = []
        
        # Tıp alanı için anahtar kelimeler
        if self.domain.lower() == "medical":
            keywords = ["patient", "treatment", "diagnosis", "symptom", "disease", "medical", "clinical", 
                     "doctor", "medicine", "prescription", "therapy", "hospital", "physician"]
            
        # Hukuk alanı için anahtar kelimeler
        elif self.domain.lower() == "legal":
            keywords = ["court", "law", "statute", "legal", "judge", "attorney", "plaintiff", "defendant",
                     "contract", "tort", "criminal", "civil", "jurisdiction", "verdict"]
            
        # Finans alanı için anahtar kelimeler
        elif self.domain.lower() == "finance":
            keywords = ["market", "investment", "stock", "bond", "fund", "return", "risk", "portfolio",
                     "asset", "liability", "equity", "dividend", "interest", "capital"]
            
        # Teknoloji alanı için anahtar kelimeler
        elif self.domain.lower() == "technology":
            keywords = ["software", "hardware", "algorithm", "data", "network", "system", "interface",
                     "cloud", "server", "program", "application", "code", "database"]
            
        return keywords
    
    def _get_domain_relations(self) -> List[str]:
        """
        Alana özgü ilişki ifadelerini döndürür.
        
        Returns:
            List[str]: İlişki ifadeleri
        """
        relations = []
        
        # Tıp alanı için ilişkiler
        if self.domain.lower() == "medical":
            relations = ["treats", "causes", "prevents", "diagnoses", "is symptom of", "indicates", 
                      "is contraindicated with", "is administered to"]
            
        # Hukuk alanı için ilişkiler
        elif self.domain.lower() == "legal":
            relations = ["rules on", "appeals", "represents", "presides over", "is plaintiff in",
                      "is defendant in", "argues", "testifies", "decides"]
            
        # Finans alanı için ilişkiler
        elif self.domain.lower() == "finance":
            relations = ["invests in", "owns", "issues", "buys", "sells", "acquires", "merges with",
                      "reports", "increases", "decreases"]
            
        # Teknoloji alanı için ilişkiler
        elif self.domain.lower() == "technology":
            relations = ["develops", "releases", "integrates with", "runs on", "supports", "is compatible with",
                      "processes", "stores", "transfers", "analyzes"]
            
        return relations