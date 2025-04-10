# test_integration.py
import unittest
from aiquantr_tokenizer.core.data_collector import DataCollector
from aiquantr_tokenizer.core.tokenizer_trainer import TokenizerTrainer
from aiquantr_tokenizer.data.sources.custom_source import CustomDataSource
from aiquantr_tokenizer.processors.code.python import PythonProcessor
from aiquantr_tokenizer.tokenizers.bpe import BPETokenizer


def test_full_tokenizer_pipeline():
    """Test the complete tokenizer pipeline from data collection to training."""
    # 1. Veri toplama
    collector = DataCollector()
    collector.add_source(CustomDataSource([
        "def example(): return True",
        "class TestClass:\n    def method(self):\n        pass",
        "print('Hello world')"
    ]))
    
    # 2. Veri işleme
    processor = PythonProcessor(remove_comments=True)
    cleaned_data = [processor.process(text) for text in collector.collect_data()]
    
    # 3. Tokenizer eğitimi
    trainer = TokenizerTrainer(batch_size=2, num_iterations=3)
    tokenizer = BPETokenizer(vocab_size=1000)
    trainer.train(tokenizer, cleaned_data)
    
    # 4. Sonuçları test et
    assert tokenizer.vocab_size > 0
    encoded = tokenizer.encode("def test(): pass")
    assert len(encoded) > 0
    decoded = tokenizer.decode(encoded)
    assert "test" in decoded