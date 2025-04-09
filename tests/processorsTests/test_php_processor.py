"""
PHP kod işleme sınıfı için testler.
"""

from tests.processorsTests.test_processors import BaseProcessorTest

class TestPhpProcessor(BaseProcessorTest):
    """
    PHP işlemcisi için test durumları.
    """
    
    def setUp(self):
        """
        Test ortamını hazırlar.
        """
        super().setUp()
        try:
            from aiquantr_tokenizer.processors.code.php import PhpProcessor
            self.PhpProcessor = PhpProcessor
        except ImportError:
            self.skipTest("aiquantr_tokenizer.processors.code.php modülü bulunamadı")
    
    def test_basic_php_processing(self):
        """
        Temel PHP kod işleme işlevselliğini test eder.
        """
        processor = self.PhpProcessor()
        
        php_code = """<?php
function example_function($param) {
    echo "Hello, " . $param;
    return true;
}
?>"""
        
        self.assertProcessingResult(
            processor,
            php_code,
            expected_parts=[
                "<?php",
                "function example_function($param) {",
                "echo",
                "return true;"
            ]
        )
    
    def test_remove_php_tags(self):
        """
        PHP etiketlerini kaldırma işlevini test eder.
        """
        processor = self.PhpProcessor(remove_php_tags=True)
        
        php_code = """<?php
$x = 10;
?>

Some HTML content

<?php
echo $x;
?>"""
        
        result = self.assertProcessingResult(
            processor,
            php_code,
            not_expected_parts=["<?php", "?>"],
            expected_parts=["$x = 10;", "echo $x;", "Some HTML content"]
        )
    
    def test_remove_html_from_php(self):
        """
        PHP'den HTML kaldırma işlevini test eder.
        """
        processor = self.PhpProcessor(remove_html=True)
        
        php_code = """<?php
$x = 10;
?>
<html>
<body>
    <h1>Title</h1>
    <?php echo $x; ?>
</body>
</html>"""
        
        result = self.assertProcessingResult(
            processor,
            php_code,
            not_expected_parts=["<html>", "<body>", "<h1>", "</h1>", "</body>", "</html>"],
            expected_parts=["<?php", "$x = 10;", "?>", "echo $x;"]
        )
    
    def test_remove_php_comments(self):
        """
        PHP yorumlarını kaldırma işlevini test eder.
        """
        processor = self.PhpProcessor(remove_comments=True)
        
        php_code = """<?php
// Bu bir satır yorumu
function example() {
    /* Bu bir
       blok yorum */
    $x = 10; // Satır sonu yorumu
    return $x;
}
?>"""
        
        self.assertProcessingResult(
            processor,
            php_code,
            not_expected_parts=["// Bu bir satır yorumu", "/* Bu bir", "blok yorum", "// Satır sonu yorumu"],
            expected_parts=["<?php", "function example() {", "$x = 10;", "return $x;"]
        )
    
    def test_tokenize_php(self):
        """
        PHP tokenize işlevini test eder.
        """
        processor = self.PhpProcessor()
        
        php_code = """<?php
function example($param) {
    $x = 10;
    return $x + $param;
}
?>"""
        
        tokens = processor.tokenize(php_code)
        
        # Tokenization doğru yapıldı mı kontrol et
        expected_tokens = ["function", "example", "(", "$param", ")", "{", "$x", "=", "10", ";", "return", "$x", "+", "$param", "}", "?>"]
        for token in expected_tokens:
            self.assertIn(token, tokens)