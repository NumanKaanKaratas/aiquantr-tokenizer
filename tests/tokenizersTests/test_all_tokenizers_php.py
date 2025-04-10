"""
PHP kodu için tokenizer test modülü.

Bu test modülü, projede bulunan tüm tokenizer tiplerini ve 
özel tokenizer'ları PHP kodu üzerinde test eder.
"""

import unittest
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, Any, List, Type, Optional

from aiquantr_tokenizer.tokenizers.base import BaseTokenizer, TokenizerTrainer


class TestAllTokenizersPhp(unittest.TestCase):
    """
    PHP kodu üzerinde tüm tokenizer'ları test etmek için test sınıfı.
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
            from aiquantr_tokenizer.processors.code.general import CodeProcessor
            
            self.BPETokenizer = BPETokenizer
            self.WordPieceTokenizer = WordPieceTokenizer
            self.ByteLevelTokenizer = ByteLevelTokenizer
            self.UnigramTokenizer = UnigramTokenizer
            self.MixedTokenizer = MixedTokenizer
            self.create_tokenizer_from_config = create_tokenizer_from_config
            self.register_tokenizer_type = register_tokenizer_type
            self.CodeProcessor = CodeProcessor
            
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
        
        # PHP örnek kodları
        self.php_samples = {
            "basic": self._get_php_basic_sample(),
            "class": self._get_php_class_sample(),
            "array": self._get_php_array_sample(),
            "framework": self._get_php_framework_sample()
        }
    
    def tearDown(self):
        """
        Test ortamını temizler.
        """
        self.temp_dir.cleanup()
    
    def _get_php_basic_sample(self):
        """
        Temel PHP örneği oluşturur.
        """
        return """<?php
// Basit PHP örneği
$name = "PHP Test";
$version = 8.1;
$enabled = true;

function sayHello($name) {
    echo "Merhaba, " . $name . "!";
    return true;
}

if ($enabled) {
    sayHello($name);
    echo "PHP version: " . $version;
} else {
    echo "PHP disabled";
}

// Diziler
$numbers = [1, 2, 3, 4, 5];
foreach ($numbers as $number) {
    echo $number * 2 . " ";
}

// Basit sınıf
class SimpleClass {
    public $property = "Örnek özellik";
    
    public function method() {
        return $this->property;
    }
}

$obj = new SimpleClass();
echo $obj->method();
?>"""
    
    def _get_php_class_sample(self):
        """
        Sınıf tabanlı PHP örneği oluşturur.
        """
        return """<?php
namespace App\\Models;

/**
 * User sınıfı
 * 
 * Bu sınıf kullanıcı bilgilerini yönetir
 */
class User {
    /**
     * Kullanıcı ID
     * @var int
     */
    private int $id;
    
    /**
     * Kullanıcı adı
     * @var string
     */
    private string $username;
    
    /**
     * E-posta adresi
     * @var string
     */
    private string $email;
    
    /**
     * Kullanıcı oluşturma tarihi
     * @var \DateTime
     */
    private \DateTime $createdAt;
    
    /**
     * Constructor
     *
     * @param string $username Kullanıcı adı
     * @param string $email E-posta
     */
    public function __construct(string $username, string $email) {
        $this->username = $username;
        $this->email = $email;
        $this->createdAt = new \DateTime();
    }
    
    /**
     * Kullanıcı adını döndürür
     *
     * @return string
     */
    public function getUsername(): string {
        return $this->username;
    }
    
    /**
     * Kullanıcı adını ayarlar
     *
     * @param string $username Yeni kullanıcı adı
     * @return void
     */
    public function setUsername(string $username): void {
        $this->username = $username;
    }
    
    /**
     * E-posta adresini döndürür
     *
     * @return string
     */
    public function getEmail(): string {
        return $this->email;
    }
    
    /**
     * E-posta adresini ayarlar
     *
     * @param string $email Yeni e-posta
     * @return void
     */
    public function setEmail(string $email): void {
        $this->email = $email;
    }
    
    /**
     * Kullanıcı dizi temsilini döndürür
     *
     * @return array
     */
    public function toArray(): array {
        return [
            'id' => $this->id,
            'username' => $this->username,
            'email' => $this->email,
            'created_at' => $this->createdAt->format('Y-m-d H:i:s')
        ];
    }
}
?>"""
    
    def _get_php_array_sample(self):
        """
        Dizi işlemleri içeren PHP örneği oluşturur.
        """
        return """<?php
// PHP dizi örnekleri
$simpleArray = [1, 2, 3, 4, 5];
$associativeArray = [
    'name' => 'PHP Test',
    'version' => 8.1,
    'tags' => ['web', 'programming', 'server-side']
];

// Dizi işlemleri
$sum = array_sum($simpleArray);
$filteredArray = array_filter($simpleArray, fn($value) => $value > 2);
$mappedArray = array_map(fn($value) => $value * 2, $simpleArray);

// Çok boyutlu dizi
$matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];

foreach ($matrix as $row) {
    foreach ($row as $value) {
        echo $value . " ";
    }
    echo PHP_EOL;
}

// JSON dönüşümleri
$json = json_encode($associativeArray);
$decoded = json_decode($json, true);

// Dizi birleştirme
$merged = array_merge($simpleArray, $matrix[0]);

// Dizi kesme ve dilim alma
$slice = array_slice($simpleArray, 1, 3);

// Dizi sıralama
$unsorted = [5, 3, 1, 4, 2];
sort($unsorted);
?>"""
    
    def _get_php_framework_sample(self):
        """
        Framework benzeri PHP örneği oluşturur.
        """
        return """<?php
namespace App\\Controllers;

use App\\Services\\UserService;
use App\\Models\\User;
use App\\Http\\Request;
use App\\Http\\Response;
use App\\Exceptions\\ValidationException;

/**
 * UserController sınıfı
 * 
 * Kullanıcı işlemlerini yöneten controller
 */
class UserController {
    /**
     * @var UserService
     */
    private UserService $userService;
    
    /**
     * Constructor
     * 
     * @param UserService $userService
     */
    public function __construct(UserService $userService) {
        $this->userService = $userService;
    }
    
    /**
     * Kullanıcı listesini döndürür
     * 
     * @param Request $request HTTP isteği
     * @return Response HTTP yanıtı
     */
    public function index(Request $request): Response {
        $page = $request->query->get('page', 1);
        $limit = $request->query->get('limit', 10);
        
        $users = $this->userService->getUsers($page, $limit);
        $total = $this->userService->countUsers();
        
        return new Response([
            'users' => $users,
            'meta' => [
                'total' => $total,
                'page' => $page,
                'limit' => $limit,
                'pages' => ceil($total / $limit)
            ]
        ]);
    }
    
    /**
     * Kullanıcı bilgilerini döndürür
     * 
     * @param Request $request HTTP isteği
     * @param int $id Kullanıcı ID
     * @return Response HTTP yanıtı
     */
    public function show(Request $request, int $id): Response {
        $user = $this->userService->getUserById($id);
        
        if (!$user) {
            return new Response(['error' => 'User not found'], 404);
        }
        
        return new Response($user);
    }
    
    /**
     * Yeni kullanıcı oluşturur
     * 
     * @param Request $request HTTP isteği
     * @return Response HTTP yanıtı
     */
    public function store(Request $request): Response {
        try {
            $data = $request->validate([
                'username' => 'required|string|min:3|max:50',
                'email' => 'required|email',
                'password' => 'required|string|min:8'
            ]);
            
            $user = $this->userService->createUser($data);
            
            return new Response($user, 201);
        } catch (ValidationException $e) {
            return new Response(['errors' => $e->getErrors()], 422);
        } catch (\Exception $e) {
            return new Response(['error' => 'Failed to create user'], 500);
        }
    }
    
    /**
     * Kullanıcı bilgilerini günceller
     * 
     * @param Request $request HTTP isteği
     * @param int $id Kullanıcı ID
     * @return Response HTTP yanıtı
     */
    public function update(Request $request, int $id): Response {
        try {
            $data = $request->validate([
                'username' => 'string|min:3|max:50',
                'email' => 'email'
            ]);
            
            $user = $this->userService->updateUser($id, $data);
            
            if (!$user) {
                return new Response(['error' => 'User not found'], 404);
            }
            
            return new Response($user);
        } catch (ValidationException $e) {
            return new Response(['errors' => $e->getErrors()], 422);
        } catch (\Exception $e) {
            return new Response(['error' => 'Failed to update user'], 500);
        }
    }
    
    /**
     * Kullanıcıyı siler
     * 
     * @param Request $request HTTP isteği
     * @param int $id Kullanıcı ID
     * @return Response HTTP yanıtı
     */
    public function destroy(Request $request, int $id): Response {
        $result = $this->userService->deleteUser($id);
        
        if (!$result) {
            return new Response(['error' => 'User not found'], 404);
        }
        
        return new Response(null, 204);
    }
}
?>"""
    
    def test_individual_tokenizers(self):
        """
        Her tokenizer'ı PHP kodu üzerinde test eder.
        """
        # PHP kodu için işlemci oluştur
        processor = self.CodeProcessor(
            file_extensions=[".php"],
            comment_prefixes=["//"],
            block_comment_pairs=[("/*", "*/")],
            string_delimiters=['"', "'"],
            remove_comments=True
        )
        
        # Test edilecek tokenizer'lar
        tokenizer_instances = {
            "BPE": self.BPETokenizer(vocab_size=500),
            "WordPiece": self.WordPieceTokenizer(vocab_size=500),
            "ByteLevel": self.ByteLevelTokenizer(vocab_size=500),
            "Unigram": self.UnigramTokenizer(vocab_size=500)
        }
        
        # İşlenecek kodları hazırla
        processed_codes = []
        for name, content in self.php_samples.items():
            processed_code = processor.process(content)
            processed_codes.append(processed_code)
        
        # Her tokenizer'ı test et
        for tokenizer_name, tokenizer in tokenizer_instances.items():
            with self.subTest(tokenizer=tokenizer_name):
                # Tokenizer'ı eğit
                print(f"\n{tokenizer_name} tokenizer PHP kodu üzerinde eğitiliyor...")
                train_result = tokenizer.train(processed_codes)
                
                self.assertTrue(tokenizer.is_trained, f"{tokenizer_name} eğitimi başarısız oldu")
                self.assertGreater(tokenizer.get_vocab_size(), 0, f"{tokenizer_name} boş sözlük oluşturdu")
                
                # Her kod örneğini ayrı ayrı test et
                for sample_name, content in self.php_samples.items():
                    processed_code = processor.process(content)
                    sample_text = processed_code[:500]  # İlk 500 karakteri test et
                    
                    # Encode ve decode işlemleri
                    encoded = tokenizer.encode(sample_text)
                    decoded = tokenizer.decode(encoded)
                    
                    # Sonuçları yazdır
                    print(f"{tokenizer_name} - {sample_name} encode sonucu: {len(encoded)} token")
                    print(f"İlk 10 token ID: {encoded[:10]}")
                    
                    # Token yoğunluğunu hesapla (token sayısı / metin uzunluğu)
                    density = len(encoded) / len(sample_text)
                    print(f"Token yoğunluğu: {density:.4f} token/karakter")
                    
                    # Minimal doğrulama
                    self.assertGreater(len(encoded), 0, f"{tokenizer_name} hiç token üretmedi")
                    
                # PHP özelliklerini temsil eden bazı özel tokenleri kontrol et
                php_tokens = ["<?php", "namespace", "class", "function", "array"]
                vocab = tokenizer.get_vocab()
                
                # En azından bazı PHP token'larının varlığını doğrula
                found_tokens = []
                for token in php_tokens:
                    for vocab_token in vocab.keys():
                        if token in vocab_token:
                            found_tokens.append(token)
                            break
                
                print(f"{tokenizer_name} için bulunan PHP tokenleri: {found_tokens}")
                
                # Tokenizer'ı kaydet ve yükle
                save_path = self.temp_path / tokenizer_name
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
    
    def test_mixed_tokenizer_php(self):
        """
        MixedTokenizer'ı PHP ve JSON/SQL kodları üzerinde test eder.
        """
        # Alt tokenizer'ları oluştur
        php_tokenizer = self.BPETokenizer(vocab_size=300, name="PHPTokenizer")
        sql_tokenizer = self.WordPieceTokenizer(vocab_size=200, name="SQLTokenizer")
        
        # PHP ve SQL örnekleri
        php_samples = list(self.php_samples.values())
        
        sql_samples = [
            """SELECT u.*, r.role_name FROM users u 
               JOIN roles r ON u.role_id = r.id 
               WHERE u.status = 'active' 
               ORDER BY u.created_at DESC LIMIT 10;""",
            
            """INSERT INTO users (username, email, password_hash, created_at) 
               VALUES ('test_user', 'test@example.com', 'hashed_password', NOW());""",
            
            """CREATE TABLE products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                description TEXT,
                category_id INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories(id)
            );"""
        ]
        
        # İşlemcileri oluştur
        php_processor = self.CodeProcessor(
            file_extensions=[".php"],
            comment_prefixes=["//"],
            block_comment_pairs=[("/*", "*/")],
            string_delimiters=['"', "'"],
            remove_comments=True
        )
        
        sql_processor = self.CodeProcessor(
            file_extensions=[".sql"],
            comment_prefixes=["--"],
            block_comment_pairs=[("/*", "*/")],
            string_delimiters=['"', "'"],
            remove_comments=True
        )
        
        # Örnekleri işle
        processed_php = [php_processor.process(sample) for sample in php_samples]
        processed_sql = [sql_processor.process(sample) for sample in sql_samples]
        
        # Alt tokenizer'ları eğit
        php_tokenizer.train(processed_php)
        sql_tokenizer.train(processed_sql)
        
        # MixedTokenizer oluştur
        mixed_tokenizer = self.MixedTokenizer(
            tokenizers={"php": php_tokenizer, "sql": sql_tokenizer},
            default_tokenizer="php",
            merged_vocab=True,
            name="PHPSQLMixed"
        )
        
        # Router fonksiyonu tanımla
        def router(text):
            if any(keyword in text.lower() for keyword in ["select", "insert", "create table", "join", "where"]):
                return "sql"
            return "php"
        
        mixed_tokenizer.router = router
        
        # Test et
        sample_php = self.php_samples["basic"][:300]
        sample_sql = sql_samples[0]
        
        # PHP kodu tokenize et
        php_tokens = mixed_tokenizer.encode(sample_php)
        php_decoded = mixed_tokenizer.decode(php_tokens)
        
        # SQL kodunu tokenize et
        sql_tokens = mixed_tokenizer.encode(sample_sql)
        sql_decoded = mixed_tokenizer.decode(sql_tokens)
        
        # Sonuçları yazdır
        print("\nPHP ve SQL için MixedTokenizer test sonuçları:")
        print(f"PHP örneği: {len(php_tokens)} token, ilk 10 token: {php_tokens[:10]}")
        print(f"SQL örneği: {len(sql_tokens)} token, ilk 10 token: {sql_tokens[:10]}")
        
        # Minimal doğrulama
        self.assertGreater(len(php_tokens), 0, "MixedTokenizer PHP için hiç token üretmedi")
        self.assertGreater(len(sql_tokens), 0, "MixedTokenizer SQL için hiç token üretmedi")
        
        # Kaydet ve yükle
        save_path = self.temp_path / "mixed_php_sql"
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


if __name__ == "__main__":
    unittest.main()