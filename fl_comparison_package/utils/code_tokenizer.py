"""
Code Tokenizer for Bug Prediction
Converts Python code into numerical features for ML models
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
import ast
import tokenize
import io


class CodeTokenizer:
    """Tokenize Python code for bug prediction models."""
    
    # Python keywords
    PYTHON_KEYWORDS = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
        'while', 'with', 'yield'
    }
    
    # Common Python built-in functions
    BUILTINS = {
        'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict',
        'set', 'tuple', 'bool', 'type', 'isinstance', 'hasattr', 'getattr',
        'setattr', 'open', 'input', 'sum', 'max', 'min', 'abs', 'round',
        'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter', 'any', 'all'
    }
    
    def __init__(self, max_length: int = 512, vocab_size: int = 10000):
        """
        Initialize tokenizer.
        
        Args:
            max_length: Maximum sequence length
            vocab_size: Maximum vocabulary size
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.fitted = False
        
    def _tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize Python code into a list of tokens.
        
        Args:
            code: Python source code string
            
        Returns:
            List of tokens
        """
        tokens = []
        
        try:
            # Use Python's tokenizer
            code_bytes = io.BytesIO(code.encode('utf-8'))
            for tok in tokenize.tokenize(code_bytes.readline):
                if tok.type == tokenize.ENCODING:
                    continue
                if tok.type == tokenize.ENDMARKER:
                    continue
                if tok.type == tokenize.NEWLINE:
                    tokens.append('<NEWLINE>')
                elif tok.type == tokenize.INDENT:
                    tokens.append('<INDENT>')
                elif tok.type == tokenize.DEDENT:
                    tokens.append('<DEDENT>')
                elif tok.type == tokenize.NL:
                    continue
                elif tok.type == tokenize.COMMENT:
                    tokens.append('<COMMENT>')
                elif tok.type == tokenize.STRING:
                    tokens.append('<STRING>')
                elif tok.type == tokenize.NUMBER:
                    tokens.append('<NUMBER>')
                elif tok.type == tokenize.NAME:
                    if tok.string in self.PYTHON_KEYWORDS:
                        tokens.append(f'KW_{tok.string}')
                    elif tok.string in self.BUILTINS:
                        tokens.append(f'BUILTIN_{tok.string}')
                    else:
                        tokens.append('<IDENTIFIER>')
                else:
                    tokens.append(tok.string)
        except tokenize.TokenizeError:
            # Fallback to simple regex tokenization
            tokens = self._simple_tokenize(code)
        except Exception:
            tokens = self._simple_tokenize(code)
            
        return tokens
    
    def _simple_tokenize(self, code: str) -> List[str]:
        """Simple regex-based tokenization fallback."""
        # Split by whitespace and punctuation
        pattern = r'(\w+|[^\w\s])'
        tokens = re.findall(pattern, code)
        
        processed_tokens = []
        for tok in tokens:
            if tok in self.PYTHON_KEYWORDS:
                processed_tokens.append(f'KW_{tok}')
            elif tok in self.BUILTINS:
                processed_tokens.append(f'BUILTIN_{tok}')
            elif tok.isdigit():
                processed_tokens.append('<NUMBER>')
            elif tok.startswith('"') or tok.startswith("'"):
                processed_tokens.append('<STRING>')
            elif tok.isidentifier():
                processed_tokens.append('<IDENTIFIER>')
            else:
                processed_tokens.append(tok)
                
        return processed_tokens
    
    def fit(self, code_samples: List[str]) -> 'CodeTokenizer':
        """
        Build vocabulary from code samples.
        
        Args:
            code_samples: List of code strings
            
        Returns:
            self
        """
        token_counts = Counter()
        
        for code in code_samples:
            tokens = self._tokenize_code(code)
            token_counts.update(tokens)
        
        # Keep most common tokens
        most_common = token_counts.most_common(self.vocab_size - len(self.vocab))
        
        for token, _ in most_common:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.fitted = True
        
        print(f"Vocabulary built with {len(self.vocab)} tokens")
        return self
    
    def encode(self, code: str) -> np.ndarray:
        """
        Encode code string to numerical sequence.
        
        Args:
            code: Python source code
            
        Returns:
            Numpy array of token IDs
        """
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        
        tokens = self._tokenize_code(code)
        
        # Convert to IDs
        ids = [self.vocab.get('<BOS>', 2)]
        for tok in tokens[:self.max_length - 2]:
            ids.append(self.vocab.get(tok, self.vocab['<UNK>']))
        ids.append(self.vocab.get('<EOS>', 3))
        
        # Pad to max_length
        while len(ids) < self.max_length:
            ids.append(self.vocab['<PAD>'])
        
        return np.array(ids[:self.max_length], dtype=np.int32)
    
    def encode_batch(self, code_samples: List[str]) -> np.ndarray:
        """
        Encode multiple code samples.
        
        Args:
            code_samples: List of code strings
            
        Returns:
            Numpy array of shape (batch_size, max_length)
        """
        return np.array([self.encode(code) for code in code_samples])
    
    def decode(self, ids: np.ndarray) -> str:
        """
        Decode token IDs back to tokens.
        
        Args:
            ids: Array of token IDs
            
        Returns:
            String of tokens (not original code)
        """
        tokens = []
        for id_ in ids:
            if id_ in self.reverse_vocab:
                tok = self.reverse_vocab[id_]
                if tok not in ['<PAD>', '<BOS>', '<EOS>']:
                    tokens.append(tok)
        return ' '.join(tokens)


class CodeFeatureExtractor:
    """Extract handcrafted features from Python code."""
    
    def __init__(self):
        self.feature_names = [
            'num_lines',
            'num_functions',
            'num_classes',
            'num_loops',
            'num_conditionals',
            'num_try_except',
            'num_imports',
            'avg_line_length',
            'max_line_length',
            'num_comments',
            'num_docstrings',
            'cyclomatic_complexity',
            'num_variables',
            'num_operators',
            'num_comparisons',
            'indentation_depth',
            'num_function_calls',
            'num_returns',
            'num_assertions',
            'code_density'
        ]
    
    def extract(self, code: str) -> np.ndarray:
        """
        Extract features from code.
        
        Args:
            code: Python source code
            
        Returns:
            Feature vector
        """
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        features = {
            'num_lines': len(lines),
            'num_functions': code.count('def '),
            'num_classes': code.count('class '),
            'num_loops': code.count('for ') + code.count('while '),
            'num_conditionals': code.count('if ') + code.count('elif ') + code.count('else:'),
            'num_try_except': code.count('try:') + code.count('except'),
            'num_imports': code.count('import '),
            'avg_line_length': np.mean([len(l) for l in lines]) if lines else 0,
            'max_line_length': max([len(l) for l in lines]) if lines else 0,
            'num_comments': sum(1 for l in lines if '#' in l),
            'num_docstrings': code.count('"""') // 2 + code.count("'''") // 2,
            'cyclomatic_complexity': self._estimate_complexity(code),
            'num_variables': len(re.findall(r'\b\w+\s*=\s*', code)),
            'num_operators': len(re.findall(r'[+\-*//%]', code)),
            'num_comparisons': len(re.findall(r'[<>=!]=?', code)),
            'indentation_depth': self._max_indentation(lines),
            'num_function_calls': len(re.findall(r'\w+\s*\(', code)),
            'num_returns': code.count('return '),
            'num_assertions': code.count('assert '),
            'code_density': len(non_empty_lines) / len(lines) if lines else 0
        }
        
        return np.array([features[name] for name in self.feature_names], dtype=np.float32)
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity."""
        decision_points = 0
        for keyword in ['if ', 'elif ', 'for ', 'while ', 'and ', 'or ', 'except']:
            decision_points += code.count(keyword)
        return decision_points + 1
    
    def _max_indentation(self, lines: List[str]) -> int:
        """Get maximum indentation depth."""
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                spaces = indent // 4  # Assuming 4-space indentation
                max_indent = max(max_indent, spaces)
        return max_indent
    
    def extract_batch(self, code_samples: List[str]) -> np.ndarray:
        """Extract features from multiple code samples."""
        return np.array([self.extract(code) for code in code_samples])


class HybridCodeEncoder:
    """Combine tokenization and handcrafted features."""
    
    def __init__(self, max_length: int = 256, vocab_size: int = 5000):
        self.tokenizer = CodeTokenizer(max_length=max_length, vocab_size=vocab_size)
        self.feature_extractor = CodeFeatureExtractor()
        
    def fit(self, code_samples: List[str]) -> 'HybridCodeEncoder':
        """Fit the tokenizer on code samples."""
        self.tokenizer.fit(code_samples)
        return self
    
    def encode(self, code: str) -> Dict[str, np.ndarray]:
        """
        Encode code to both token IDs and features.
        
        Returns:
            Dict with 'tokens' and 'features' keys
        """
        return {
            'tokens': self.tokenizer.encode(code),
            'features': self.feature_extractor.extract(code)
        }
    
    def encode_batch(self, code_samples: List[str]) -> Dict[str, np.ndarray]:
        """Encode multiple code samples."""
        return {
            'tokens': self.tokenizer.encode_batch(code_samples),
            'features': self.feature_extractor.extract_batch(code_samples)
        }
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.vocab)
    
    @property
    def num_features(self) -> int:
        return len(self.feature_extractor.feature_names)


if __name__ == "__main__":
    # Example usage
    sample_code = '''
def divide(a, b):
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

result = divide(10, 2)
print(result)
'''
    
    # Test tokenizer
    tokenizer = CodeTokenizer(max_length=128)
    tokenizer.fit([sample_code])
    
    encoded = tokenizer.encode(sample_code)
    print(f"Token IDs shape: {encoded.shape}")
    print(f"First 20 tokens: {encoded[:20]}")
    
    # Test feature extractor
    extractor = CodeFeatureExtractor()
    features = extractor.extract(sample_code)
    print(f"\nFeatures ({len(features)}):")
    for name, value in zip(extractor.feature_names, features):
        print(f"  {name}: {value}")
    
    # Test hybrid encoder
    hybrid = HybridCodeEncoder()
    hybrid.fit([sample_code])
    encoded = hybrid.encode(sample_code)
    print(f"\nHybrid encoding:")
    print(f"  Tokens shape: {encoded['tokens'].shape}")
    print(f"  Features shape: {encoded['features'].shape}")
