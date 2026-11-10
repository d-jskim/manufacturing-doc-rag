"""텍스트의 토큰 수를 센다. 생성 모델과 임베딩 모델 기준을 같은 방식으로 사용한다."""


import tiktoken
from transformers import AutoTokenizer


class TiktokenCounter:
    """생성 모델 기준. 모델을 인식하지 못하면 근사 인코딩을 사용한다."""
    
    def __init__(self, model: str, fallback_encoding: str):
        try:
            self._encoding = tiktoken.encoding_for_model(model)
            self.is_approximate = False
        except KeyError:
            self._encoding = tiktoken.get_encoding(fallback_encoding)
            self.is_approximate = True
        self.name = self._encoding.name
        
        
    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))
    
    
class HuggingFaceCounter:
    """임베딩 모델 기준. 실제 입력과 같도록 접두어와 특수 토큰을 포함해 센다."""
    
    
    def __init__(self, model_name: str, prefix: str=""):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prefix = prefix
        self.name = model_name
        self.is_approximate = False
        
    
    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(self.prefix + text, add_special_tokens=True, verbose=False))


    def count_body(self, text: str) -> int:
        """본문만 센다(접두어·특수 토큰 제외). 청크 분할의 길이 기준."""
        return len(self._tokenizer.encode(text, add_special_tokens=False, verbose=False))


    def count_overhead(self) -> int:
        """본문 없이 접두어와 특수 토큰만 넣었을 때의 토큰 수."""
        return self.count("")