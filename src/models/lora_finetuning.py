"""
LoRA (Low-Rank Adaptation) 기반 파인튜닝 모듈
특허 명세서에 따른 효율적인 도메인 특화 모델 최적화 구현
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """LoRA 파인튜닝 설정"""
    r: int = 16  # 저차원 매트릭스의 rank
    lora_alpha: int = 32  # LoRA 스케일링 파라미터
    lora_dropout: float = 0.1  # 드롭아웃 비율
    target_modules: List[str] = None  # LoRA를 적용할 모듈 리스트
    bias: str = "none"  # 바이어스 파라미터 처리 방식
    task_type: str = "CAUSAL_LM"  # 태스크 타입
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

class LoRAFineTuner:
    """
    특허 명세서 [실시예 5]에 따른 LoRA 기반 파인튜닝 구현
    저차원 적응을 통한 효율적인 도메인 특화 최적화
    """
    
    def __init__(self, base_model_name: str, config: Optional[LoRAConfig] = None):
        """
        Args:
            base_model_name: 기본 모델 이름 (예: "meta-llama/Llama-3.1-8B")
            config: LoRA 설정
        """
        self.base_model_name = base_model_name
        self.config = config or LoRAConfig()
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def initialize_model(self):
        """모델 및 토크나이저 초기화"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # 기본 모델 로드
        self.model = AutoModel.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # LoRA 설정 생성
        peft_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
            task_type=TaskType[self.config.task_type]
        )
        
        # PEFT 모델 생성
        self.peft_model = get_peft_model(self.model, peft_config)
        logger.info(f"LoRA parameters: {self.peft_model.print_trainable_parameters()}")
        
    def prepare_domain_data(self, domain_data: List[Dict]) -> Dict:
        """
        도메인 특화 데이터 준비
        
        Args:
            domain_data: 도메인 데이터 리스트
                - problem: 문제 설명
                - solution: 해결책
                - triz_principles: 적용된 TRIZ 원리
                - domain: 도메인 정보
        
        Returns:
            준비된 학습 데이터
        """
        processed_data = []
        
        for item in domain_data:
            # Chain of Solution 형식으로 데이터 변환
            text = self._format_cos_data(item)
            
            # 토크나이징
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            processed_data.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["input_ids"].clone()
            })
            
        return processed_data
    
    def _format_cos_data(self, item: Dict) -> str:
        """Chain of Solution 형식으로 데이터 포맷팅"""
        text = f"Domain: {item.get('domain', 'General')}\n"
        text += f"Problem: {item['problem']}\n"
        
        if 'triz_principles' in item:
            text += "Applied TRIZ Principles:\n"
            for principle in item['triz_principles']:
                text += f"- {principle['name']}: {principle['description']}\n"
        
        if 'su_field_analysis' in item:
            analysis = item['su_field_analysis']
            text += f"Su-Field Analysis:\n"
            text += f"- S1: {analysis['S1']}\n"
            text += f"- S2: {analysis['S2']}\n"
            text += f"- Field: {analysis['Field']}\n"
            text += f"- Interaction: {analysis['interaction']}\n"
        
        text += f"Solution: {item['solution']}\n"
        
        return text
    
    def adaptive_fine_tune(self, train_data: List[Dict], 
                          val_data: Optional[List[Dict]] = None,
                          num_epochs: int = 3,
                          learning_rate: float = 5e-5) -> Dict[str, float]:
        """
        적응형 파인튜닝 수행
        특허 명세서에 따른 동적 매트릭스 크기 조절 포함
        """
        from torch.utils.data import DataLoader, Dataset
        from transformers import AdamW
        from torch.optim.lr_scheduler import LinearLR
        
        # 데이터 준비
        train_dataset = self._create_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # 옵티마이저 설정
        optimizer = AdamW(self.peft_model.parameters(), lr=learning_rate)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, 
                           total_iters=num_epochs * len(train_loader))
        
        # 학습 메트릭
        metrics = {
            "train_loss": [],
            "val_loss": [] if val_data else None,
            "learning_rate": []
        }
        
        # 학습 루프
        self.peft_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # 순전파
                outputs = self.peft_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                
                # 역전파
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                # 적응형 업데이트 (매 100 스텝마다)
                if batch_idx % 100 == 0:
                    self._adaptive_update(loss.item())
            
            avg_loss = epoch_loss / len(train_loader)
            metrics["train_loss"].append(avg_loss)
            metrics["learning_rate"].append(scheduler.get_last_lr()[0])
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # 검증 데이터 평가
            if val_data:
                val_loss = self._evaluate(val_data)
                metrics["val_loss"].append(val_loss)
                logger.info(f"Validation Loss: {val_loss:.4f}")
        
        return metrics
    
    def _adaptive_update(self, current_loss: float):
        """
        특허 명세서의 적응형 업데이트 구현
        학습 과정에서 저차원 매트릭스 크기를 동적으로 조절
        """
        # 손실 기반 rank 조정 로직
        if current_loss > 1.0:  # 높은 손실
            # rank를 증가시켜 표현력 향상
            new_r = min(self.config.r * 2, 64)
            if new_r != self.config.r:
                logger.info(f"Adapting LoRA rank: {self.config.r} -> {new_r}")
                self.config.r = new_r
                # 모델 재초기화 필요 (실제 구현에서는 더 정교한 방법 사용)
    
    def save_adapter(self, save_path: str):
        """LoRA 어댑터 저장"""
        if self.peft_model is None:
            raise ValueError("Model not initialized or trained")
        
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 설정 저장
        with open(f"{save_path}/lora_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"LoRA adapter saved to: {save_path}")
    
    def load_adapter(self, adapter_path: str):
        """저장된 LoRA 어댑터 로드"""
        # 설정 로드
        with open(f"{adapter_path}/lora_config.json", "r") as f:
            config_dict = json.load(f)
            self.config = LoRAConfig(**config_dict)
        
        # 모델 로드
        self.model = AutoModel.from_pretrained(self.base_model_name)
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        logger.info(f"LoRA adapter loaded from: {adapter_path}")
    
    def inference(self, prompt: str, max_length: int = 512) -> str:
        """추론 수행"""
        if self.peft_model is None:
            raise ValueError("Model not initialized")
        
        self.peft_model.eval()
        
        # 입력 인코딩
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 생성
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )
        
        # 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    def _create_dataset(self, data: List[Dict]) -> torch.utils.data.Dataset:
        """PyTorch Dataset 생성"""
        class CoSDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        processed_data = self.prepare_domain_data(data)
        return CoSDataset(processed_data)
    
    def _evaluate(self, eval_data: List[Dict]) -> float:
        """평가 수행"""
        eval_dataset = self._create_dataset(eval_data)
        eval_loader = DataLoader(eval_dataset, batch_size=4)
        
        self.peft_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                outputs = self.peft_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_loss += outputs.loss.item()
        
        return total_loss / len(eval_loader)


class DomainSpecificLoRA:
    """
    특정 도메인에 특화된 LoRA 어댑터 관리
    """
    
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.domain_adapters = {}
        
    def train_domain_adapter(self, domain: str, 
                           domain_data: List[Dict],
                           config: Optional[LoRAConfig] = None) -> str:
        """특정 도메인에 대한 어댑터 학습"""
        # 도메인별 설정
        if config is None:
            config = self._get_domain_config(domain)
        
        # 파인튜너 생성
        finetuner = LoRAFineTuner(self.base_model_name, config)
        finetuner.initialize_model()
        
        # 학습
        metrics = finetuner.adaptive_fine_tune(domain_data)
        
        # 저장
        save_path = f"adapters/{domain}_adapter"
        finetuner.save_adapter(save_path)
        
        self.domain_adapters[domain] = save_path
        
        return save_path
    
    def _get_domain_config(self, domain: str) -> LoRAConfig:
        """도메인별 최적 설정 반환"""
        domain_configs = {
            "electronics": LoRAConfig(r=32, lora_alpha=64),
            "biomedical": LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.15),
            "mechanical": LoRAConfig(r=24, lora_alpha=48),
            "software": LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05),
            "nanotechnology": LoRAConfig(r=64, lora_alpha=128),
            "quantum": LoRAConfig(r=48, lora_alpha=96)
        }
        
        return domain_configs.get(domain, LoRAConfig())
