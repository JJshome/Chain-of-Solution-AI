# Chain of Solution AI System
# 통합 문제해결 방법론을 활용한 인공지능 시스템 및 그 방법

<div align="center">
  <img src="doc/images/cos_system_architecture.svg" alt="Chain of Solution System Architecture" width="800">
</div>

## 🚀 Overview

Chain of Solution (CoS)은 다양한 문제 해결 방법론(TRIZ, 마인드맵, 디자인 씽킹, OS매트릭스, 6 시그마 등)을 대규모 언어 모델과 결합하여 복잡한 기술적, 비즈니스적 문제를 효과적으로 해결할 수 있는 AI 기반 시스템입니다.

### 주요 특징
- **TRIZ60 원리**: 기존 40개 원리에 20개의 현대적 원리를 추가한 확장된 문제 해결 이론
- **Su-Field 100 분석**: 복잡한 시스템 내 상호작용을 포함한 최신 기술적 문제 해결 방식
- **대규모 언어 모델 통합**: Llama3.1-8B/70B 모델을 활용한 자연어 처리 및 추론
- **동적 재구성**: 실시간으로 시스템 구조를 최적화하는 AI 피드백 루프
- **다중 도메인 지원**: 나노기술, 자율주행, 생명공학, 양자 컴퓨팅 등 첨단 분야 적용

## 📋 System Architecture

```
Chain of Solution AI System
├── 문제 입력부 (100)
├── TRIZ60 원리 적용부 (110)
├── Su-Field 분석부 (120)
├── 100가지 표준해결책 적용부 (130)
├── 대규모 언어 모델 처리부 (140)
│   └── Llama3.1-8B/70B
├── LoRA 기반 파인튜닝부 (150)
├── 동적 재구성부 (160)
├── AI 피드백 루프부 (170)
├── 다중 상태 제어부 (180)
├── 비선형 시스템 분석부 (190)
├── 확률적 최적화부 (200)
├── 다중 도메인 적용부 (210)
├── 실시간 최적화부 (220)
└── 자가 진단부 (230)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM (32GB+ recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/JJshome/Chain-of-Solution-AI.git
cd Chain-of-Solution-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download LLM models (optional for full functionality)
python scripts/download_models.py
```

## 🚀 Quick Start

### Basic Usage

```python
from src import ChainOfSolution

# Initialize system
cos = ChainOfSolution({
    'model_type': 'llama3.1',
    'model_size': '8B',
    'use_triz60': True,
    'use_su_field_100': True,
    'enable_dynamic_reconfiguration': True,
    'enable_ai_feedback_loop': True
})

# Define problem
problem = {
    'description': '전기화학 바이오센서에서 간섭을 보상하는 두 전극 테스트 스트립 설계',
    'domain': 'electrochemical_biosensor',
    'constraints': ['비용 효율성', '소형화', '정확도 향상']
}

# Solve problem
solution = cos.solve_problem(problem)

# Display results
print(f"Solution Summary: {solution['summary']}")
print(f"Applied TRIZ Principles: {solution['triz_principles']}")
print(f"Su-Field Analysis: {solution['su_field_analysis']}")
print(f"Recommendations: {solution['recommendations']}")
```

## 📚 Core Components

### 1. TRIZ60 Principles
확장된 60가지 발명 원리를 포함:
- 전통적 40개 원리 (1-40)
- 현대적 20개 원리 (41-60): 동적 재구성, 에너지 재분배, 비선형 상호작용 등

### 2. Su-Field 100 Analysis
100가지 표준해결책으로 확장:
- 기존 76가지 표준해결책
- 추가 24가지 현대 기술 솔루션 (양자 컴퓨팅, 나노기술, AI 피드백 등)

### 3. Multi-Domain Applications
다양한 산업 분야 적용:
- 제조업: 6시그마 품질관리 통합
- R&D: 기술전략, 특허맵, 연구개발 Life Cycle
- 비즈니스: Product Life Cycle, Supply Chain, ERP, 마케팅

### 4. Advanced Features
- **LoRA Fine-tuning**: 도메인 특화 최적화
- **Dynamic Reconfiguration**: 실시간 시스템 재구성
- **Probabilistic Optimization**: 불확실성 하에서의 최적화
- **Non-linear System Analysis**: 복잡계 분석 및 예측

## 📖 Documentation

### Examples

다양한 실시예 제공:
- [전기화학 바이오센서 설계](examples/electrochemical_biosensor.py)
- [6시그마 품질관리 적용](examples/six_sigma_quality.py)
- [기술전략 수립](examples/technology_strategy.py)
- [제품 수명주기 관리](examples/product_lifecycle.py)

### API Reference

상세 API 문서는 [docs/api](docs/api/README.md)를 참조하세요.

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance benchmarks
python scripts/benchmark.py
```

## 🤝 Contributing

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- Patent Application: "통합 문제해결 방법론을 활용한 인공지능 시스템 및 그 방법"
- [TRIZ Official Website](https://www.triz.org)
- [Llama Model Documentation](https://github.com/facebookresearch/llama)

## 📞 Contact

- Author: Jee Hwan Jang
- Email: [contact email]
- Organization: Sungkyunkwan University & Ucaretron Inc.

---

<div align="center">
  <sub>Built with ❤️ using Chain of Solution Framework</sub>
</div>
