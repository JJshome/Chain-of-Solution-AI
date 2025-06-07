# Core Functionality and Setup for Chain of Solution AI System

This document provides an overview of the core problem-solving functionality demonstrated in the Quick Start guide and outlines the installation prerequisites and setup steps for the Chain of Solution (CoS) AI System, based on the root `README.md`.

## Core Functionality: Problem-Solving Process

The "Quick Start" example illustrates the primary workflow for utilizing the CoS system to address problems:

1.  **Initialization of the System:**
    *   The process begins by importing the `ChainOfSolution` class from the `src` module.
    *   An instance of the `ChainOfSolution` system is then created. During initialization, various configurations can be passed, such_as:
        *   `model_type` (e.g., 'llama3.1')
        *   `model_size` (e.g., '8B')
        *   Flags to enable or disable specific methodologies like `use_triz60`, `use_su_field_100`, `enable_dynamic_reconfiguration`, and `enable_ai_feedback_loop`.

2.  **Defining the Problem:**
    *   The user defines the problem as a dictionary. This dictionary includes:
        *   `description`: A natural language statement of the problem (e.g., "전기화학 바이오센서에서 간섭을 보상하는 두 전극 테스트 스트립 설계" - Designing a two-electrode test strip to compensate for interference in an electrochemical biosensor).
        *   `domain`: The specific field or area the problem belongs to (e.g., 'electrochemical_biosensor').
        *   `constraints`: A list of conditions or limitations that the solution should adhere to (e.g., '비용 효율성' - cost-effectiveness, '소형화' - miniaturization, '정확도 향상' - accuracy improvement).

3.  **Solving the Problem:**
    *   The `solve_problem()` method of the CoS instance is called, passing the problem dictionary as an argument.
    *   The system then processes this input, presumably applying its integrated methodologies (like TRIZ60, Su-Field analysis) and leveraging the configured LLM to generate a solution.

4.  **Getting Results:**
    *   The `solve_problem()` method returns a dictionary containing the solution and related analysis.
    *   The example demonstrates accessing and printing:
        *   `summary`: A summary of the proposed solution.
        *   `triz_principles`: TRIZ principles that were applied.
        *   `su_field_analysis`: Results from the Su-Field analysis.
        *   `recommendations`: Specific actionable recommendations.

This flow showcases how CoS takes a structured problem definition and uses its AI-driven problem-solving engine to deliver a multifaceted solution.

## Installation

### Prerequisites

To successfully install and run the Chain of Solution AI System, the following are required:

*   **Python:** Version 3.8 or higher.
*   **CUDA:** Version 11.8 or higher (this is for GPU support, implying that CPU-only operation might be possible but slower for LLM tasks).
*   **RAM:** A minimum of 16GB is required, with 32GB or more recommended for optimal performance, especially when using larger LLM models.

### Main Setup Steps

The following steps outline the process to get the system running:

1.  **Clone Repository:**
    *   Obtain the source code by cloning the GitHub repository:
        ```bash
        git clone https://github.com/JJshome/Chain-of-Solution-AI.git
        cd Chain-of-Solution-AI
        ```

2.  **Create Virtual Environment:**
    *   It is recommended to create and activate a Python virtual environment to manage dependencies:
        ```bash
        python -m venv venv
        source venv/bin/activate  # For Linux/macOS
        # On Windows: venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    *   Install the necessary Python packages listed in `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Download LLM Models (Optional):**
    *   For full functionality, especially leveraging the advanced AI capabilities, the LLM models need to be downloaded. A script is provided for this:
        ```bash
        python scripts/download_models.py
        ```
    *   This step is marked as optional, suggesting the system might operate in a limited mode without local models, or it might be configured to use API-based models.
