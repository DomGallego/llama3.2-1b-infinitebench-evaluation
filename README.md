# Meta Llama 3.2 1B Performance on Long-Context Narrative Reasoning: An InfiniteBench Evaluation

## Project Overview

This project investigates the capabilities of the **Meta Llama 3.2 1B Instruct** model in handling and reasoning over extremely long narrative contexts.  Leveraging the challenging **InfiniteBench** benchmark, specifically the `longbook_choice_eng` split of the `En.MC` task, this evaluation delves into the model's performance on complex multiple-choice questions derived from lengthy literary works. InfiniteBench is renowned for its rigorous demands, pushing Language Models (LLMs) to their limits with contexts exceeding 100,000 tokens. This project specifically tests the Meta Llama 3.2 1B model's ability to process contexts up to its 128k token limit and extract relevant information for accurate question answering.

This initiative is driven by the growing importance of Long Context Large Language Models, and is inspired by the cutting-edge research presented in:

> **∞Bench: Extending Long Context Evaluation Beyond 100K Tokens**
> Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, Maosong Sun
> [arXiv:2402.13718](https://arxiv.org/abs/2402.13718)

The primary goal is to rigorously assess the model's accuracy in answering questions that require deep comprehension of extended narratives, while also exploring the impact of various prompt engineering strategies and hyperparameter configurations on performance. This project demonstrates a practical approach to benchmarking and fine-tuning LLMs for real-world applications requiring long-context understanding.

## Methodology


To conduct this evaluation, a systematic methodology was implemented, encompassing the following key steps:

* **Model Integration:** The state-of-the-art **Meta Llama 3.2 1B Instruct** model was seamlessly integrated from the Hugging Face Hub ([meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)).
* **Benchmark Dataset Utilization:** The industry-standard **InfiniteBench** dataset was employed, sourced directly from Hugging Face Datasets ([xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)). The evaluation focused on the `longbook_choice_eng` split within the `En.MC` task, providing a targeted assessment of long-context narrative understanding.
* **Prompt Engineering:**  A robust prompt template, carefully adapted from the research paper "∞Bench: Extending Long Context Evaluation Beyond 100K Tokens," was utilized. Iterative refinements and variations were implemented across different evaluation versions to optimize performance.
* **Context Management:**  To address the 128k token context window limitation of the Meta Llama 3.2 1B model, a strategic truncation technique, mirroring the approach detailed in the ∞Bench paper, was implemented. This method intelligently preserves crucial information from both the beginning and end of the extensive input contexts.
* **Performance Measurement:** Accuracy was chosen as the core metric to objectively quantify the model's success in selecting the correct answer from a set of multiple-choice options, providing a clear and interpretable measure of performance.
* **Configuration Variants:**  To gain a comprehensive understanding of the model's behavior, multiple evaluation versions were executed, systematically varying prompt templates and key hyperparameters, including temperature and top\_p, allowing for a nuanced analysis of their impact.

Context truncation was achieved using a method that intelligently retained segments from the start and end of the input text, ensuring crucial information was preserved:

```python
return prompt_tokens[:max_length // 2] + prompt_tokens[-max_length // 2:]
```




## Results and Analysis


The project yielded insightful performance results across different configurations:

| Version | Description                                                                       | Result   |
|---------|-----------------------------------------------------------------------------------|----------|
| V1      | Benchmark-standard prompt, temperature=0.5, top\_p=0.5                              | 34.93%   |
| V2      | Modified prompt template, temperature=0.5, top\_p=0.5                              | 33.62%   |
| V3      | Modified prompt template, temperature=0.5, top\_p=0.5                              | 30.13%   |
| V4      | Modified prompt template (XML-style), temperature=0.5, top\_p=0.5                | 27.95%   |
| **V5**    | Benchmark-standard prompt, temperature=1, top\_p=1                              | **36.24%** |

**Evaluation Time:** Completing the evaluation across all 229 data rows required approximately 40 minutes.

**Benchmark Comparison:** Meta AI's reported benchmark score for InfiniteBench/En.MC (128k) is 38. Version 5 of this project achieved a close accuracy of 36.24%, demonstrating comparable performance to Meta's reported figures.

**Key Findings:** The highest accuracy was achieved with Version 5, utilizing the benchmark-standard prompt and increased sampling hyperparameters (temperature=1, top\_p=1). This suggests that for complex long-context reasoning tasks, encouraging the model to explore a broader range of potential outputs, enabled by higher temperature and top\_p settings, can lead to improved performance. This highlights the importance of hyperparameter tuning in optimizing LLM performance for specific tasks and benchmarks.

## Code Implementation

The complete project codebase, including data loading, model interaction, evaluation functions, and result analysis, is available as a Jupyter Notebook: `llama3.2-1b-infinitebench-evaluation.ipynb`.

Key components of the code include:

* **Library Setup:** Importing essential libraries such as `torch`, `transformers`, `datasets`, `pandas`, and `tqdm` to facilitate model interaction, data handling, and performance tracking.

* **Model and Tokenizer Initialization:** Loading the Meta Llama 3.2 1B Instruct model and its corresponding tokenizer directly from the Hugging Face model hub.

* **Benchmark Function Design:** Implementing the `benchmark_evaluation` function to encapsulate the core evaluation workflow, including prompt generation, model inference, answer parsing, and accuracy calculation.

* **Utility Functions:** Development of helper functions such as `parse_method_v1`, `parse_method_v2` (for output parsing), `is_similar` (for answer similarity assessment), and `truncation_keep_ends` (for context truncation).

* **Prompt Template Variations:** Defining and managing multiple prompt templates (`prompt_template_v1` to `prompt_template_v4`) to explore different prompting strategies.

* **Evaluation Execution and Output:** Running the benchmark across Versions 1-5, systematically recording predictions and performance metrics, and exporting detailed results to CSV files for in-depth analysis.

## Resources

* **InfiniteBench Dataset:** [https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)

* **Meta Llama 3.2 1B Instruct Model:** [https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## License

This project is released under the MIT License.