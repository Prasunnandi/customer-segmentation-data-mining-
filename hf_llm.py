"""
Local LangChain-compatible LLM using transformers text-generation pipeline.
Uses distilgpt2 (~82MB) — works on all transformers versions, fully offline.
"""
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

_PIPELINE_CACHE: dict = {}

def _get_pipeline(model_id: str, max_new_tokens: int):
    if model_id not in _PIPELINE_CACHE:
        from transformers import pipeline
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1

        _PIPELINE_CACHE[model_id] = pipeline(
            "text-generation",
            model=model_id,
            device=device,
            pad_token_id=50256,
        )
    return _PIPELINE_CACHE[model_id]

class HFInferenceLLM(LLM):
    model_id: str = "distilgpt2"
    max_new_tokens: int = 200

    @property
    def _llm_type(self) -> str:
        return "hf_local_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            pipe = _get_pipeline(self.model_id, self.max_new_tokens)
            truncated = prompt[:800]
            result = pipe(truncated, max_new_tokens=self.max_new_tokens, do_sample=False)
            if result and isinstance(result, list):
                full_text = result[0].get("generated_text", "")
                answer = full_text[len(truncated):].strip()
                return answer if answer else full_text.strip()
            return str(result)
        except Exception as e:
            return f"[Local LLM Error]: {str(e)}"
