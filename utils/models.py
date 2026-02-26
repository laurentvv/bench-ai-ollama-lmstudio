import os
import asyncio
import random
from typing import Optional

import time
from litellm import acompletion

from dotenv import load_dotenv
load_dotenv()

from openai import RateLimitError


MODEL_MAP = {
    "qwen3:14b": "qwen3:14b",
    "llama3": "llama3",
    "llama2": "llama2",
    "codellama": "codellama",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "llava": "llava",
    "gemma": "gemma",
    "phi3": "phi3",
    "qwen2": "qwen2",
    "qwen3:14b-q4_K_M": "qwen3:14b-q4_K_M",
    "command-r": "command-r",
    "command-r-plus": "command-r-plus",
}

EXPONENTIAL_BASE = 2    


class MajorityVoteModel:
    def __init__(self, model, num_responses: int = 3):
        self.model = model
        self.num_responses = num_responses
    
    async def predict(self, prompt: str):
        tasks = [self.model.predict(prompt) for _ in range(self.num_responses)]
        return await asyncio.gather(*tasks)


class LiteLLMModel:
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, temp: float = 0.7, max_tokens: int = 2048, top_p: float = 0.95, max_retries: int = 3):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temp = temp
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries

        # Vérifier si le modèle est dans MODEL_MAP ou s'il commence déjà par 'ollama/'
        if self.model_name not in MODEL_MAP and not self.model_name.startswith('ollama/'):
            # Ajouter automatiquement le modèle au dictionnaire
            MODEL_MAP[self.model_name] = self.model_name
            print(f"Modèle ajouté automatiquement: {self.model_name}")

        # Préfixer avec 'ollama/' seulement si ce n'est pas déjà fait
        if not self.model_name.startswith('ollama/'):
            self.model_name = f"ollama/{self.model_name}"

    
    async def predict(self, prompt: str):
        delay = 2

        for i in range(self.max_retries):
            try:
                messages = []
                if self.system_prompt is not None:
                    messages.append({
                        "role": "system",
                        "content": self.system_prompt
                    })
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                response = await acompletion(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )

                if response.choices[0].message.content is not None:
                    return response.choices[0].message.content
                else:
                    print(response)
                    return "[MODEL_LIMIT_REACHED] No content in response"
            except RateLimitError as e:
                delay *= EXPONENTIAL_BASE * (1 + random.random())
                print(
                    f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...", e
                )
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                print(f"Error in retry {i+1}, retrying...", e)
                if i == self.max_retries - 1:  # Dernière tentative
                    if "maximum context length" in str(e).lower() or "token limit" in str(e).lower():
                        return f"[MODEL_LIMIT_REACHED] {str(e)}"
                continue

        return "[MODEL_LIMIT_REACHED] Failed to get response after max retries"
