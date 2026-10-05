# Artifacts

| File | What it is |
|---|---|
| `core_adapter_round1.safetensors` (+ `_config.json`) | LoRA (r=16, alpha=32) of core round 1 on top of `Qwen/Qwen2.5-1.5B-Instruct`. Merging it gives D1, the only defender that changed in the core loop (D2 = D3 = D1, their round adapters are zero-update). |
| `bonus_attacker_round3.safetensors` (+ `_config.json`) | LoRA of the co-trained RL attacker after bonus round 3, the final one (base model: `Qwen/Qwen2.5-1.5B-Instruct`). |

Load with PEFT: `PeftModel.from_pretrained(base, "<folder containing adapter_config.json + adapter_model.safetensors>")`
(rename the `.safetensors` file to `adapter_model.safetensors` and the config to `adapter_config.json` first).

Hugging Face upload (adapters + the redacted success pool) : **TODO - add link before submission.**
The raw logs / success pools contain harmful text and are deliberately not in this repo.
