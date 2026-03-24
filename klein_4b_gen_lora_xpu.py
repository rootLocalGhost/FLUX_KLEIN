import os
import time
import gc
import json
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from einops import rearrange
from safetensors.torch import load_file as load_sft
from transformers import AutoModelForCausalLM, AutoTokenizer

from klein.model import Flux2, Klein4BParams
from klein.autoencoder import AutoEncoder, AutoEncoderParams
from klein.sampling import get_schedule, denoise, batched_prc_txt, batched_prc_img

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model", "FLUX.2-klein-4B")
TRANSFORMER_PATH = os.path.join(MODEL_DIR, "transformer_merged", "flux-2-klein-4b.safetensors")
VAE_PATH = os.path.join(MODEL_DIR, "autoencoder", "ae.safetensors")
TEXT_ENCODER_DIR = os.path.join(MODEL_DIR, "text_encoder")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")
LORAS_DIR = os.path.join(SCRIPT_DIR, "loras")

if not os.path.exists(TRANSFORMER_PATH):
    print("\n>> Model weights not found locally. Downloading from Hugging Face...")
    print(f">> Syncing with repository: rootlocalghost/FLUX.2-klein-4B")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="rootlocalghost/FLUX.2-klein-4B",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
else:
    print("\n>> Local model weights found! Skipping download check.")


class LocalQwen3Embedder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        print(f"   -> Loading Qwen3 from: {TEXT_ENCODER_DIR}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                TEXT_ENCODER_DIR,
                torch_dtype=torch.bfloat16,
                device_map=device,
                local_files_only=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        except OSError:
            print("\n>> Local text encoder not found. Downloading from rootlocalghost/FLUX.2-klein-4B...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="rootlocalghost/FLUX.2-klein-4B",
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                TEXT_ENCODER_DIR,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        self.max_length = 512

    @torch.no_grad()
    def forward(self, txt: list[str]):
        all_input_ids = []
        all_attention_masks = []
        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        out = torch.stack([output.hidden_states[k] for k in [9, 18, 27]], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")


def empty_cache_xpu():
    gc.collect()
    torch.xpu.empty_cache()


def apply_lora_to_model(model: nn.Module, lora_path: str | None, lora_alpha: float = 1.0):
    if not lora_path:
        print(">> No LoRA selected, skipping LoRA application.")
        return

    if not os.path.isfile(lora_path):
        print(f"WARNING: LoRA file not found at {lora_path}. Skipping LoRA.")
        return

    print(f"\n>> Applying LoRA weights from: {lora_path}")
    lora_sd = load_sft(lora_path, device="cpu")

    lora_groups = {}
    for k, v in lora_sd.items():
        if ".lora_down" in k:
            base = k.replace(".lora_down", "")
            lora_groups.setdefault(base, {})["down"] = v
        elif ".lora_up" in k:
            base = k.replace(".lora_up", "")
            lora_groups.setdefault(base, {})["up"] = v
        elif ".lora_alpha" in k:
            base = k.replace(".lora_alpha", "")
            lora_groups.setdefault(base, {})["alpha"] = float(v.item())

    applied = 0
    for base_key, group in lora_groups.items():
        if "up" not in group or "down" not in group:
            continue
        if not hasattr(model, base_key) and base_key not in model.state_dict():
            # skip if base param not found
            continue

        up = group["up"].to(torch.float32)
        down = group["down"].to(torch.float32)
        alpha = group.get("alpha", None)

        try:
            if base_key in model.state_dict():
                base_param = model.state_dict()[base_key]
            else:
                base_param = getattr(model, base_key)

            r = down.shape[0]
            delta = (up @ down)
            scale = float(alpha) / r if alpha is not None and r > 0 else 1.0
            delta = delta * lora_alpha * scale
            base_param.data = base_param.data.to(delta.device, dtype=base_param.dtype)
            base_param.data += delta.to(base_param.device, dtype=base_param.dtype)
            applied += 1
        except Exception as e:
            # shape mismatch or missing params; skip gracefully
            continue

    print(f"   -> Applied LoRA to {applied} modules")


def load_prompts_from_file():
    prompt_file = os.path.join(SCRIPT_DIR, "prompts", "prompts.txt")
    if not os.path.exists(prompt_file):
        prompt_file = os.path.join(SCRIPT_DIR, "prompts.txt")

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"prompts file not found at {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f.readlines()]
    prompts = [line for line in raw if line and not line.startswith("#")]
    if not prompts:
        raise ValueError(f"No valid prompts found in {prompt_file}")

    return prompts


def ask_num_steps():
    num_steps_input = input("Enter denoising steps [default 4]: ").strip() or "4"
    try:
        num_steps = int(num_steps_input)
        if num_steps < 1:
            raise ValueError
    except ValueError:
        print("Invalid steps entered. Defaulting to 4.")
        num_steps = 4
    return num_steps


def generate_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = 42, num_steps: int = 4, lora_path: str | None = None, lora_alpha: float = 1.0, output_path: str = None):
    device = "xpu"
    dtype = torch.bfloat16

    print(f"\n--- Starting LoRA Generation: '{prompt}' (seed={seed}, steps={num_steps}) ---")

    print("\n>> [1/3] Loading Text Encoder (Qwen3-4B)...")
    text_encoder = LocalQwen3Embedder(device="cpu")
    text_encoder.model.to(device)

    print(">> Generating text embeddings...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        txt_hidden_states = text_encoder([prompt])
        txt, txt_ids = batched_prc_txt(txt_hidden_states)
        txt = txt.to(device, dtype=dtype)
        txt_ids = txt_ids.to(device)

    del text_encoder
    empty_cache_xpu()

    print(">> [2/3] Loading FLUX.2 Transformer + LoRA...")
    with torch.device("meta"):
        model = Flux2(Klein4BParams()).to(dtype)
    sd = load_sft(TRANSFORMER_PATH, device="cpu")
    model.load_state_dict(sd, strict=True, assign=True)

    apply_lora_to_model(model, lora_path, lora_alpha=lora_alpha)
    model = model.to(device)

    print(">> Preparing initial noise latents...")
    torch.manual_seed(seed)
    h_latent, w_latent = height // 16, width // 16
    img = torch.randn(1, 128, h_latent, w_latent, dtype=dtype, device=device)

    img_flat, img_ids = batched_prc_img(img)
    img_flat = img_flat.to(device, dtype=dtype)
    img_ids = img_ids.to(device)

    guidance = 1.0
    timesteps = get_schedule(num_steps, img_flat.shape[1])

    print(f">> Denoising for {num_steps} steps...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        img_flat = denoise(model, img_flat, img_ids, txt, txt_ids, timesteps, guidance=guidance)

    del model
    del sd
    empty_cache_xpu()

    print(">> [3/3] Loading Autoencoder...")
    img_latent = rearrange(img_flat, "b (h w) c -> b c h w", h=h_latent, w=w_latent)

    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams()).to(dtype)
    sd_ae = load_sft(VAE_PATH, device=device)
    ae.load_state_dict(sd_ae, strict=True, assign=True)
    ae = ae.to(device)

    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        decoded = ae.decode(img_latent)

    del ae
    del sd_ae
    empty_cache_xpu()

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()

    image = Image.fromarray((decoded[0] * 255).astype(np.uint8))
    output_dir = os.path.join(SCRIPT_DIR, 'outputs', 'gen_lora')
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(output_dir, f"gen_lora_{int(time.time())}_{seed}.png")
    else:
        if not os.path.isabs(output_path):
            output_path = os.path.join(output_dir, output_path)

    image.save(output_path)

    metadata = {
        'prompt': prompt,
        'width': width,
        'height': height,
        'seed': seed,
        'num_steps': num_steps,
        'device': device,
        'dtype': str(dtype),
    }
    metadata_path = os.path.splitext(output_path)[0] + '.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n>> Success! Image saved to: {output_path}")
    print(f">> Metadata saved to: {metadata_path}")


def list_available_loras():
    if not os.path.exists(LORAS_DIR):
        return []
    return sorted(
        [f for f in os.listdir(LORAS_DIR) if f.lower().endswith(".safetensors")],
        key=str.lower,
    )


def run_flow():
    if not os.path.exists(TRANSFORMER_PATH):
        print(f"ERROR: Cannot find Transformer at {TRANSFORMER_PATH}")
        return

    loras = list_available_loras()
    lora_path = None

    if not loras:
        print(">> No LoRA files found in 'loras/' - generating without LoRA.")
    else:
        print(">> Available LoRA files:")
        print("  0) [no LoRA]")
        for idx, filename in enumerate(loras, start=1):
            print(f"  {idx}) {filename}")

        choice = input("Select LoRA by number [0 for none]: ").strip() or '0'
        try:
            choice_num = int(choice)
        except ValueError:
            choice_num = 0

        if choice_num > 0 and choice_num <= len(loras):
            lora_path = os.path.join(LORAS_DIR, loras[choice_num - 1])
            print(f">> Selected LoRA: {lora_path}")
        else:
            print(">> No LoRA selected. Generating without LoRA.")

    lora_alpha_input = input("Enter LoRA alpha scaling [default 1.0]: ").strip() or '1.0'
    try:
        lora_alpha = float(lora_alpha_input)
    except ValueError:
        print("Invalid alpha value, using 1.0")
        lora_alpha = 1.0

    num_steps = ask_num_steps()
    user_prompt = input('Enter prompt (leave blank to use prompts/prompts.txt): ').strip()

    if not user_prompt:
        prompts = load_prompts_from_file()
        if len(prompts) == 1:
            print("Only one prompt loaded.")
            count_str = input('How many images to generate? [default 1]: ').strip() or '1'
            try:
                count = int(count_str)
                if count < 1:
                    raise ValueError
            except ValueError:
                print('Invalid number, defaulting to 1')
                count = 1

            for _ in range(count):
                seed = np.random.randint(0, 2**31 - 1)
                generate_image(prompts[0], seed=seed, num_steps=num_steps, lora_path=lora_path, lora_alpha=lora_alpha)
            return

        for prompt in prompts:
            seed = np.random.randint(0, 2**31 - 1)
            generate_image(prompt, seed=seed, num_steps=num_steps, lora_path=lora_path, lora_alpha=lora_alpha)
        return

    count_str = input('How many images to generate? [default 1]: ').strip() or '1'
    try:
        count = int(count_str)
        if count < 1:
            raise ValueError
    except ValueError:
        print('Invalid number, defaulting to 1')
        count = 1

    if count == 1:
        seed_input = input('Enter seed (leave blank for random): ').strip()
        try:
            seed = int(seed_input) if seed_input else np.random.randint(0, 2**31 - 1)
        except ValueError:
            print('Invalid seed, using random')
            seed = np.random.randint(0, 2**31 - 1)
        generate_image(user_prompt, seed=seed, num_steps=num_steps, lora_path=lora_path, lora_alpha=lora_alpha)
    else:
        for _ in range(count):
            seed = np.random.randint(0, 2**31 - 1)
            generate_image(user_prompt, seed=seed, num_steps=num_steps, lora_path=lora_path, lora_alpha=lora_alpha)


if __name__ == "__main__":
    run_flow()