import os
import gc
import json
import time
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from einops import rearrange
from safetensors.torch import load_file as load_sft
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the core math and architecture components from the BFL reference code
from klein.model_fp8 import Flux2, Klein4BParams
from klein.autoencoder import AutoEncoder, AutoEncoderParams
from klein.sampling import get_schedule, denoise, batched_prc_txt, batched_prc_img

# ==========================================
# 1. AUTOMATIC PATH RESOLUTION & DOWNLOAD
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model", "FLUX.2-klein-4B")

TRANSFORMER_PATH = os.path.join(MODEL_DIR, "flux-2-klein-4b-fp8.safetensors")
VAE_PATH = os.path.join(MODEL_DIR, "autoencoder", "ae.safetensors")
TEXT_ENCODER_DIR = os.path.join(MODEL_DIR, "text_encoder")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")

# Only ping Hugging Face if the main model file is completely missing
if not os.path.exists(TRANSFORMER_PATH):
    print("\n>> Model weights not found locally. Downloading from Hugging Face...")
    print(f">> Syncing with repository: rootlocalghost/FLUX.2-klein-4B")
    snapshot_download(
        repo_id="rootlocalghost/FLUX.2-klein-4B",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,  # Forces real files instead of weird Windows cache shortcuts
    )
else:
    print("\n>> Local model weights found! Skipping download check.")


# ==========================================
# 2. CUSTOM LOCAL TEXT ENCODER
# ==========================================
class LocalQwen3Embedder(nn.Module):
    """Bypasses the original HF hub downloader to use your local folders."""

    def __init__(self, device="cpu"):
        super().__init__()
        print(f"   -> Loading Qwen3 from: {TEXT_ENCODER_DIR}")
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
        # FLUX.2 Klein specifically uses layers 9, 18, and 27 from Qwen3
        out = torch.stack([output.hidden_states[k] for k in [9, 18, 27]], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")


# ==========================================
# 3. MEMORY MANAGEMENT UTILS
# ==========================================
def empty_cache_xpu():
    """Aggressively clear RAM and native Intel XPU VRAM."""
    gc.collect()
    torch.xpu.empty_cache()


# ==========================================
# 4. MAIN GENERATION PIPELINE
# ==========================================
def generate_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = 42):
    device = "xpu"
    dtype = torch.bfloat16

    print(f"\n--- Starting Generation: '{prompt}' ---")

    # --- PHASE A: Text Encoding ---
    print("\n>> [1/3] Loading Text Encoder (Qwen3-4B)...")
    text_encoder = LocalQwen3Embedder(device="cpu")
    text_encoder.model.to(device)

    print(">> Generating text embeddings...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        txt_hidden_states = text_encoder([prompt])
        txt, txt_ids = batched_prc_txt(txt_hidden_states)
        txt = txt.to(device, dtype=dtype)
        txt_ids = txt_ids.to(device)

    print(">> Unloading Text Encoder to free VRAM...")
    del text_encoder
    empty_cache_xpu()

    # --- PHASE B: Transformer (DiT) Denoising ---
    print("\n>> [2/3] Loading FLUX.2 Transformer...")
    print(f"   -> Loading weights from: {TRANSFORMER_PATH}")

    with torch.device("meta"):
        model = Flux2(Klein4BParams()).to(dtype)
    sd = load_sft(TRANSFORMER_PATH, device=device)
    model.load_state_dict(sd, strict=True, assign=True)
    model = model.to(device)

    print(">> Preparing initial noise latents...")
    torch.manual_seed(seed)
    h_latent, w_latent = height // 16, width // 16
    img = torch.randn(1, 128, h_latent, w_latent, dtype=dtype, device=device)

    img_flat, img_ids = batched_prc_img(img)
    img_flat = img_flat.to(device, dtype=dtype)
    img_ids = img_ids.to(device)

    num_steps = 4
    guidance = 1.0
    timesteps = get_schedule(num_steps, img_flat.shape[1])

    print(f">> Denoising for {num_steps} steps...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        img_flat = denoise(
            model, img_flat, img_ids, txt, txt_ids, timesteps, guidance=guidance
        )

    print(">> Unloading Transformer to free VRAM...")
    del model
    del sd
    empty_cache_xpu()

    # --- PHASE C: Autoencoder Decoding ---
    print("\n>> [3/3] Loading Autoencoder...")
    print(f"   -> Loading weights from: {VAE_PATH}")
    img_latent = rearrange(img_flat, "b (h w) c -> b c h w", h=h_latent, w=w_latent)

    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams()).to(dtype)
    sd_ae = load_sft(VAE_PATH, device=device)
    ae.load_state_dict(sd_ae, strict=True, assign=True)
    ae = ae.to(device)

    print(">> Decoding latent to image...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        decoded = ae.decode(img_latent)

    print(">> Unloading Autoencoder...")
    del ae
    del sd_ae
    empty_cache_xpu()

    # --- PHASE D: Save Output ---
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()

    image = Image.fromarray((decoded[0] * 255).astype(np.uint8))
    output_dir = os.path.join(SCRIPT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"gen_fp8_{timestamp}_{seed}.png")
            output_path = os.path.join(output_dir, output_path)

    image.save(output_path)

    metadata = {
        'prompt': prompt,
        'width': width,
        'height': height,
        'seed': seed,
        'device': device,
        'dtype': str(dtype),
    }
    metadata_path = os.path.splitext(output_path)[0] + '.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n>> Success! Image saved to: {output_path}")
    print(f">> Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    if not os.path.exists(TRANSFORMER_PATH):
        print(f"ERROR: Cannot find Transformer at {TRANSFORMER_PATH}")
        exit(1)

    def detect_device():
        print("\n>> [0/3] Device detection and information")
        print(f"   - PyTorch version: {torch.__version__}")
        has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
        has_cuda = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

        print(f"   - XPU available: {has_xpu}")
        print(f"   - CUDA available: {has_cuda}")
        print(f"   - MPS available: {has_mps}")

        selected = 'xpu' if has_xpu else ('cuda' if has_cuda else ('mps' if has_mps else 'cpu'))
        print(f"   - Using device: {selected}")
        return selected

    def load_prompts_from_file():
        prompt_file = os.path.join(SCRIPT_DIR, 'prompts', 'prompts.txt')
        if not os.path.exists(prompt_file):
            prompt_file = os.path.join(SCRIPT_DIR, 'prompts.txt')
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"prompts file not found at {prompt_file}")

        with open(prompt_file, 'r', encoding='utf-8') as f:
            raw = [line.strip() for line in f.readlines()]
        prompts = [line for line in raw if line and not line.startswith('#')]
        if not prompts:
            raise ValueError(f"No valid prompts found in {prompt_file}")

        print(f"   - Loaded {len(prompts)} prompt(s) from: {prompt_file}")
        return prompts

    def generate_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = 42, num_steps: int = 4, output_path: str = None):
        device = 'xpu'
        dtype = torch.bfloat16

        print(f"\n--- Starting Generation: '{prompt}' (seed={seed}, steps={num_steps}) ---")

        # do not set output_path here; let phase D assign timestamp + seed filename when output_path is None

        # --- PHASE A: Text Encoding ---
        print("\n>> [1/3] Loading Text Encoder (Qwen3-4B)...")
        text_encoder = LocalQwen3Embedder(device='cpu')
        text_encoder.model.to(device)

        print(">> Generating text embeddings...")
        with torch.no_grad(), torch.autocast(device_type='xpu', dtype=dtype):
            txt_hidden_states = text_encoder([prompt])
            txt, txt_ids = batched_prc_txt(txt_hidden_states)
            txt = txt.to(device, dtype=dtype)
            txt_ids = txt_ids.to(device)

        print(">> Unloading Text Encoder to free VRAM...")
        del text_encoder
        empty_cache_xpu()

        # --- PHASE B: Transformer (DiT) Denoising ---
        print(">> [2/3] Loading FLUX.2 Transformer...")
        print(f"   -> Loading weights from: {TRANSFORMER_PATH}")

        with torch.device('meta'):
            model = Flux2(Klein4BParams()).to(dtype)
        sd = load_sft(TRANSFORMER_PATH, device=device)
        model.load_state_dict(sd, strict=True, assign=True)
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
        with torch.no_grad(), torch.autocast(device_type='xpu', dtype=dtype):
            img_flat = denoise(model, img_flat, img_ids, txt, txt_ids, timesteps, guidance=guidance)

        print(">> Unloading Transformer to free VRAM...")
        del model
        del sd
        empty_cache_xpu()

        # --- PHASE C: Autoencoder Decoding ---
        print(">> [3/3] Loading Autoencoder...")
        print(f"   -> Loading weights from: {VAE_PATH}")
        img_latent = rearrange(img_flat, 'b (h w) c -> b c h w', h=h_latent, w=w_latent)

        with torch.device('meta'):
            ae = AutoEncoder(AutoEncoderParams()).to(dtype)
        sd_ae = load_sft(VAE_PATH, device=device)
        ae.load_state_dict(sd_ae, strict=True, assign=True)
        ae = ae.to(device)

        print(">> Decoding latent to image...")
        with torch.no_grad(), torch.autocast(device_type='xpu', dtype=dtype):
            decoded = ae.decode(img_latent)

        print(">> Unloading Autoencoder...")
        del ae
        del sd_ae
        empty_cache_xpu()

        # --- PHASE D: Save Output ---
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()

        image = Image.fromarray((decoded[0] * 255).astype(np.uint8))

        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(SCRIPT_DIR, f"klein_{timestamp}_{seed}.png")

        image.save(output_path)

        metadata = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'seed': seed,
            'device': device,
            'dtype': str(dtype),
            'num_steps': num_steps,
        }
        metadata_path = os.path.splitext(output_path)[0] + '.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n>> Success! Image saved to: {output_path}")
        print(f">> Metadata saved to: {metadata_path}")

    def ask_num_steps():
        num_steps_input = input('Enter denoising steps [default 4]: ').strip() or '4'
        try:
            num_steps = int(num_steps_input)
            if num_steps < 1:
                raise ValueError
        except ValueError:
            print('Invalid steps entered. Defaulting to 4.')
            num_steps = 4
        return num_steps

    def run_flow():
        selected_device = detect_device()
        print('>> Device selected:', selected_device)

        num_steps = ask_num_steps()
        user_prompt = input('Enter prompt (leave blank to use prompts/prompts.txt): ').strip()
        if not user_prompt:
            prompts = load_prompts_from_file()
            if len(prompts) == 1:
                prompt = prompts[0]
                count_str = input('Only one prompt loaded. How many images to generate with this prompt? [default 1]: ').strip() or '1'
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
                    generate_image(prompt, seed=seed, num_steps=num_steps)
                else:
                    for _ in range(count):
                        seed = np.random.randint(0, 2**31 - 1)
                        generate_image(prompt, seed=seed, num_steps=num_steps)
                return

            for prompt in prompts:
                seed = np.random.randint(0, 2**31 - 1)
                generate_image(prompt, seed=seed, num_steps=num_steps)
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
            generate_image(user_prompt, seed=seed, num_steps=num_steps)
        else:
            for _ in range(count):
                seed = np.random.randint(0, 2**31 - 1)
                generate_image(user_prompt, seed=seed, num_steps=num_steps)

    run_flow()
