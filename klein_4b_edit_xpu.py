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
from huggingface_hub import snapshot_download

# Import the core math and architecture components
from klein.model import Flux2, Klein4BParams
from klein.autoencoder import AutoEncoder, AutoEncoderParams
from klein.sampling import (
    get_schedule,
    denoise,
    batched_prc_txt,
    batched_prc_img,
    encode_image_refs,
)

# ==========================================
# 1. AUTOMATIC PATH RESOLUTION & DOWNLOAD
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "FLUX.2-klein-4B-FP16")

TRANSFORMER_PATH = os.path.join(MODEL_DIR, "transformer", "flux-2-klein-4b.safetensors")
VAE_PATH = os.path.join(MODEL_DIR, "autoencoder", "ae.safetensors")
TEXT_ENCODER_DIR = os.path.join(MODEL_DIR, "text_encoder")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")

# Only ping Hugging Face if the main model file is completely missing
if not os.path.exists(TRANSFORMER_PATH):
    print("\n>> Model weights not found locally. Downloading from Hugging Face...")
    print(f">> Syncing with repository: rootlocalghost/FLUX.2-klein-4B-FP16")
    snapshot_download(
        repo_id="rootlocalghost/FLUX.2-klein-4B-FP16",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,  # Forces real files instead of weird Windows cache shortcuts
    )
else:
    print("\n>> Local model weights found! Skipping download check.")

INPUT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "input", "edit", "edit.png")


# ==========================================
# 2. CUSTOM LOCAL TEXT ENCODER
# ==========================================
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
            print("\n>> Local text encoder not found. Downloading from rootlocalghost/FLUX.2-klein-4B-FP16...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="rootlocalghost/FLUX.2-klein-4B-FP16",
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
        all_input_ids, all_attention_masks = [], []
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


# ==========================================
# 3. MEMORY MANAGEMENT UTILS
# ==========================================
def empty_cache_xpu():
    """Aggressively clear RAM and native Intel XPU VRAM."""
    gc.collect()
    torch.xpu.empty_cache()


# ==========================================
# 4. MAIN EDITING PIPELINE
# ==========================================
def edit_image(prompt: str, image_path: str, seed: int = 42, num_steps: int = 4):
    device = "xpu"
    dtype = torch.bfloat16

    print(f"\n--- Starting Edit: '{prompt}' (seed={seed}, steps={num_steps}) ---")

    # --- PHASE A: Text Encoding ---
    print("\n>> [1/4] Loading Text Encoder (Qwen3-4B)...")
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

    # --- PHASE B: Image Encoding (The new step!) ---
    print("\n>> [2/4] Loading Autoencoder for Image Encoding...")
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams()).to(dtype)
    sd_ae = load_sft(VAE_PATH, device=device)
    ae.load_state_dict(sd_ae, strict=True, assign=True)
    ae = ae.to(device)

    print(f">> Encoding reference image: {image_path}")
    input_img = Image.open(image_path).convert("RGB")

    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        # This converts your image into reference tokens for the Transformer
        ref_tokens, ref_ids = encode_image_refs(ae, [input_img])
        ref_tokens = ref_tokens.to(device, dtype=dtype)
        ref_ids = ref_ids.to(device)

    del ae, sd_ae
    empty_cache_xpu()

    # --- PHASE C: Transformer (DiT) Denoising ---
    print("\n>> [3/4] Loading FLUX.2 Transformer...")
    with torch.device("meta"):
        model = Flux2(Klein4BParams()).to(dtype)
    sd = load_sft(TRANSFORMER_PATH, device=device)
    model.load_state_dict(sd, strict=True, assign=True)
    model = model.to(device)

    print(">> Preparing noise latents matched to image size...")
    torch.manual_seed(seed)

    # We match the output shape to the input image, ensuring it is a multiple of 16
    w, h = input_img.size
    new_w, new_h = (w // 16) * 16, (h // 16) * 16
    h_latent, w_latent = new_h // 16, new_w // 16

    img = torch.randn(1, 128, h_latent, w_latent, dtype=dtype, device=device)
    img_flat, img_ids = batched_prc_img(img)
    img_flat = img_flat.to(device, dtype=dtype)
    img_ids = img_ids.to(device)

    guidance = 1.0  # Keep this at 1.0 for the distilled Klein models
    timesteps = get_schedule(num_steps, img_flat.shape[1])

    print(f">> Denoising for {num_steps} steps using text AND image references...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        # We pass the ref_tokens and ref_ids into the img_cond_seq arguments
        img_flat = denoise(
            model=model,
            img=img_flat,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,  # <--- The magic happens here
            img_cond_seq_ids=ref_ids,  # <--- and here
        )

    del model, sd
    empty_cache_xpu()

    # --- PHASE D: Autoencoder Decoding ---
    print("\n>> [4/4] Reloading Autoencoder to Decode Final Image...")
    img_latent = rearrange(img_flat, "b (h w) c -> b c h w", h=h_latent, w=w_latent)

    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams()).to(dtype)
    sd_ae = load_sft(VAE_PATH, device=device)
    ae.load_state_dict(sd_ae, strict=True, assign=True)
    ae = ae.to(device)

    print(">> Decoding latent to image...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        decoded = ae.decode(img_latent)

    del ae, sd_ae
    empty_cache_xpu()

    # --- SAVE OUTPUT ---
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()

    image = Image.fromarray((decoded[0] * 255).astype(np.uint8))
    output_dir = os.path.join(SCRIPT_DIR, 'output/edits')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"edit_{int(time.time())}_{seed}.png")
    image.save(output_path)

    metadata = {
        'prompt': prompt,
        'width': new_w if 'new_w' in locals() else w,
        'height': new_h if 'new_h' in locals() else h,
        'seed': seed,
        'device': device,
        'dtype': str(dtype),
        'num_steps': num_steps,
    }
    metadata_path = os.path.splitext(output_path)[0] + '.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n>> Success! Edited image saved to: {output_path}")
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
    if not os.path.exists(TRANSFORMER_PATH):
        print(f"ERROR: Cannot find Transformer at {TRANSFORMER_PATH}")
        exit(1)

    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERROR: Cannot find the reference image at {INPUT_IMAGE_PATH}")
        print("Make sure you run klein_4b_gen_xpu.py first to generate it!")
        exit(1)

    num_steps = ask_num_steps()

    user_prompt = input('Enter prompt (leave blank to use prompts/prompts.txt): ').strip()
    if not user_prompt:
        prompts_file = os.path.join(SCRIPT_DIR, 'prompts', 'prompts.txt')
        if not os.path.exists(prompts_file):
            prompts_file = os.path.join(SCRIPT_DIR, 'prompts.txt')
        if not os.path.exists(prompts_file):
            print('No prompts file found. Exiting.')
            return

        with open(prompts_file, 'r', encoding='utf-8') as f:
            raw = [line.strip() for line in f.readlines()]
        prompts = [line for line in raw if line and not line.startswith('#')]

        if not prompts:
            print('No valid prompts found in prompts file. Exiting.')
            return

        if len(prompts) == 1:
            prompt = prompts[0]
            count_str = input('Only one prompt loaded. How many edits to generate? [default 1]: ').strip() or '1'
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
                edit_image(prompt, image_path=INPUT_IMAGE_PATH, seed=seed, num_steps=num_steps)
            else:
                for _ in range(count):
                    seed = np.random.randint(0, 2**31 - 1)
                    edit_image(prompt, image_path=INPUT_IMAGE_PATH, seed=seed, num_steps=num_steps)
            return

        for prompt in prompts:
            seed = np.random.randint(0, 2**31 - 1)
            edit_image(prompt, image_path=INPUT_IMAGE_PATH, seed=seed, num_steps=num_steps)
        return

    count_str = input('How many edits to generate? [default 1]: ').strip() or '1'
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
        edit_image(user_prompt, image_path=INPUT_IMAGE_PATH, seed=seed, num_steps=num_steps)
    else:
        for _ in range(count):
            seed = np.random.randint(0, 2**31 - 1)
            edit_image(user_prompt, image_path=INPUT_IMAGE_PATH, seed=seed, num_steps=num_steps)


if __name__ == "__main__":
    run_flow()
