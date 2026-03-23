import os
import gc
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from einops import rearrange
from safetensors.torch import load_file as load_sft
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the core math and architecture components
from flux2.model import Flux2, Klein4BParams
from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.sampling import (
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
MODEL_DIR = os.path.join(SCRIPT_DIR, "model", "FLUX.2-klein-4B")

TRANSFORMER_PATH = os.path.join(MODEL_DIR, "flux-2-klein-4b.safetensors")
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

# The two images we generated in the previous steps
INPUT_IMAGE_1_PATH = os.path.join(SCRIPT_DIR, "output_klein_4b_xpu.jpg")  # Blue crystal
INPUT_IMAGE_2_PATH = os.path.join(SCRIPT_DIR, "edited_klein_4b_xpu.jpg")  # Red crystal


# ==========================================
# 2. CUSTOM LOCAL TEXT ENCODER
# ==========================================
class LocalQwen3Embedder(nn.Module):
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
# 4. MAIN MULTI-REF PIPELINE
# ==========================================
def generate_from_multiple_references(
    prompt: str, image_paths: list[str], seed: int = 42
):
    device = "xpu"
    dtype = torch.bfloat16

    print(f"\n--- Starting Multi-Ref Gen: '{prompt}' ---")

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

    # --- PHASE B: Image Encoding (MULTIPLE IMAGES) ---
    print("\n>> [2/4] Loading Autoencoder for Reference Encoding...")
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams()).to(dtype)
    sd_ae = load_sft(VAE_PATH, device=device)
    ae.load_state_dict(sd_ae, strict=True, assign=True)
    ae = ae.to(device)

    print(f">> Encoding {len(image_paths)} reference images...")
    # Load all images into a list
    input_imgs = [Image.open(path).convert("RGB") for path in image_paths]

    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        # Pass the entire list of images to the encoder at once
        ref_tokens, ref_ids = encode_image_refs(ae, input_imgs)
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

    print(">> Preparing fresh noise canvas matched to first reference size...")
    torch.manual_seed(seed)

    # We anchor the output size to the first image in the list
    w, h = input_imgs[0].size
    new_w, new_h = (w // 16) * 16, (h // 16) * 16
    h_latent, w_latent = new_h // 16, new_w // 16

    img = torch.randn(1, 128, h_latent, w_latent, dtype=dtype, device=device)
    img_flat, img_ids = batched_prc_img(img)
    img_flat = img_flat.to(device, dtype=dtype)
    img_ids = img_ids.to(device)

    num_steps = 4
    guidance = 1.0
    timesteps = get_schedule(num_steps, img_flat.shape[1])

    print(f">> Denoising for {num_steps} steps using ALL reference tokens...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        img_flat = denoise(
            model=model,
            img=img_flat,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
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
    output_path = os.path.join(SCRIPT_DIR, "multi_refgen_klein_4b_xpu.jpg")
    image.save(output_path)
    print(f"\n>> Success! Multi-reference generated image saved to: {output_path}")


if __name__ == "__main__":
    if not os.path.exists(TRANSFORMER_PATH):
        print(f"ERROR: Cannot find Transformer at {TRANSFORMER_PATH}")
        exit(1)

    if not os.path.exists(INPUT_IMAGE_1_PATH) or not os.path.exists(INPUT_IMAGE_2_PATH):
        print("ERROR: Cannot find one or both reference images!")
        print(f"Make sure {INPUT_IMAGE_1_PATH} and {INPUT_IMAGE_2_PATH} exist.")
        exit(1)

    # Let's combine the blue and red concepts!
    multi_refgen_prompt = "A highly detailed macro photography shot of a crystal that is half glowing blue and half glowing red, merging together in a dark cave, cinematic lighting, 8k resolution"

    # Notice we are passing a list of image paths now
    generate_from_multiple_references(
        multi_refgen_prompt, image_paths=[INPUT_IMAGE_1_PATH, INPUT_IMAGE_2_PATH]
    )
