import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class GenerationResult:
    image_bytes: bytes
    output_path: str
    seed: int
    device: str
    model_id: str
    lora_adapter_path: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float


class DiffusionGenerator:
    def __init__(
        self,
        base_model_id: str,
        adapter_path: Path | None,
        output_dir: Path,
        device: str = "auto",
    ):
        self.base_model_id = base_model_id
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.output_dir = Path(output_dir)
        self.requested_device = device
        self._pipe = None
        self._device = None

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def info(self) -> dict:
        return {
            "base_model_id": self.base_model_id,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "lora_adapter_path": str(self.adapter_path) if self.adapter_path else "",
            "output_dir": str(self.output_dir),
            "device": self._device or self.requested_device,
            "loaded": self.is_loaded(),
        }

    def _load(self):
        if self._pipe is not None:
            return

        import torch
        from diffusers import StableDiffusionPipeline

        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device

        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if self.adapter_path:
            if not self.adapter_path.exists():
                raise FileNotFoundError(f"Diffusion adapter path does not exist: {self.adapter_path}")
            pipe.load_lora_weights(str(self.adapter_path))
        pipe = pipe.to(device)

        self._pipe = pipe
        self._device = device

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> GenerationResult:
        self._load()

        import torch

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("width and height must be divisible by 8")

        if seed is None:
            seed = int(time.time() * 1000) % 2_147_483_647

        image, actual_steps = self._run_with_fallback(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        image_bytes = self._to_png_bytes(image)
        output_path = self._save_image(prompt, seed, image_bytes)

        return GenerationResult(
            image_bytes=image_bytes,
            output_path=str(output_path),
            seed=seed,
            device=self._device,
            model_id=self.base_model_id,
            lora_adapter_path=str(self.adapter_path) if self.adapter_path else "",
            width=width,
            height=height,
            num_inference_steps=actual_steps,
            guidance_scale=guidance_scale,
        )

    def _run_once(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Image.Image:
        import torch

        generator = torch.Generator(device=self._device).manual_seed(seed)
        return self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    def _run_with_fallback(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> tuple[Image.Image, int]:
        try:
            image = self._run_once(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            return image, num_inference_steps
        except IndexError as exc:
            message = str(exc).lower()
            if "out of bounds" not in message:
                raise

            safe_steps = max(8, min(num_inference_steps - 1, 20))
            if safe_steps == num_inference_steps:
                raise

            image = self._run_once(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                num_inference_steps=safe_steps,
                guidance_scale=guidance_scale,
            )
            return image, safe_steps

    def _save_image(self, prompt: str, seed: int, image_bytes: bytes) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        output_path = self.output_dir / f"diffusion_generation_{int(time.time())}_{seed}_{prompt_hash}.png"
        output_path.write_bytes(image_bytes)
        return output_path

    @staticmethod
    def _to_png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


DiffusionLoraGenerator = DiffusionGenerator
