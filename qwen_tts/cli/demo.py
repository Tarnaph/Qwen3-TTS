# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A gradio demo for Qwen3 TTS models.
"""

import argparse
import math
import time
import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from .. import Qwen3TTSModel, VoiceClonePromptItem


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _maybe(v):
    return v if v is not None else gr.update()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio demo for Qwen3 TTS models (CustomVoice / VoiceDesign / Base).\n\n"
            "Examples:\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000 --ip 127.0.0.01\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --dtype bfloat16 --no-flash-attn\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    # Positional checkpoint (also supports -c/--checkpoint)
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (positional).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (optional if positional is provided).",
    )

    # Model loading / from_pretrained args
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )

    # Gradio server args
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="Server bind IP for Gradio (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for Gradio (default: 8000).",
    )
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Gradio queue concurrency (default: 16).",
    )

    # HTTPS args
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    # Optional generation args
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for generation (optional).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (optional).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (optional).")
    parser.add_argument("--subtalker-top-k", type=int, default=None, help="Subtalker top-k (optional, only for tokenizer v2).")
    parser.add_argument("--subtalker-top-p", type=float, default=None, help="Subtalker top-p (optional, only for tokenizer v2).")
    parser.add_argument(
        "--subtalker-temperature", type=float, default=None, help="Subtalker temperature (optional, only for tokenizer v2)."
    )

    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        raise SystemExit(0)  # main() prints help
    return ckpt


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)

        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid

    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0

        if m <= 1.0 + 1e-6:
            pass
        else:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav


def _parse_srt(content: str) -> List[Tuple[int, str]]:
    """Parse SRT content and return list of (index, text) tuples.

    Handles:
    - UTF-8 BOM (\ufeff) at file start
    - Windows CRLF line endings
    - SRT HTML tags (<i>, <b>, <font color=...>)
    - Malformed blocks where timecode regex fails
    """
    import re
    # Strip BOM and normalise line endings
    content = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    entries = []
    # Split on blank lines (one or more)
    blocks = re.split(r"\n{2,}", content.strip())
    _timecode_re = re.compile(
        r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
    )
    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # Line 0 should be the numeric index
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        # Find the timecode line (usually lines[1] but search robustly)
        tc_idx = None
        for li, ln in enumerate(lines[1:], start=1):
            if _timecode_re.match(ln.strip()):
                tc_idx = li
                break
        if tc_idx is None:
            # No timecode found — skip this block
            continue
        # Everything after the timecode line is subtitle text
        text_lines = lines[tc_idx + 1 :]
        raw = " ".join(ln.strip() for ln in text_lines if ln.strip())
        # Strip HTML tags that SRT files sometimes contain
        text = re.sub(r"<[^>]+>", "", raw).strip()
        if text:
            entries.append((idx, text))
    return entries


def _detect_model_kind(ckpt: str, tts: Qwen3TTSModel) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    else:
        raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


def _vram_status() -> str:
    """Return a compact VRAM usage string, e.g. 'VRAM: 4.2/8.0 GB (52%)'.
    Returns empty string if CUDA is not available."""
    if not torch.cuda.is_available():
        return ""
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = int(reserved / total * 100) if total > 0 else 0
    return f"VRAM {reserved:.1f}/{total:.0f}GB ({pct}%)"


# Lazy-loaded VoiceDesign model (loads on first use, avoids startup delay)
_vd_tts: Optional[Qwen3TTSModel] = None


def build_demo(tts: Qwen3TTSModel, ckpt: str, gen_kwargs_default: Dict[str, Any]) -> gr.Blocks:
    model_kind = _detect_model_kind(ckpt, tts)

    supported_langs_raw = None
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()

    supported_spks_raw = None
    if callable(getattr(tts.model, "get_supported_speakers", None)):
        supported_spks_raw = tts.model.get_supported_speakers()

    lang_choices_disp, lang_map = _build_choices_and_map([x for x in (supported_langs_raw or [])])
    spk_choices_disp, spk_map = _build_choices_and_map([x for x in (supported_spks_raw or [])])

    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Qwen3 TTS Demo
**Checkpoint:** `{ckpt}`  
**Model Type:** `{model_kind}`  
"""
        )

        if model_kind == "custom_voice":
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="Text (待合成文本)",
                        lines=4,
                        placeholder="Enter text to synthesize (输入要合成的文本).",
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="Language (语种)",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                        spk_in = gr.Dropdown(
                            label="Speaker (说话人)",
                            choices=spk_choices_disp,
                            value="Vivian",
                            interactive=True,
                        )
                    instruct_in = gr.Textbox(
                        label="Instruction (Optional) (控制指令，可不输入)",
                        lines=2,
                        placeholder="e.g. Say it in a very angry tone (例如：用特别伤心的语气说).",
                    )
                    btn = gr.Button("Generate (生成)", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                    err = gr.Textbox(label="Status (状态)", lines=2)

            def run_instruct(text: str, lang_disp: str, spk_disp: str, instruct: str):
                try:
                    if not text or not text.strip():
                        return None, "Text is required (必须填写文本)."
                    if not spk_disp:
                        return None, "Speaker is required (必须选择说话人)."
                    language = lang_map.get(lang_disp, "Auto")
                    speaker = spk_map.get(spk_disp, spk_disp)
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_custom_voice(
                        text=text.strip(),
                        language=language,
                        speaker=speaker,
                        instruct=(instruct or "").strip() or None,
                        **kwargs,
                    )
                    return _wav_to_gradio_audio(wavs[0], sr), "Finished. (生成完成)"
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(run_instruct, inputs=[text_in, lang_in, spk_in, instruct_in], outputs=[audio_out, err])

        elif model_kind == "voice_design":
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="Text (待合成文本)",
                        lines=4,
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="Language (语种)",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                    design_in = gr.Textbox(
                        label="Voice Design Instruction (音色描述)",
                        lines=3,
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                    )
                    btn = gr.Button("Generate (生成)", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                    err = gr.Textbox(label="Status (状态)", lines=2)

            def run_voice_design(text: str, lang_disp: str, design: str):
                try:
                    if not text or not text.strip():
                        return None, "Text is required (必须填写文本)."
                    if not design or not design.strip():
                        return None, "Voice design instruction is required (必须填写音色描述)."
                    language = lang_map.get(lang_disp, "Auto")
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_voice_design(
                        text=text.strip(),
                        language=language,
                        instruct=design.strip(),
                        **kwargs,
                    )
                    return _wav_to_gradio_audio(wavs[0], sr), "Finished. (生成完成)"
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(run_voice_design, inputs=[text_in, lang_in, design_in], outputs=[audio_out, err])

        else:  # voice_clone for base
            with gr.Tabs():
                with gr.Tab("Clone & Generate (克隆并合成)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            ref_audio = gr.Audio(
                                label="Reference Audio (参考音频)",
                            )
                            ref_text = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder="Required if not set use x-vector only (不勾选use x-vector only时必填).",
                            )
                            xvec_only = gr.Checkbox(
                                label="Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)",
                                value=False,
                            )

                        with gr.Column(scale=2):
                            text_in = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            lang_in = gr.Dropdown(
                                label="Language (语种)",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            instruct_in = gr.Textbox(
                                label="Instructions / Style (Optional) (控制指令，可不输入)",
                                lines=2,
                                placeholder="e.g. Speak in a very excited tone (例如：用特别激动的语气说).",
                            )
                            btn = gr.Button("Generate (生成)", variant="primary")

                        with gr.Column(scale=3):
                            audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                            err = gr.Textbox(label="Status (状态)", lines=2)

                    def run_voice_clone(ref_aud, ref_txt: str, use_xvec: bool, text: str, lang_disp: str, instruct: str):
                        try:
                            if not text or not text.strip():
                                return None, "Target text is required (必须填写待合成文本)."
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "Reference audio is required (必须上传参考音频)."
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "Reference text is required when use x-vector only is NOT enabled.\n"
                                    "(未勾选 use x-vector only 时，必须提供参考音频文本；否则请勾选 use x-vector only，但效果会变差.)"
                                )
                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            t0 = time.time()
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                                instruct=(instruct or "").strip() or None,
                                **kwargs,
                            )
                            elapsed = time.time() - t0
                            return _wav_to_gradio_audio(wavs[0], sr), f"✅ Finished in {elapsed:.1f}s (生成完成)"
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    btn.click(
                        run_voice_clone,
                        inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in, instruct_in],
                        outputs=[audio_out, err],
                    )

                with gr.Tab("Save / Load Voice (保存/加载克隆音色)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Save Voice (保存音色)
Upload reference audio and text, choose use x-vector only or not, then save a reusable voice prompt file.  
(上传参考音频和参考文本，选择是否使用 use x-vector only 模式后保存为可复用的音色文件)
"""
                            )
                            ref_audio_s = gr.Audio(label="Reference Audio (参考音频)", type="numpy")
                            ref_text_s = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder="Required if not set use x-vector only (不勾选use x-vector only时必填).",
                            )
                            xvec_only_s = gr.Checkbox(
                                label="Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)",
                                value=False,
                            )
                            save_btn = gr.Button("Save Voice File (保存音色文件)", variant="primary")
                            prompt_file_out = gr.File(label="Voice File (音色文件)")

                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Load Voice & Generate (加载音色并合成)
Upload a previously saved voice file, then synthesize new text.  
(上传已保存提示文件后，输入新文本进行合成)
"""
                            )
                            prompt_file_in = gr.File(label="Upload Prompt File (上传提示文件)")
                            text_in2 = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            lang_in2 = gr.Dropdown(
                                label="Language (语种)",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            instruct_in2 = gr.Textbox(
                                label="Instructions / Style (Optional) (控制指令，可不输入)",
                                lines=2,
                                placeholder="e.g. Speak in a very excited tone (例如：用特别激动的语气说).",
                            )
                            gen_btn2 = gr.Button("Generate (生成)", variant="primary")

                        with gr.Column(scale=3):
                            audio_out2 = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                            err2 = gr.Textbox(label="Status (状态)", lines=2)

                    def save_prompt(ref_aud, ref_txt: str, use_xvec: bool):
                        try:
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "Reference audio is required (必须上传参考音频)."
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "Reference text is required when use x-vector only is NOT enabled.\n"
                                    "(未勾选 use x-vector only 时，必须提供参考音频文本；否则请勾选 use x-vector only，但效果会变差.)"
                                )
                            items = tts.create_voice_clone_prompt(
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                            )
                            payload = {
                                "items": [asdict(it) for it in items],
                            }
                            fd, out_path = tempfile.mkstemp(prefix="voice_clone_prompt_", suffix=".pt")
                            os.close(fd)
                            torch.save(payload, out_path)
                            return out_path, "Finished. (生成完成)"
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    def load_prompt_and_gen(file_obj, text: str, lang_disp: str, instruct: str):
                        try:
                            if file_obj is None:
                                return None, "Voice file is required (必须上传音色文件)."
                            if not text or not text.strip():
                                return None, "Target text is required (必须填写待合成文本)."

                            path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
                            payload = torch.load(path, map_location="cpu", weights_only=True)
                            if not isinstance(payload, dict) or "items" not in payload:
                                return None, "Invalid file format (文件格式不正确)."

                            items_raw = payload["items"]
                            if not isinstance(items_raw, list) or len(items_raw) == 0:
                                return None, "Empty voice items (音色为空)."

                            items: List[VoiceClonePromptItem] = []
                            for d in items_raw:
                                if not isinstance(d, dict):
                                    return None, "Invalid item format in file (文件内部格式错误)."
                                ref_code = d.get("ref_code", None)
                                if ref_code is not None and not torch.is_tensor(ref_code):
                                    ref_code = torch.tensor(ref_code)
                                ref_spk = d.get("ref_spk_embedding", None)
                                if ref_spk is None:
                                    return None, "Missing ref_spk_embedding (缺少说话人向量)."
                                if not torch.is_tensor(ref_spk):
                                    ref_spk = torch.tensor(ref_spk)

                                items.append(
                                    VoiceClonePromptItem(
                                        ref_code=ref_code,
                                        ref_spk_embedding=ref_spk,
                                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                                        icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                                        ref_text=d.get("ref_text", None),
                                    )
                                )

                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            t0 = time.time()
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                voice_clone_prompt=items,
                                instruct=(instruct or "").strip() or None,
                                **kwargs,
                            )
                            elapsed = time.time() - t0
                            return _wav_to_gradio_audio(wavs[0], sr), f"✅ Finished in {elapsed:.1f}s (生成完成)"
                        except Exception as e:
                            return None, (
                                f"Failed to read or use voice file. Check file format/content.\n"
                                f"(读取或使用音色文件失败，请检查文件格式或内容)\n"
                                f"{type(e).__name__}: {e}"
                            )

                    save_btn.click(save_prompt, inputs=[ref_audio_s, ref_text_s, xvec_only_s], outputs=[prompt_file_out, err2])
                    gen_btn2.click(load_prompt_and_gen, inputs=[prompt_file_in, text_in2, lang_in2, instruct_in2], outputs=[audio_out2, err2])

                with gr.Tab("SRT Batch (字幕批量合成)"):
                    gr.Markdown(
                        """
### SRT Batch Generation (字幕批量合成)
Upload an `.srt` subtitle file. Each subtitle entry will be synthesized and saved as a numbered file in the output folder.  
Choose the voice source: **Reference Audio** (raw audio) or **Load Voice File** (pre-saved `.pt` prompt).  
(上传字幕文件，每条字幕将被单独合成并按序号保存)
"""
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.Tab("Reference Audio (参考音频)"):
                                    srt_ref_audio = gr.Audio(
                                        label="Reference Audio (参考音频)",
                                    )
                                    srt_ref_text = gr.Textbox(
                                        label="Reference Text (参考音频文本)",
                                        lines=2,
                                        placeholder="Required if not using x-vector only (不勾选use x-vector only时必填).",
                                    )
                                    srt_xvec_only = gr.Checkbox(
                                        label="Use x-vector only (仅用说话人向量)",
                                        value=False,
                                    )

                                with gr.Tab("Load Voice File (加载音色文件)"):
                                    gr.Markdown(
                                        "Upload a previously saved voice prompt file (`.pt`) from the **Save / Load Voice** tab."
                                    )
                                    srt_prompt_file = gr.File(
                                        label="Voice Prompt File (.pt)",
                                        file_types=[".pt"],
                                    )

                            srt_lang = gr.Dropdown(
                                label="Language (语种)",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            srt_instruct = gr.Textbox(
                                label="Instructions / Style (Optional) (控制指令，可不输入)",
                                lines=2,
                                placeholder="e.g. Speak in a calm, solemn tone.",
                            )

                        with gr.Column(scale=2):
                            srt_file = gr.File(
                                label="SRT Subtitle File (.srt)",
                                file_types=[".srt"],
                            )
                            srt_out_dir = gr.Textbox(
                                label="Output Folder (输出文件夹路径)",
                                placeholder="e.g. C:/Users/you/Desktop/output_audio",
                                lines=1,
                            )
                            srt_format = gr.Radio(
                                label="Output Format (输出格式)",
                                choices=["WAV", "MP3", "MP4"],
                                value="MP3",
                            )

                            # --- Block selection ---
                            with gr.Accordion("🧩 Block Generation (geração por blocos)", open=False):
                                gr.Markdown(
                                    "Divide the SRT into blocks of N entries. Select which blocks to generate. "
                                    "**Leave all unchecked to generate everything** (default)."
                                )
                                srt_block_size = gr.Number(
                                    label="Entries per block (legendas por bloco)",
                                    value=10,
                                    minimum=1,
                                    step=1,
                                    precision=0,
                                    info="Change this after uploading the SRT to recalculate blocks.",
                                )
                                srt_block_info = gr.Markdown("📂 Upload an SRT file to see available blocks.")
                                srt_blocks_sel = gr.CheckboxGroup(
                                    label="Blocks to generate (leave empty = ALL)",
                                    choices=[],
                                    value=[],
                                )

                            srt_auto_retry = gr.Checkbox(
                                label="🔄 Auto-retry on failure",
                                value=False,
                                info="If enabled, failed entries are automatically retried after the batch finishes (until no new successes).",
                            )
                            srt_auto_next = gr.Checkbox(
                                label="⏭️ Auto-start next block",
                                value=False,
                                info="If enabled, automatically continues to the next block after this one finishes without needing to click.",
                            )
                            srt_btn = gr.Button("Generate (批量生成)", variant="primary")
                            srt_retry_btn = gr.Button(
                                "🔁 Retry Failed (0)",
                                variant="secondary",
                                visible=False,
                            )
                            # Hidden button for chaining the blocks
                            srt_next_trigger = gr.Button("Hidden Next Trigger", visible=False)

                        with gr.Column(scale=3):
                            srt_progress_bar = gr.Slider(
                                label="Progress (进度)",
                                minimum=0,
                                maximum=100,
                                value=0,
                                step=1,
                                interactive=False,
                            )
                            srt_status = gr.Textbox(
                                label="Log / Status (日志/状态)",
                                lines=20,
                                max_lines=40,
                                interactive=False,
                            )

                    # State: failed entries [(idx, text)], full log_lines list, and pending blocks list
                    srt_failed_state = gr.State([])
                    srt_log_state = gr.State([])
                    srt_pending_blocks_state = gr.State([])

                    # --- Block preview: updates CheckboxGroup when SRT or block size changes ---
                    def _update_blocks(srt_file_obj, block_size):
                        """Parse SRT and return block choices for the CheckboxGroup."""
                        if srt_file_obj is None:
                            return (
                                gr.update(choices=[], value=[]),
                                "📂 Upload an SRT file to see available blocks.",
                            )
                        try:
                            srt_path = getattr(srt_file_obj, "name", None) or str(srt_file_obj)
                            with open(srt_path, "r", encoding="utf-8", errors="replace") as _f:
                                _content = _f.read()
                            all_entries = _parse_srt(_content)
                            n = max(int(block_size or 10), 1)
                            num_blocks = math.ceil(len(all_entries) / n) if all_entries else 0
                            choices = [
                                f"Bloco {i + 1}  (#{all_entries[i * n][0]} – #{all_entries[min((i + 1) * n, len(all_entries)) - 1][0]})"
                                for i in range(num_blocks)
                            ]
                            info = (
                                f"**{len(all_entries)} legendas → {num_blocks} bloco(s) de {n}** "
                                f"{'(último bloco menor)' if len(all_entries) % n else ''}"
                            )
                            return gr.update(choices=choices, value=[]), info
                        except Exception as _e:
                            return gr.update(choices=[], value=[]), f"⚠️ Erro ao ler SRT: {_e}"

                    srt_file.change(
                        _update_blocks,
                        inputs=[srt_file, srt_block_size],
                        outputs=[srt_blocks_sel, srt_block_info],
                    )
                    srt_block_size.change(
                        _update_blocks,
                        inputs=[srt_file, srt_block_size],
                        outputs=[srt_blocks_sel, srt_block_info],
                    )

                    def run_srt_batch(
                        ref_aud, ref_txt: str, use_xvec: bool,
                        prompt_file_obj,
                        lang_disp: str, instruct: str,
                        srt_file_obj, out_dir: str, fmt: str,
                        block_size, selected_blocks,
                        auto_retry: bool, auto_next: bool, pending_blocks_state, log_lines_state, is_chain_triggered=False
                    ):
                        import soundfile as sf
                        import subprocess
                        import gc

                        try:
                            if srt_file_obj is None:
                                yield 0, "❌ SRT file is required (必须上传字幕文件).", [], [], []
                                return
                            if not out_dir or not out_dir.strip():
                                yield 0, "❌ Output folder is required (必须填写输出文件夹路径).", [], [], []
                                return

                            out_dir = out_dir.strip()
                            os.makedirs(out_dir, exist_ok=True)

                            srt_path = getattr(srt_file_obj, "name", None) or str(srt_file_obj)
                            with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
                                srt_content = f.read()

                            entries = _parse_srt(srt_content)
                            if not entries:
                                yield 0, "❌ No subtitle entries found in SRT file.", [], [], []
                                return

                            # --- Filter by selected blocks ---
                            block_sz = max(int(block_size or 10), 1)
                            blocks_to_run = selected_blocks or []
                            
                            if is_chain_triggered:
                                # We are continuing from a previous chain. Use pending
                                blocks_to_run = pending_blocks_state
                            else:
                                # If all blocks are unselected, that means run ALL blocks, 
                                # but for the auto-next logic to work, we need an explicit list of block labels.
                                if not blocks_to_run:
                                    num_blocks = math.ceil(len(entries) / block_sz) if entries else 0
                                    blocks_to_run = [
                                        f"Bloco {i + 1}  (#{entries[i * block_sz][0]} – #{entries[min((i + 1) * block_sz, len(entries)) - 1][0]})"
                                        for i in range(num_blocks)
                                    ]

                            current_block_label = blocks_to_run[0] if blocks_to_run else None
                            new_pending_blocks = blocks_to_run[1:] if len(blocks_to_run) > 1 else []

                            if current_block_label:
                                b = int(current_block_label.split()[1]) - 1
                                selected_block_idxs = set(range(b * block_sz, min((b + 1) * block_sz, len(entries))))
                                entries = [e for i, e in enumerate(entries) if i in selected_block_idxs]
                            
                            if not entries:
                                yield 0, "❌ Block parsing error: produced no entries.", [], [], []
                                return

                            language = lang_map.get(lang_disp, "Auto")
                            instruct_val = (instruct or "").strip() or None
                            kwargs = _gen_common_kwargs()

                            # --- determine voice source ---
                            use_prompt_file = prompt_file_obj is not None
                            voice_items = None

                            if use_prompt_file:
                                path = getattr(prompt_file_obj, "name", None) or getattr(prompt_file_obj, "path", None) or str(prompt_file_obj)
                                payload = torch.load(path, map_location="cpu", weights_only=True)
                                if not isinstance(payload, dict) or "items" not in payload:
                                    yield 0, "❌ Invalid voice file format (文件格式不正确)."
                                    return
                                items_raw = payload["items"]
                                if not isinstance(items_raw, list) or len(items_raw) == 0:
                                    yield 0, "❌ Empty voice items (音色为空)."
                                    return
                                voice_items: List[VoiceClonePromptItem] = []
                                for d in items_raw:
                                    if not isinstance(d, dict):
                                        yield 0, "❌ Invalid item format in file (文件内部格式错误)."
                                        return
                                    ref_code = d.get("ref_code", None)
                                    if ref_code is not None and not torch.is_tensor(ref_code):
                                        ref_code = torch.tensor(ref_code)
                                    ref_spk = d.get("ref_spk_embedding", None)
                                    if ref_spk is None:
                                        yield 0, "❌ Missing ref_spk_embedding (缺少说话人向量).", [], [], []
                                        return
                                    if not torch.is_tensor(ref_spk):
                                        ref_spk = torch.tensor(ref_spk)
                                    voice_items.append(
                                        VoiceClonePromptItem(
                                            ref_code=ref_code,
                                            ref_spk_embedding=ref_spk,
                                            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                                            icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                                            ref_text=d.get("ref_text", None),
                                        )
                                    )
                            else:
                                if at is None:
                                    yield 0, "❌ Reference audio or Voice File is required (必须上传参考音频或音色文件).", [], [], []
                                    return
                                if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                    yield 0, "❌ Reference text is required when x-vector only is NOT enabled.", [], [], []
                                    return

                            
                            log_lines = list(log_lines_state) if is_chain_triggered else []
                            if not is_chain_triggered:
                                log_lines.extend([
                                    f"Voice source: {'Voice File (.pt)' if use_prompt_file else 'Reference Audio'}",
                                ])
                            
                            if current_block_label:
                                log_lines.append(
                                    f"\n▶️ Starting {current_block_label} "
                                    f"({len(entries)} subtitle(s))"
                                )
                                if new_pending_blocks:
                                    log_lines.append(f"   (Queue: {len(new_pending_blocks)} block(s) pending after this)")

                            failed_entries: List[Tuple[int, str]] = []
                            yield 0, "\n".join(log_lines), failed_entries, log_lines, new_pending_blocks
                            total_t0 = time.time()

                            for i, (idx, text) in enumerate(entries):
                                pct = int(i / len(entries) * 100)

                                # Skip entries that are too short to produce quality audio
                                if len(text.split()) < 2:
                                    log_lines.append(f"⚠️  [{i+1}/{len(entries)}] #{idx} — Skipped (text too short: {repr(text)})")
                                    done_pct = int((i + 1) / len(entries) * 100)
                                    yield done_pct, "\n".join(log_lines[-40:]), failed_entries, log_lines, new_pending_blocks
                                    continue

                                preview_pre = text[:80] + "..." if len(text) > 80 else text
                                vram_info = _vram_status()
                                vram_tag = f" [{vram_info}]" if vram_info else ""
                                pending_line = f"⏳ [{i+1}/{len(entries)}] #{idx}{vram_tag} — Generating: {preview_pre}"
                                log_lines.append(pending_line)
                                # Show full text in log so user can verify correct parsing
                                if len(text) > 80:
                                    log_lines.append(f"   Full text ({len(text)} chars): {text}")
                                yield pct, "\n".join(log_lines[-40:]), failed_entries, log_lines, new_pending_blocks
                                t0 = time.time()
                                try:
                                    gen_kwargs = dict(
                                        text=text,
                                        language=language,
                                        instruct=instruct_val,
                                        **kwargs,
                                    )
                                    if use_prompt_file:
                                        gen_kwargs["voice_clone_prompt"] = voice_items
                                    else:
                                        gen_kwargs["ref_audio"] = at
                                        gen_kwargs["ref_text"] = (ref_txt.strip() if ref_txt else None)
                                        gen_kwargs["x_vector_only_mode"] = bool(use_xvec)

                                    # [Option B] Dynamic max_new_tokens: cap based on text length.
                                    # Use character count (more reliable than word count for
                                    # punctuation-heavy or non-Latin text).
                                    # ~12 audio tokens per character is a safe upper bound.
                                    # Cap raised to 8192 to avoid truncating longer subtitle lines.
                                    if "max_new_tokens" not in gen_kwargs or gen_kwargs.get("max_new_tokens") is None:
                                        char_count = max(len(text), 1)
                                        gen_kwargs["max_new_tokens"] = min(char_count * 12 + 300, 8192)

                                    # [Option A] Run generation in a thread with a hard timeout.
                                    # If the model hangs (OOM deadlock, infinite loop, CUDA stall),
                                    # we skip the entry and continue the batch instead of freezing forever.
                                    _GEN_TIMEOUT = 180  # seconds per entry
                                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

                                    def _run_generation():
                                        with torch.no_grad():
                                            return tts.generate_voice_clone(**gen_kwargs)

                                    with ThreadPoolExecutor(max_workers=1) as _pool:
                                        _fut = _pool.submit(_run_generation)
                                        try:
                                            wavs, sr = _fut.result(timeout=_GEN_TIMEOUT)
                                        except FuturesTimeout:
                                            raise RuntimeError(
                                                f"Generation timed out after {_GEN_TIMEOUT}s — "
                                                "entry skipped (possible CUDA stall or infinite loop)"
                                            )


                                    elapsed = time.time() - t0
                                    ext = (fmt or "MP3").lower()
                                    wav_path = os.path.join(out_dir, f"{idx:04d}.wav")
                                    sf.write(wav_path, wavs[0], sr)
                                    if ext == "wav":
                                        out_path = wav_path
                                    else:
                                        out_path = os.path.join(out_dir, f"{idx:04d}.{ext}")
                                        try:
                                            if ext == "mp3":
                                                cmd = ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", out_path]
                                            else:  # mp4
                                                cmd = ["ffmpeg", "-y", "-i", wav_path,
                                                       "-c:a", "aac", "-b:a", "192k",
                                                       "-vn", out_path]
                                            subprocess.run(cmd, check=True, capture_output=True)
                                            os.remove(wav_path)
                                        except FileNotFoundError:
                                            log_lines.append(f"   ⚠️ ffmpeg not found — saved as WAV instead")
                                            out_path = wav_path
                                        except subprocess.CalledProcessError as ce:
                                            log_lines.append(f"   ⚠️ ffmpeg error: {ce.stderr.decode(errors='replace')[:200]}")
                                            out_path = wav_path
                                    preview = text[:60] + "..." if len(text) > 60 else text
                                    log_lines[-1] = f"✅ [{i+1}/{len(entries)}] #{idx} — {elapsed:.1f}s → {out_path}"
                                    log_lines.append(f"   {preview}")
                                except Exception as e:
                                    log_lines[-1] = f"❌ [{i+1}/{len(entries)}] #{idx} — FAILED: {type(e).__name__}: {e}"
                                    failed_entries.append((idx, text))
                                finally:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    gc.collect()

                                done_pct = int((i + 1) / len(entries) * 100)
                                yield done_pct, "\n".join(log_lines[-40:]), failed_entries, log_lines, new_pending_blocks

                            total_elapsed = time.time() - total_t0
                            vram_final = _vram_status()
                            vram_suffix = f" | {vram_final}" if vram_final else ""
                            n_failed = len(failed_entries)
                            log_lines.append(f"\n🏁 Done! {len(entries)} file(s) in {total_elapsed:.1f}s → {out_dir}{vram_suffix}")
                            if n_failed > 0:
                                if auto_retry:
                                    log_lines.append(f"⚠️  {n_failed} entry(ies) failed — 🔄 Auto-retry enabled, retrying now...")
                                else:
                                    log_lines.append(f"⚠️  {n_failed} entry(ies) failed — click 🔁 to reprocess.")

                            # --- Auto-retry loop ---
                            retry_round = 0
                            while auto_retry and failed_entries:
                                retry_round += 1
                                prev_failed_count = len(failed_entries)
                                to_retry = list(failed_entries)
                                failed_entries = []
                                log_lines.append(f"\n─── 🔄 Auto-retry round {retry_round}: {len(to_retry)} entr(ies) ───\n")
                                yield 0, "\n".join(log_lines[-40:]), failed_entries, log_lines, new_pending_blocks

                                for i, (idx, text) in enumerate(to_retry):
                                    pct = int(i / len(to_retry) * 100)
                                    preview_pre = text[:80] + "..." if len(text) > 80 else text
                                    vram_info = _vram_status()
                                    vram_tag = f" [{vram_info}]" if vram_info else ""
                                    log_lines.append(f"⏳ [retry-{retry_round} {i+1}/{len(to_retry)}] #{idx}{vram_tag} — {preview_pre}")
                                    yield pct, "\n".join(log_lines[-40:]), failed_entries, log_lines, new_pending_blocks
                                    t0 = time.time()
                                    try:
                                        gen_kwargs = dict(text=text, language=language, instruct=instruct_val, **kwargs)
                                        if use_prompt_file:
                                            gen_kwargs["voice_clone_prompt"] = voice_items
                                        else:
                                            gen_kwargs["ref_audio"] = at
                                            gen_kwargs["ref_text"] = (ref_txt.strip() if ref_txt else None)
                                            gen_kwargs["x_vector_only_mode"] = bool(use_xvec)
                                        char_count = max(len(text), 1)
                                        gen_kwargs["max_new_tokens"] = min(char_count * 12 + 300, 8192)

                                        _GEN_TIMEOUT = 180
                                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

                                        def _run_gen_retry():
                                            with torch.no_grad():
                                                return tts.generate_voice_clone(**gen_kwargs)

                                        with ThreadPoolExecutor(max_workers=1) as _pool:
                                            _fut = _pool.submit(_run_gen_retry)
                                            try:
                                                wavs, sr = _fut.result(timeout=_GEN_TIMEOUT)
                                            except FuturesTimeout:
                                                raise RuntimeError(f"Timed out after {_GEN_TIMEOUT}s")

                                        elapsed = time.time() - t0
                                        ext = (fmt or "MP3").lower()
                                        wav_path = os.path.join(out_dir, f"{idx:04d}.wav")
                                        sf.write(wav_path, wavs[0], sr)
                                        if ext == "wav":
                                            out_path = wav_path
                                        else:
                                            out_path = os.path.join(out_dir, f"{idx:04d}.{ext}")
                                            try:
                                                if ext == "mp3":
                                                    cmd = ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", out_path]
                                                else:
                                                    cmd = ["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac", "-b:a", "192k", "-vn", out_path]
                                                subprocess.run(cmd, check=True, capture_output=True)
                                                os.remove(wav_path)
                                            except (FileNotFoundError, subprocess.CalledProcessError):
                                                out_path = wav_path
                                        log_lines[-1] = f"✅ [retry-{retry_round} {i+1}/{len(to_retry)}] #{idx} — {elapsed:.1f}s → {out_path}"
                                        log_lines.append(f"   {text[:60]}{'...' if len(text) > 60 else ''}")
                                    except Exception as e:
                                        log_lines[-1] = f"❌ [retry-{retry_round} {i+1}/{len(to_retry)}] #{idx} — {type(e).__name__}: {e}"
                                        failed_entries.append((idx, text))
                                    finally:
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                        gc.collect()

                                    done_pct = int((i + 1) / len(to_retry) * 100)
                                    yield done_pct, "\n".join(log_lines[-40:]), failed_entries, log_lines, new_pending_blocks

                                # If no improvement, stop to avoid infinite loop
                                if len(failed_entries) >= prev_failed_count:
                                    log_lines.append(f"\n⚠️ Auto-retry round {retry_round}: no improvement — stopping.")
                                    break
                                if failed_entries:
                                    log_lines.append(f"✅ Round {retry_round} recovered {prev_failed_count - len(failed_entries)}, {len(failed_entries)} still failing.")

                            n_failed = len(failed_entries)
                            yield 100, "\n".join(log_lines[-40:]), failed_entries, log_lines, new_pending_blocks

                        except Exception as e:
                            yield 0, f"❌ {type(e).__name__}: {e}", [], [], []

                    def run_srt_retry(
                        failed_entries, log_lines_raw,
                        ref_aud, ref_txt: str, use_xvec: bool,
                        prompt_file_obj,
                        lang_disp: str, instruct: str,
                        out_dir: str, fmt: str,
                    ):
                        import soundfile as sf
                        import subprocess
                        import gc

                        if not failed_entries:
                            yield 0, "\n".join((log_lines_raw or [])[-40:]), [], log_lines_raw or [], gr.update(visible=False)
                            return

                        log_lines = list(log_lines_raw or [])
                        n = len(failed_entries)
                        log_lines.append(f"\n─── 🔁 Retrying {n} failed entry(ies) ───\n")
                        yield 0, "\n".join(log_lines[-40:]), failed_entries, log_lines, gr.update(visible=False)

                        out_dir = (out_dir or "").strip()
                        language = lang_map.get(lang_disp, "Auto")
                        instruct_val = (instruct or "").strip() or None
                        kwargs = _gen_common_kwargs()
                        use_prompt_file = prompt_file_obj is not None
                        voice_items = None
                        at = None

                        try:
                            if use_prompt_file:
                                path = getattr(prompt_file_obj, "name", None) or getattr(prompt_file_obj, "path", None) or str(prompt_file_obj)
                                payload = torch.load(path, map_location="cpu", weights_only=True)
                                items_raw = payload.get("items", [])
                                voice_items: List[VoiceClonePromptItem] = []
                                for d in items_raw:
                                    ref_code = d.get("ref_code", None)
                                    if ref_code is not None and not torch.is_tensor(ref_code):
                                        ref_code = torch.tensor(ref_code)
                                    ref_spk = d.get("ref_spk_embedding")
                                    if not torch.is_tensor(ref_spk):
                                        ref_spk = torch.tensor(ref_spk)
                                    voice_items.append(
                                        VoiceClonePromptItem(
                                            ref_code=ref_code,
                                            ref_spk_embedding=ref_spk,
                                            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                                            icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                                            ref_text=d.get("ref_text", None),
                                        )
                                    )
                            else:
                                at = _audio_to_tuple(ref_aud)

                            new_failed: List[Tuple[int, str]] = []
                            total_t0 = time.time()

                            for i, (idx, text) in enumerate(failed_entries):
                                pct = int(i / n * 100)
                                preview_pre = text[:50] + "..." if len(text) > 50 else text
                                vram_info = _vram_status()
                                vram_tag = f" [{vram_info}]" if vram_info else ""
                                log_lines.append(f"⏳ [retry {i+1}/{n}] #{idx}{vram_tag} — Generating: {preview_pre}")
                                yield pct, "\n".join(log_lines[-40:]), new_failed, log_lines, gr.update(visible=False)
                                t0 = time.time()
                                try:
                                    gen_kwargs = dict(text=text, language=language, instruct=instruct_val, **kwargs)
                                    if use_prompt_file:
                                        gen_kwargs["voice_clone_prompt"] = voice_items
                                    else:
                                        gen_kwargs["ref_audio"] = at
                                        gen_kwargs["ref_text"] = (ref_txt.strip() if ref_txt else None)
                                        gen_kwargs["x_vector_only_mode"] = bool(use_xvec)
                                    char_count = max(len(text), 1)
                                    gen_kwargs.setdefault("max_new_tokens", min(char_count * 12 + 300, 8192))

                                    _GEN_TIMEOUT = 180
                                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

                                    def _run_generation():
                                        with torch.no_grad():
                                            return tts.generate_voice_clone(**gen_kwargs)

                                    with ThreadPoolExecutor(max_workers=1) as _pool:
                                        _fut = _pool.submit(_run_generation)
                                        try:
                                            wavs, sr = _fut.result(timeout=_GEN_TIMEOUT)
                                        except FuturesTimeout:
                                            raise RuntimeError(
                                                f"Generation timed out after {_GEN_TIMEOUT}s — "
                                                "entry skipped (possible CUDA stall or infinite loop)"
                                            )

                                    elapsed = time.time() - t0
                                    ext = (fmt or "MP3").lower()
                                    wav_path = os.path.join(out_dir, f"{idx:04d}.wav")
                                    sf.write(wav_path, wavs[0], sr)
                                    if ext == "wav":
                                        out_path = wav_path
                                    else:
                                        out_path = os.path.join(out_dir, f"{idx:04d}.{ext}")
                                        try:
                                            if ext == "mp3":
                                                cmd = ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", out_path]
                                            else:
                                                cmd = ["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac", "-b:a", "192k", "-vn", out_path]
                                            subprocess.run(cmd, check=True, capture_output=True)
                                            os.remove(wav_path)
                                        except FileNotFoundError:
                                            log_lines.append(f"   ⚠️ ffmpeg not found — saved as WAV instead")
                                            out_path = wav_path
                                        except subprocess.CalledProcessError as ce:
                                            log_lines.append(f"   ⚠️ ffmpeg error: {ce.stderr.decode(errors='replace')[:200]}")
                                            out_path = wav_path
                                    preview = text[:60] + "..." if len(text) > 60 else text
                                    log_lines[-1] = f"✅ [retry {i+1}/{n}] #{idx} — {elapsed:.1f}s → {out_path}"
                                    log_lines.append(f"   {preview}")
                                except Exception as e:
                                    log_lines[-1] = f"❌ [retry {i+1}/{n}] #{idx} — FAILED: {type(e).__name__}: {e}"
                                    new_failed.append((idx, text))
                                finally:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    gc.collect()

                                done_pct = int((i + 1) / n * 100)
                                yield done_pct, "\n".join(log_lines[-40:]), new_failed, log_lines, gr.update(visible=False)

                            total_elapsed = time.time() - total_t0
                            n_failed = len(new_failed)
                            log_lines.append(f"\n🏁 Retry done! {n - n_failed}/{n} recovered in {total_elapsed:.1f}s")
                            if n_failed > 0:
                                log_lines.append(f"⚠️  {n_failed} entry(ies) still failing.")
                            retry_update = gr.update(visible=n_failed > 0, value=f"🔁 Retry Failed ({n_failed})")
                            yield 100, "\n".join(log_lines[-40:]), new_failed, log_lines, retry_update

                        except Exception as e:
                            log_lines.append(f"\n❌ Retry error: {type(e).__name__}: {e}")
                            yield 0, "\n".join(log_lines[-40:]), failed_entries, log_lines, gr.update(visible=True, value=f"🔁 Retry Failed ({len(failed_entries)})")

                    # Step 1: When user clicks "Generate", it starts the first block
                    # The generator returns the updated state as the final tuple element.
                    srt_btn.click(
                        run_srt_batch,
                        inputs=[
                            srt_ref_audio, srt_ref_text, srt_xvec_only, srt_prompt_file,
                            srt_lang, srt_instruct,
                            srt_file, srt_out_dir, srt_format,
                            srt_block_size, srt_blocks_sel,
                            srt_auto_retry, srt_auto_next, srt_pending_blocks_state, srt_log_state
                        ],
                        outputs=[srt_progress_bar, srt_status, srt_failed_state, srt_log_state, srt_pending_blocks_state],
                    ).then(
                        lambda pending, auto_next: gr.update(visible=bool(pending and auto_next)),
                        inputs=[srt_pending_blocks_state, srt_auto_next],
                        outputs=[srt_next_trigger]
                    ).then(
                        lambda n_failed: gr.update(visible=n_failed > 0, value=f"🔁 Retry Failed ({n_failed})"),
                        inputs=[lambda f: len(f) if f else 0],
                        outputs=[srt_retry_btn]
                    )

                    # Step 2: Hidden trigger for next blocks
                    # Is clicked virtually if the visibility becomes true (which means there are pending blocks and auto_next is True)
                    srt_next_trigger.change(
                        lambda is_vis: None if not is_vis else True, # return a dummy to trigger the chain
                        inputs=[srt_next_trigger], outputs=[]
                    ).then(
                        # We pass `is_chain_triggered=True` via lambda: True
                        run_srt_batch,
                        inputs=[
                            srt_ref_audio, srt_ref_text, srt_xvec_only, srt_prompt_file,
                            srt_lang, srt_instruct,
                            srt_file, srt_out_dir, srt_format,
                            srt_block_size, srt_blocks_sel,
                            srt_auto_retry, srt_auto_next, srt_pending_blocks_state, srt_log_state, gr.State(True)
                        ],
                        outputs=[srt_progress_bar, srt_status, srt_failed_state, srt_log_state, srt_pending_blocks_state],
                    ).then(
                        lambda pending, auto_next: gr.update(visible=bool(pending and auto_next)),
                        inputs=[srt_pending_blocks_state, srt_auto_next],
                        outputs=[srt_next_trigger]
                    ).then(
                        lambda n_failed: gr.update(visible=n_failed > 0, value=f"🔁 Retry Failed ({n_failed})"),
                        inputs=[lambda f: len(f) if f else 0],
                        outputs=[srt_retry_btn]
                    )

                    srt_retry_btn.click(
                        run_srt_retry,
                        inputs=[srt_failed_state, srt_log_state, srt_ref_audio, srt_ref_text, srt_xvec_only, srt_prompt_file, srt_lang, srt_instruct, srt_out_dir, srt_format],
                        outputs=[srt_progress_bar, srt_status, srt_failed_state, srt_log_state, srt_retry_btn],
                    )

                with gr.Tab("🎨 Voice Design"):
                    gr.Markdown(
                        """
### 🎨 Voice Design — Criar voz com descrição de estilo
Descreva a voz que você quer: gênero, idade, tom emocional, ritmo, idioma nativo...  
O modelo `Qwen3-TTS-12Hz-1.7B-VoiceDesign` é carregado **automaticamente** na primeira geração.  
> ⚠️ Requer a pasta `Qwen3-TTS-12Hz-1.7B-VoiceDesign/` localmente (ou conexão com HuggingFace).
"""
                    )
                    with gr.Row():
                        # ── Coluna esquerda: inputs ──────────────────────────
                        with gr.Column(scale=2):
                            vd_text = gr.Textbox(
                                label="📝 Texto para sintetizar (Text to Synthesize)",
                                lines=5,
                                placeholder="Digite o texto que será falado com a voz criada...\nEx: It's in the top drawer... wait, it's empty? No way!",
                            )
                            vd_lang = gr.Dropdown(
                                label="🌐 Idioma (Language)",
                                choices=lang_choices_disp or ["Auto"],
                                value="Auto",
                                interactive=True,
                                info="Defina o idioma do texto. 'Auto' detecta automaticamente.",
                            )
                            vd_design = gr.Textbox(
                                label="🎭 Descrição da voz (Voice Design Instruction)",
                                lines=5,
                                placeholder=(
                                    "Descreva a voz em detalhes:\n"
                                    "- Gênero: male / female\n"
                                    "- Idade: e.g. '25 years old'\n"
                                    "- Tom: calm / energetic / sad / angry / whispering\n"
                                    "- Ritmo: slow / fast / natural\n"
                                    "- Estilo: storyteller / news anchor / casual conversation\n"
                                    "Ex: 'Female, 30, warm and calm narrator with a slight smile in the voice'"
                                ),
                                info="Quanto mais detalhada a descrição, melhor o resultado.",
                            )

                            gr.Examples(
                                label="⚡ Exemplos rápidos (clique para carregar)",
                                examples=[
                                    ["It's in the top drawer... wait, it's empty? No way, that's impossible!",
                                     "English",
                                     "Female, 28 years old, disbelief turning into panic, voice slightly trembling, fast speech pattern"],
                                    ["Welcome, and thank you for joining us today. Let's begin.",
                                     "English",
                                     "Male, 45 years old, deep calm baritone, professional news anchor style, clear articulation, slow and authoritative"],
                                    ["Hey! Did you hear that? Something moved in the dark...",
                                     "English",
                                     "Female, 20 years old, terrified whispering, very low volume, tense and breathless, close-mic sensation"],
                                    ["Haha! Come on, it'll be fun! Trust me on this one!",
                                     "English",
                                     "Male, 22 years old, cheerful and energetic, bright timbre, fast animated speech, enthusiastic"],
                                    ["Não se preocupe. Tudo vai ficar bem, eu prometo.",
                                     "Portuguese",
                                     "Female, 35 years old, warm and comforting, slow reassuring tone, slightly soft voice, maternal feeling"],
                                    ["Attention all units. We have a situation. Proceed with caution.",
                                     "English",
                                     "Male, 40 years old, military commander, tense and urgent, clipped speech, low and controlled"],
                                ],
                                inputs=[vd_text, vd_lang, vd_design],
                            )

                            with gr.Accordion("💾 Salvar saída (Save Output)", open=False):
                                vd_out_dir = gr.Textbox(
                                    label="Pasta de saída (Output Folder)",
                                    placeholder="Ex: C:\\Users\\Voce\\Desktop\\vozes_criadas",
                                    info="Deixe vazio para não salvar em arquivo. O áudio ainda aparece no player.",
                                )

                            vd_btn = gr.Button("🎨 Generate with Voice Design", variant="primary", size="lg")

                        # ── Coluna direita: outputs ──────────────────────────
                        with gr.Column(scale=3):
                            vd_status = gr.Textbox(
                                label="⏱️ Status",
                                lines=3,
                                interactive=False,
                                value="Aguardando geração...",
                            )
                            vd_audio_out = gr.Audio(
                                label="🔊 Áudio Gerado (Generated Audio)",
                                type="numpy",
                            )
                            gr.Markdown(
                                """
**💡 Dica:** Após gerar, você pode usar este áudio como **referência na aba Clone & Generate**  
para clonar este estilo de voz e sintetizar qualquer outro texto!
"""
                            )

                    def run_voice_design_tab(
                        text: str,
                        lang_disp: str,
                        design: str,
                        out_dir: str,
                        progress=gr.Progress(track_tqdm=False),
                    ):
                        import soundfile as sf
                        global _vd_tts

                        if not text or not text.strip():
                            yield "❌ Texto é obrigatório.", None
                            return
                        if not design or not design.strip():
                            yield "❌ Descrição da voz é obrigatória.", None
                            return

                        # Lazy load the VoiceDesign model
                        if _vd_tts is None:
                            progress(0, desc="Carregando modelo VoiceDesign...")
                            yield "⏳ Carregando modelo VoiceDesign pela primeira vez...\n(pode demorar 1-2 minutos)", None
                            try:
                                import os as _os
                                # Detect local path relative to the Base model checkpoint
                                base_ckpt_dir = _os.path.dirname(_os.path.abspath(ckpt)) if _os.path.isdir(ckpt) else _os.path.dirname(ckpt)
                                local_vd = _os.path.join(_os.path.dirname(_os.path.abspath(ckpt)), "..", "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
                                # Also try same directory as script
                                local_vd2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
                                vd_ckpt = None
                                for candidate in [local_vd, local_vd2, "Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"]:
                                    normalized = _os.path.normpath(candidate)
                                    if _os.path.isdir(normalized):
                                        vd_ckpt = normalized
                                        break
                                if vd_ckpt is None:
                                    vd_ckpt = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"  # fallback to HuggingFace

                                device = gen_kwargs_default.get("device_map", "cuda:0") if "device_map" in gen_kwargs_default else "cuda:0"
                                kwargs_load = {k: v for k, v in gen_kwargs_default.items() if k not in ("max_new_tokens",)}
                                _vd_tts = Qwen3TTSModel.from_pretrained(
                                    vd_ckpt,
                                    device_map="cuda:0",
                                    dtype=torch.bfloat16,
                                    attn_implementation="eager",
                                )
                            except Exception as e:
                                hint = (
                                    "\n\n💡 Se o modelo não foi baixado ainda:\n"
                                    "  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign "
                                    "--local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                                )
                                yield f"❌ Falha ao carregar modelo VoiceDesign:\n{type(e).__name__}: {e}{hint}", None
                                return

                        progress(0.3, desc="Gerando áudio...")
                        yield "⏳ Gerando áudio com Voice Design...", None
                        t0 = time.time()
                        try:
                            language = lang_map.get(lang_disp, "Auto") if lang_map else lang_disp
                            wavs, sr = _vd_tts.generate_voice_design(
                                text=text.strip(),
                                language=language,
                                instruct=design.strip(),
                            )
                            elapsed = time.time() - t0
                            progress(1.0, desc="Concluído!")

                            # Save to file if requested
                            saved_msg = ""
                            if out_dir and out_dir.strip():
                                import os as _os
                                _os.makedirs(out_dir.strip(), exist_ok=True)
                                import time as _time
                                fname = f"vd_{int(_time.time())}.wav"
                                fpath = _os.path.join(out_dir.strip(), fname)
                                sf.write(fpath, wavs[0], sr)
                                saved_msg = f"\n💾 Salvo em: {fpath}"

                            yield f"✅ Concluído em {elapsed:.1f}s{saved_msg}", _wav_to_gradio_audio(wavs[0], sr)
                        except Exception as e:
                            yield f"❌ Erro na geração:\n{type(e).__name__}: {e}", None

                    vd_btn.click(
                        run_voice_design_tab,
                        inputs=[vd_text, vd_lang, vd_design, vd_out_dir],
                        outputs=[vd_status, vd_audio_out],
                    )

                with gr.Tab("🔀 Voice Blend"):
                    gr.Markdown(
                        """
### 🔀 Voice Blend — Misturar duas vozes em uma
Faça upload de dois áudios de referência e use o **slider** para controlar a proporção da mistura.  
A combinação é feita matematicamente nos *speaker embeddings* (vetores de identidade) de cada voz.
> 💡 **Dica:** Funciona melhor quando os dois áudios têm boa qualidade (5–15 segundos, sem ruído).
"""
                    )
                    with gr.Row():
                        # ── Coluna esquerda: inputs ──────────────────────────
                        with gr.Column(scale=4):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### 🎤 Voz A")
                                    blend_ref_a = gr.Audio(
                                        label="Áudio de Referência A",
                                        type="numpy",
                                    )
                                    blend_txt_a = gr.Textbox(
                                        label="Texto do Áudio A (Reference Text A)",
                                        lines=2,
                                        placeholder="O que foi dito no áudio A (opcional — deixe vazio para usar só o x-vector).",
                                    )
                                with gr.Column():
                                    gr.Markdown("#### 🎤 Voz B")
                                    blend_ref_b = gr.Audio(
                                        label="Áudio de Referência B",
                                        type="numpy",
                                    )
                                    blend_txt_b = gr.Textbox(
                                        label="Texto do Áudio B (Reference Text B)",
                                        lines=2,
                                        placeholder="O que foi dito no áudio B (opcional — deixe vazio para usar só o x-vector).",
                                    )

                            blend_ratio = gr.Slider(
                                label="🎚️ Proporção: ← 100% Voz A  |  100% Voz B →",
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=1,
                                info="0 = 100% Voz A  |  50 = Mistura igual  |  100 = 100% Voz B",
                            )

                            blend_text = gr.Textbox(
                                label="📝 Texto para sintetizar (Target Text)",
                                lines=4,
                                placeholder="Digite o texto que você quer sintetizar com a voz misturada...",
                            )
                            blend_lang = gr.Dropdown(
                                label="🌐 Idioma (Language)",
                                choices=lang_choices_disp or ["Auto"],
                                value="Auto",
                                interactive=True,
                            )
                            blend_instruct = gr.Textbox(
                                label="💬 Instrução de estilo (opcional)",
                                lines=2,
                                placeholder="Ex: Fale devagar e com calma. (não tem efeito no modelo Base)",
                            )
                            blend_btn = gr.Button("🔀 Generate Blended Voice", variant="primary", size="lg")

                        # ── Coluna direita: outputs ──────────────────────────
                        with gr.Column(scale=3):
                            blend_status = gr.Textbox(
                                label="⏱️ Status",
                                lines=3,
                                interactive=False,
                                value="Aguardando geração...",
                            )
                            blend_audio_out = gr.Audio(
                                label="🔊 Voz Misturada (Blended Voice)",
                                type="numpy",
                            )
                            blend_file_out = gr.File(
                                label="🎤 Arquivo de Voz Misturada (.pt) — clique para baixar"
                            )
                            gr.Markdown(
                                """
**ℹ️ Como funciona:**  
1. Extrai o *speaker embedding* (identidade vocal) de cada áudio  
2. Faz a média ponderada pelo slider  
3. Gera o áudio com a identidade misturada  

Tente variações de 30/70, 50/50, 70/30 para encontrar o blend perfeito!
"""
                            )

                    def run_voice_blend(
                        ref_a, txt_a: str,
                        ref_b, txt_b: str,
                        ratio: float,
                        text: str, lang_disp: str, instruct: str,
                        progress=gr.Progress(track_tqdm=False),
                    ):
                        """Blend two voice embeddings and synthesize target text."""
                        if ref_a is None or ref_b is None:
                            yield "❌ Ambos os áudios de referência são obrigatórios.", None, None
                            return
                        if not text or not text.strip():
                            yield "❌ Texto alvo é obrigatório.", None, None
                            return

                        progress(0.1, desc="Extraindo embedding da Voz A...")
                        yield "⏳ [1/4] Extraindo embedding da Voz A...", None, None

                        try:
                            at_a = _audio_to_tuple(ref_a)
                            at_b = _audio_to_tuple(ref_b)
                            if at_a is None:
                                yield "❌ Áudio A inválido.", None, None
                                return
                            if at_b is None:
                                yield "❌ Áudio B inválido.", None, None
                                return

                            kwargs = _gen_common_kwargs()
                            txt_a_v = (txt_a or "").strip() or None
                            txt_b_v = (txt_b or "").strip() or None

                            # Extract prompt items for each voice
                            items_a = tts.create_voice_clone_prompt(
                                ref_audio=at_a,
                                ref_text=txt_a_v,
                                x_vector_only_mode=(txt_a_v is None),
                            )

                            progress(0.3, desc="Extraindo embedding da Voz B...")
                            yield "⏳ [2/4] Extraindo embedding da Voz B...", None, None

                            items_b = tts.create_voice_clone_prompt(
                                ref_audio=at_b,
                                ref_text=txt_b_v,
                                x_vector_only_mode=(txt_b_v is None),
                            )

                            progress(0.55, desc="Misturando embeddings...")
                            yield f"⏳ [3/4] Misturando embeddings (A={100-int(ratio)}% / B={int(ratio)}%)...", None, None

                            # Weighted blend of speaker embeddings
                            alpha = ratio / 100.0  # 0.0 = full A, 1.0 = full B
                            spk_a = items_a[0].ref_spk_embedding.float()
                            spk_b = items_b[0].ref_spk_embedding.float()
                            blended_spk = ((1.0 - alpha) * spk_a + alpha * spk_b)

                            # Use ref_code from whichever side dominates (or A if 50/50)
                            dominant_items = items_a if alpha <= 0.5 else items_b
                            blended_item = VoiceClonePromptItem(
                                ref_code=dominant_items[0].ref_code,
                                ref_spk_embedding=blended_spk,
                                x_vector_only_mode=True,
                                icl_mode=False,
                                ref_text=None,
                            )

                            progress(0.7, desc="Sintetizando...")
                            yield "⏳ [4/4] Sintetizando com a voz misturada...", None, None

                            t0 = time.time()
                            language = lang_map.get(lang_disp, "Auto") if lang_map else lang_disp
                            instruct_val = (instruct or "").strip() or None
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                voice_clone_prompt=[blended_item],
                                instruct=instruct_val,
                                **kwargs,
                            )
                            elapsed = time.time() - t0
                            progress(1.0, desc="Concluído!")

                            pct_a = 100 - int(ratio)
                            pct_b = int(ratio)

                            # Auto-save blended voice as .pt (tempfile, same as Save/Load Voice tab)
                            import os, tempfile, torch
                            payload = {
                                "items": [
                                    {
                                        "ref_code": blended_item.ref_code,
                                        "ref_spk_embedding": blended_item.ref_spk_embedding,
                                        "x_vector_only_mode": True,
                                        "icl_mode": False,
                                        "ref_text": None,
                                    }
                                ]
                            }
                            fd, pt_path = tempfile.mkstemp(
                                prefix=f"blend_{pct_a}A_{pct_b}B_", suffix=".pt"
                            )
                            os.close(fd)
                            torch.save(payload, pt_path)

                            yield (
                                f"✅ Concluído em {elapsed:.1f}s | Mix: {pct_a}% Voz A + {pct_b}% Voz B | Modo: x-vector blend"
                            ), _wav_to_gradio_audio(wavs[0], sr), pt_path

                        except Exception as e:
                            yield f"❌ Erro: {type(e).__name__}: {e}", None, None

                    blend_btn.click(
                        run_voice_blend,
                        inputs=[
                            blend_ref_a, blend_txt_a,
                            blend_ref_b, blend_txt_b,
                            blend_ratio, blend_text, blend_lang, blend_instruct,
                        ],
                        outputs=[blend_status, blend_audio_out, blend_file_out],
                    )


        gr.Markdown(
            """
**Disclaimer (免责声明)**  
- The audio is automatically generated/synthesized by an AI model solely to demonstrate the model's capabilities; it may be inaccurate or inappropriate, does not represent the views of the developer/operator, and does not constitute professional advice. You are solely responsible for evaluating, using, distributing, or relying on this audio; to the maximum extent permitted by applicable law, the developer/operator disclaims liability for any direct, indirect, incidental, or consequential damages arising from the use of or inability to use the audio, except where liability cannot be excluded by law. Do not use this service to intentionally generate or replicate unlawful, harmful, defamatory, fraudulent, deepfake, or privacy/publicity/copyright/trademark‑infringing content; if a user prompts, supplies materials, or otherwise facilitates any illegal or infringing conduct, the user bears all legal consequences and the developer/operator is not responsible.
- 音频由人工智能模型自动生成/合成，仅用于体验与展示模型效果，可能存在不准确或不当之处；其内容不代表开发者/运营方立场，亦不构成任何专业建议。用户应自行评估并承担使用、传播或依赖该音频所产生的一切风险与责任；在适用法律允许的最大范围内，开发者/运营方不对因使用或无法使用本音频造成的任何直接、间接、附带或后果性损失承担责任（法律另有强制规定的除外）。严禁利用本服务故意引导生成或复制违法、有害、诽谤、欺诈、深度伪造、侵犯隐私/肖像/著作权/商标等内容；如用户通过提示词、素材或其他方式实施或促成任何违法或侵权行为，相关法律后果由用户自行承担，与开发者/运营方无关。
"""
        )

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0

    ckpt = _resolve_checkpoint(args)

    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    tts = Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo = build_demo(tts, ckpt, gen_kwargs_default)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    # [Option C] Gradio queue: fast heartbeat so Gradio doesn't kill long-running
    # SRT batch generators. status_update_rate=5 sends keep-alive to browser every 5s.
    demo.queue(
        default_concurrency_limit=int(args.concurrency),
        status_update_rate=5,
        max_size=20,
    ).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
