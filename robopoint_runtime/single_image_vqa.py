#!/usr/bin/env python3
"""Standalone, offline, 4-bit image Q&A using the trained RoboPoint checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws/src/vlpoint'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--question', default=None)
    parser.add_argument('--interactive', action='store_true', help='Keep model loaded for multi-turn image chat')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'robopoint_runtime/runs/robopoint_vqa_only_317677')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--output', type=Path, help='Optional JSON result file')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image not found: {args.image}')
    if not (args.model_path / 'config.json').is_file():
        parser.error(f'Model config not found: {args.model_path}')
    if args.question is not None and (not args.question.strip() or '<image>' in args.question):
        parser.error('Use a nonempty question without the reserved <image> marker.')
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    from PIL import Image, ImageOps
    from point.constants import IMAGE_TOKEN_INDEX
    from point.conversation import conv_templates
    from point.mm_utils import process_images, tokenizer_image_token
    from point.model.builder import load_pretrained_model

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run in the GPU-enabled vlpoint-inference container.')
    free, total = torch.cuda.mem_get_info()
    if free < 9 * 1024**3:
        raise RuntimeError(f'Only {free / 1024**3:.2f} GiB GPU memory free. Stop the existing VLM worker and SAM3/Isaac GPU workloads first; this script loads its own model (conservative 9 GiB free-memory check).')
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    print('Loading 4-bit model (CLIP + LLM, SAM3 disabled)...', flush=True)
    started = time.perf_counter()
    tokenizer, model, processor, context_len = load_pretrained_model(
        str(args.model_path), None, args.model_path.name,
        load_4bit=True, device='cuda', device_map='cuda:0',
    )
    model.eval()
    model.config.mm_use_sam3_conditioning = False
    tower = model.get_vision_tower()
    tower.mm_use_sam3_conditioning = False
    load_s = time.perf_counter() - started
    history = []
    records = []
    image_path = args.image.resolve()
    marker = '<im_start><image><im_end>' if getattr(model.config, 'mm_use_im_start_end', False) else '<image>'

    def ask(question):
        if not question.strip() or '<image>' in question:
            raise ValueError('Use a nonempty question without the reserved <image> marker.')
        # Rebuild each generation from the current image and retained text history.
        # Old KV caches are not retained between turns.
        turns = list(history)
        while True:
            conv = conv_templates['llava_v1'].copy()
            for index, (old_question, old_answer) in enumerate(turns):
                conv.append_message(conv.roles[0], (marker + '\n' if index == 0 else '') + old_question)
                conv.append_message(conv.roles[1], old_answer)
            conv.append_message(conv.roles[0], ('' if turns else marker + '\n') + question)
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            limit = min(context_len, model.config.max_position_embeddings)
            available = limit - (ids.shape[-1] - 1 + tower.num_patches)
            if available >= args.max_new_tokens:
                break
            if not turns:
                raise ValueError(f'Only {available} output tokens remain; shorten question or reduce --max-new-tokens.')
            turns.pop(0)
        if len(turns) < len(history):
            print('[Context full: oldest turns removed.]', flush=True)
        pixels = process_images([image], processor, model.config)
        if isinstance(pixels, list):
            pixels = [p.to(model.device, dtype=torch.float16) for p in pixels]
        else:
            pixels = pixels.to(model.device, dtype=torch.float16)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            output = model.generate(ids, images=pixels, image_sizes=[image.size],
                                    do_sample=False, max_new_tokens=args.max_new_tokens,
                                    use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        answer = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        history[:] = turns + [(question, answer)]
        result = {'model': str(args.model_path), 'image': str(image_path),
                  'question': question, 'answer': answer,
                  'load_seconds': round(load_s, 3), 'inference_seconds': round(elapsed, 3)}
        records.append(result)
        print('\n' + answer, flush=True)
        print(f'\nInference: {elapsed:.2f}s', flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records if args.interactive else result, ensure_ascii=False, indent=2) + '\n')

    print(f'Model ready. Load: {load_s:.2f}s', flush=True)
    if not args.interactive:
        ask(args.question or 'Describe what is visible in this image.')
        return
    print('Enter a question. Commands: /image PATH, /reset, /help, /quit', flush=True)
    pending = args.question
    while True:
        try:
            question = pending if pending is not None else input('You> ').strip()
            pending = None
            if not question:
                continue
            if question in ('/quit', '/exit'):
                break
            if question == '/help':
                print('/image PATH: change image and clear history; /reset: clear history; /quit: exit')
                continue
            if question == '/reset':
                history.clear()
                print('Conversation cleared. Current image retained.')
                continue
            if question.startswith('/image '):
                raw_path = question[len('/image '):].strip()
                if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] and raw_path[0] in ('"', "'"):
                    raw_path = raw_path[1:-1]
                host_root = os.environ.get('VLMOTION_HOST_ROOT', '')
                if host_root and raw_path.startswith(host_root + '/'):
                    raw_path = str(ROOT / raw_path[len(host_root) + 1:])
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = ROOT / candidate
                with Image.open(candidate) as source:
                    replacement = ImageOps.exif_transpose(source).convert('RGB')
                image = replacement
                image_path = candidate.resolve()
                history.clear()
                print(f'Image: {image_path}; conversation cleared.')
                continue
            if question.startswith('/'):
                print('Unknown command. Type /help.')
                continue
            ask(question)
        except (EOFError, KeyboardInterrupt):
            break
        except (ValueError, OSError) as exc:
            print(f'Error: {exc}', file=sys.stderr)
    print('Session closed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
