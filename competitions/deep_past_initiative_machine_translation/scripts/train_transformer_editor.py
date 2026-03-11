"""Train a pretrained seq2seq editor for Deep Past and generate submission."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm should exist, but keep training usable without it
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.deep_past_initiative_machine_translation.models.baseline import (  # noqa: E402
    COMPETITION_SLUG,
    DEFAULT_RAW_DIR,
    build_datasets,
    discover_competition_files,
    evaluate_text_predictions,
)
from competitions.deep_past_initiative_machine_translation.scripts.build_doc_memory_submission import (  # noqa: E402
    _build_sentence_docs,
    _clean_text,
    _load_lexicon_normalizer,
    _normalize_transliteration,
)


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "transformer_editor"
)
DEFAULT_METRICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "transformer_editor_metrics.json"
)
DEFAULT_SUBMISSION_PATH = (
    Path(__file__).resolve().parents[1]
    / "submissions"
    / "submission_transformer_editor.csv"
)
DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "submission_transformer_editor_report.json"
)


@dataclass(frozen=True)
class Seq2SeqConfig:
    model_name: str
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    grad_accum_steps: int
    epochs: int
    warmup_ratio: float
    max_source_length: int
    max_target_length: int
    generation_beams: int
    generation_max_new_tokens: int
    max_grad_norm: float


class TextPairDataset(Dataset):
    """Tokenized seq2seq pairs."""

    def __init__(self, inputs: dict[str, list[list[int]]], labels: list[list[int]]) -> None:
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained seq2seq editor and generate Deep Past submission."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--target-column", type=str, default=None)
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/byt5-small",
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--device-preference",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--generation-beams", type=int, default=4)
    parser.add_argument("--generation-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-sentence-memory-pairs",
        type=int,
        default=4500,
        help="Upper bound for supplemental sentence-memory pairs.",
    )
    parser.add_argument(
        "--disable-sentence-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable supplemental sentence-memory pairs from Sentences_Oare* files.",
    )
    parser.add_argument(
        "--train-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train final model on full train set after holdout.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars for training and generation.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Batches between train progress updates.",
    )
    return parser.parse_args()


def resolve_device(preferred: str) -> torch.device:
    pref = preferred.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {preferred}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_editor_prompt(source_text: str, retrieval_draft: str) -> str:
    source = _clean_text(source_text)
    draft = _clean_text(retrieval_draft)
    if not draft:
        draft = "<empty>"
    return (
        "Task: translate Akkadian transliteration into natural English.\n"
        f"Source: {source}\n"
        f"Draft: {draft}\n"
        "Final translation:"
    )


def compute_bleu_chrf(references: list[str], predictions: list[str]) -> dict[str, float]:
    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF(word_order=2)
    bleu = float(bleu_metric.corpus_score(predictions, [references]).score)
    chrf = float(chrf_metric.corpus_score(predictions, [references]).score)
    return {
        "bleu": bleu,
        "chrf": chrf,
    }


def build_sentence_memory_pairs(
    *,
    raw_dir: Path,
    train_frame: pd.DataFrame,
    train_id_column: str,
    source_columns: tuple[str, ...],
    token_normalizer: dict[str, str],
    max_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    if len(source_columns) != 1:
        return []

    train_sources_by_id = {
        _clean_text(row[train_id_column]): _clean_text(row[source_columns[0]])
        for _, row in train_frame.iterrows()
    }
    docs = _build_sentence_docs(
        raw_dir=raw_dir,
        token_normalizer=token_normalizer,
        train_sources_by_id=train_sources_by_id,
    )

    pairs: list[tuple[str, str]] = []
    for doc in docs.values():
        for source_norm, target in zip(
            doc.source_sentences_norm, doc.sentence_translations, strict=True
        ):
            source_clean = _clean_text(source_norm)
            target_clean = _clean_text(target)
            if not source_clean or not target_clean:
                continue
            pairs.append((source_clean, target_clean))

    unique_pairs = list(dict.fromkeys(pairs))
    if len(unique_pairs) <= max_pairs:
        return unique_pairs

    rng = random.Random(seed)
    sampled = unique_pairs[:]
    rng.shuffle(sampled)
    return sampled[:max_pairs]


def fit_retrieval_memory(
    sources: list[str], targets: list[str]
) -> tuple[TfidfVectorizer, Any, list[str]]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 7),
        min_df=1,
        max_features=500_000,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(sources)
    return vectorizer, matrix, targets


def retrieve_drafts(
    *,
    query_sources: list[str],
    vectorizer: TfidfVectorizer,
    memory_matrix: Any,
    memory_targets: list[str],
    blocked_memory_indices: list[int | None] | None = None,
    search_top_k: int = 16,
) -> tuple[list[str], list[float]]:
    if not query_sources:
        return [], []

    query_matrix = vectorizer.transform(query_sources)
    similarities = (query_matrix @ memory_matrix.T).toarray()

    top_k = min(max(2, search_top_k), similarities.shape[1])
    outputs: list[str] = []
    scores: list[float] = []
    for row_idx in range(similarities.shape[0]):
        row = similarities[row_idx]
        blocked = (
            None
            if blocked_memory_indices is None
            else blocked_memory_indices[row_idx]
        )
        candidate_indices = np.argpartition(row, -top_k)[-top_k:]
        ordered = sorted(candidate_indices.tolist(), key=lambda idx: row[idx], reverse=True)
        chosen = None
        for idx in ordered:
            if blocked is not None and int(idx) == int(blocked):
                continue
            chosen = int(idx)
            break
        if chosen is None:
            chosen = int(ordered[0]) if ordered else int(np.argmax(row))
        outputs.append(_clean_text(memory_targets[chosen]))
        scores.append(float(row[chosen]))
    return outputs, scores


def tokenize_pairs(
    tokenizer: Any,
    prompts: list[str],
    targets: list[str],
    max_source_length: int,
    max_target_length: int,
) -> TextPairDataset:
    model_inputs = tokenizer(
        prompts,
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )["input_ids"]
    return TextPairDataset(model_inputs, labels)


def build_collate_fn(tokenizer: Any) -> Any:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id for batching.")

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        inputs = [
            {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
            }
            for row in batch
        ]
        labels = [{"input_ids": row["labels"]} for row in batch]
        padded_inputs = tokenizer.pad(inputs, return_tensors="pt")
        padded_labels = tokenizer.pad(labels, return_tensors="pt")["input_ids"]
        padded_labels = padded_labels.clone()
        padded_labels[padded_labels == pad_id] = -100
        padded_inputs["labels"] = padded_labels
        return padded_inputs

    return collate


def load_transformer(model_name: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "transformers is required. Install with `pip install transformers sentencepiece safetensors`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def train_seq2seq(
    *,
    model: Any,
    tokenizer: Any,
    config: Seq2SeqConfig,
    device: torch.device,
    train_prompts: list[str],
    train_targets: list[str],
    val_prompts: list[str] | None,
    val_targets: list[str] | None,
    seed: int,
    progress: bool,
    log_interval: int,
    phase_name: str,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any] | None]:
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    train_dataset = tokenize_pairs(
        tokenizer=tokenizer,
        prompts=train_prompts,
        targets=train_targets,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )
    collate_fn = build_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / config.grad_accum_steps))
    total_steps = max(1, config.epochs * steps_per_epoch)
    warmup_steps = int(config.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0e12
    history: list[dict[str, Any]] = []
    best_val_metrics: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        epoch_iterator = (
            tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"{phase_name} train {epoch}/{config.epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            if progress and tqdm is not None
            else train_loader
        )

        for batch_index, batch in enumerate(epoch_iterator, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / float(config.grad_accum_steps)
            loss.backward()
            running_loss += float(loss.item()) * float(config.grad_accum_steps)

            should_step = (
                batch_index % config.grad_accum_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            if (
                progress
                and tqdm is not None
                and hasattr(epoch_iterator, "set_postfix")
                and (
                    batch_index % max(1, int(log_interval)) == 0
                    or batch_index == len(train_loader)
                )
            ):
                mean_loss = running_loss / float(batch_index)
                current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else config.learning_rate
                epoch_iterator.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    opt=f"{optimizer_steps}/{steps_per_epoch}",
                )

        avg_train_loss = running_loss / max(1, len(train_loader))
        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "optimizer_steps": optimizer_steps,
        }

        if val_prompts is not None and val_targets is not None and val_prompts:
            predictions = generate_predictions(
                model=model,
                tokenizer=tokenizer,
                prompts=val_prompts,
                device=device,
                batch_size=config.eval_batch_size,
                num_beams=config.generation_beams,
                max_new_tokens=config.generation_max_new_tokens,
                progress=progress,
                progress_desc=f"{phase_name} val gen {epoch}/{config.epochs}",
            )
            val_metrics = evaluate_text_predictions(val_targets, predictions)
            val_metrics.update(compute_bleu_chrf(references=val_targets, predictions=predictions))
            epoch_row["val"] = val_metrics
            score = (
                (0.45 * val_metrics["sequence_ratio"])
                + (0.35 * val_metrics["char_f1"])
                + (0.20 * (val_metrics["chrf"] / 100.0))
            )
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_val_metrics = val_metrics
        history.append(epoch_row)
        print(
            f"[epoch {epoch}/{config.epochs}] train_loss={avg_train_loss:.5f}"
            + (
                f", val_seq={epoch_row['val']['sequence_ratio']:.5f}, "
                f"val_bleu={epoch_row['val']['bleu']:.3f}, "
                f"val_chrf={epoch_row['val']['chrf']:.3f}"
                if "val" in epoch_row
                else ""
            )
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_metrics


def generate_predictions(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    progress: bool,
    progress_desc: str,
) -> list[str]:
    model.eval()
    outputs: list[str] = []
    starts = list(range(0, len(prompts), batch_size))
    iterator = (
        tqdm(
            starts,
            total=len(starts),
            desc=progress_desc,
            leave=False,
            dynamic_ncols=True,
        )
        if progress and tqdm is not None
        else starts
    )
    with torch.no_grad():
        for start in iterator:
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend([_clean_text(text) for text in decoded])
    return outputs


def build_memory(
    *,
    train_sources_norm: list[str],
    train_targets: list[str],
    train_indices: list[int],
    sentence_pairs: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    memory_sources = [_clean_text(train_sources_norm[idx]) for idx in train_indices]
    memory_targets = [_clean_text(train_targets[idx]) for idx in train_indices]
    for source, target in sentence_pairs:
        memory_sources.append(_clean_text(source))
        memory_targets.append(_clean_text(target))
    return memory_sources, memory_targets


def prepare_prompts(
    *, sources: list[str], drafts: list[str]
) -> list[str]:
    return [
        build_editor_prompt(source_text=source, retrieval_draft=draft)
        for source, draft in zip(sources, drafts, strict=True)
    ]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device_preference)
    set_seed(args.seed)

    files = discover_competition_files(args.raw_dir)
    dataset = build_datasets(files, target_column=args.target_column)
    token_normalizer = _load_lexicon_normalizer(args.raw_dir)

    train_sources_norm = [
        _normalize_transliteration(source, token_normalizer)
        for source in dataset.train_source_texts
    ]
    test_sources_norm = [
        _normalize_transliteration(source, token_normalizer)
        for source in dataset.test_source_texts
    ]
    train_targets = [_clean_text(value) for value in dataset.train_targets]

    sentence_pairs: list[tuple[str, str]] = []
    if not args.disable_sentence_memory:
        sentence_pairs = build_sentence_memory_pairs(
            raw_dir=args.raw_dir,
            train_frame=dataset.train_frame,
            train_id_column=dataset.train_id_column,
            source_columns=dataset.source_columns,
            token_normalizer=token_normalizer,
            max_pairs=max(0, int(args.max_sentence_memory_pairs)),
            seed=args.seed,
        )

    indices = np.arange(len(train_sources_norm), dtype=np.int64)
    holdout_size = max(1, int(round(args.holdout_fraction * len(indices))))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=holdout_size,
        random_state=args.seed,
        shuffle=True,
    )
    train_idx_list = [int(idx) for idx in train_idx.tolist()]
    val_idx_list = [int(idx) for idx in val_idx.tolist()]

    holdout_memory_sources, holdout_memory_targets = build_memory(
        train_sources_norm=train_sources_norm,
        train_targets=train_targets,
        train_indices=train_idx_list,
        sentence_pairs=sentence_pairs,
    )
    holdout_vectorizer, holdout_matrix, holdout_target_lookup = fit_retrieval_memory(
        holdout_memory_sources, holdout_memory_targets
    )

    train_query_sources = [_clean_text(train_sources_norm[idx]) for idx in train_idx_list]
    train_blocked = list(range(len(train_idx_list)))
    train_drafts, train_draft_scores = retrieve_drafts(
        query_sources=train_query_sources,
        vectorizer=holdout_vectorizer,
        memory_matrix=holdout_matrix,
        memory_targets=holdout_target_lookup,
        blocked_memory_indices=train_blocked,
    )

    val_query_sources = [_clean_text(train_sources_norm[idx]) for idx in val_idx_list]
    val_drafts, val_draft_scores = retrieve_drafts(
        query_sources=val_query_sources,
        vectorizer=holdout_vectorizer,
        memory_matrix=holdout_matrix,
        memory_targets=holdout_target_lookup,
        blocked_memory_indices=None,
    )

    train_prompt_sources = [_clean_text(dataset.train_source_texts[idx]) for idx in train_idx_list]
    val_prompt_sources = [_clean_text(dataset.train_source_texts[idx]) for idx in val_idx_list]
    train_prompts = prepare_prompts(sources=train_prompt_sources, drafts=train_drafts)
    val_prompts = prepare_prompts(sources=val_prompt_sources, drafts=val_drafts)
    train_target_texts = [train_targets[idx] for idx in train_idx_list]
    val_target_texts = [train_targets[idx] for idx in val_idx_list]

    draft_holdout_metrics = evaluate_text_predictions(val_target_texts, val_drafts)
    draft_holdout_metrics.update(
        compute_bleu_chrf(references=val_target_texts, predictions=val_drafts)
    )

    print(f"Competition: {COMPETITION_SLUG}")
    print(f"Device: {device}")
    print(f"Rows: train={len(dataset.train_frame)}, test={len(dataset.test_frame)}")
    print(f"Supplemental sentence-memory pairs: {len(sentence_pairs)}")
    print(
        "Holdout retrieval draft baseline: "
        f"seq={draft_holdout_metrics['sequence_ratio']:.5f}, "
        f"char_f1={draft_holdout_metrics['char_f1']:.5f}, "
        f"bleu={draft_holdout_metrics['bleu']:.3f}, "
        f"chrf={draft_holdout_metrics['chrf']:.3f}"
    )

    tokenizer, holdout_model = load_transformer(args.model_name)
    holdout_config = Seq2SeqConfig(
        model_name=args.model_name,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        train_batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        epochs=max(1, int(args.epochs)),
        warmup_ratio=float(args.warmup_ratio),
        max_source_length=int(args.max_source_length),
        max_target_length=int(args.max_target_length),
        generation_beams=max(1, int(args.generation_beams)),
        generation_max_new_tokens=max(16, int(args.generation_max_new_tokens)),
        max_grad_norm=float(args.max_grad_norm),
    )
    holdout_model, holdout_history, holdout_val_metrics = train_seq2seq(
        model=holdout_model,
        tokenizer=tokenizer,
        config=holdout_config,
        device=device,
        train_prompts=train_prompts,
        train_targets=train_target_texts,
        val_prompts=val_prompts,
        val_targets=val_target_texts,
        seed=args.seed,
        progress=bool(args.progress),
        log_interval=max(1, int(args.log_interval)),
        phase_name="holdout",
    )
    holdout_predictions = generate_predictions(
        model=holdout_model,
        tokenizer=tokenizer,
        prompts=val_prompts,
        device=device,
        batch_size=holdout_config.eval_batch_size,
        num_beams=holdout_config.generation_beams,
        max_new_tokens=holdout_config.generation_max_new_tokens,
        progress=bool(args.progress),
        progress_desc="holdout final eval",
    )
    transformer_holdout_metrics = evaluate_text_predictions(
        val_target_texts, holdout_predictions
    )
    transformer_holdout_metrics.update(
        compute_bleu_chrf(references=val_target_texts, predictions=holdout_predictions)
    )
    print(
        "Holdout transformer metrics: "
        f"seq={transformer_holdout_metrics['sequence_ratio']:.5f}, "
        f"char_f1={transformer_holdout_metrics['char_f1']:.5f}, "
        f"bleu={transformer_holdout_metrics['bleu']:.3f}, "
        f"chrf={transformer_holdout_metrics['chrf']:.3f}"
    )

    full_memory_sources, full_memory_targets = build_memory(
        train_sources_norm=train_sources_norm,
        train_targets=train_targets,
        train_indices=[int(idx) for idx in indices.tolist()],
        sentence_pairs=sentence_pairs,
    )
    full_vectorizer, full_matrix, full_target_lookup = fit_retrieval_memory(
        full_memory_sources, full_memory_targets
    )
    full_train_drafts, full_train_draft_scores = retrieve_drafts(
        query_sources=[_clean_text(source) for source in train_sources_norm],
        vectorizer=full_vectorizer,
        memory_matrix=full_matrix,
        memory_targets=full_target_lookup,
        blocked_memory_indices=list(range(len(train_sources_norm))),
    )
    test_drafts, test_draft_scores = retrieve_drafts(
        query_sources=[_clean_text(source) for source in test_sources_norm],
        vectorizer=full_vectorizer,
        memory_matrix=full_matrix,
        memory_targets=full_target_lookup,
        blocked_memory_indices=None,
    )

    final_model = holdout_model
    final_history: list[dict[str, Any]] = []
    final_config = Seq2SeqConfig(
        model_name=args.model_name,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        train_batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        epochs=max(1, int(args.final_epochs)),
        warmup_ratio=float(args.warmup_ratio),
        max_source_length=int(args.max_source_length),
        max_target_length=int(args.max_target_length),
        generation_beams=max(1, int(args.generation_beams)),
        generation_max_new_tokens=max(16, int(args.generation_max_new_tokens)),
        max_grad_norm=float(args.max_grad_norm),
    )
    if args.train_full:
        tokenizer, final_model = load_transformer(args.model_name)
        final_prompts = prepare_prompts(
            sources=[_clean_text(source) for source in dataset.train_source_texts],
            drafts=full_train_drafts,
        )
        final_model, final_history, _ = train_seq2seq(
            model=final_model,
            tokenizer=tokenizer,
            config=final_config,
            device=device,
            train_prompts=final_prompts,
            train_targets=train_targets,
            val_prompts=None,
            val_targets=None,
            seed=args.seed + 17,
            progress=bool(args.progress),
            log_interval=max(1, int(args.log_interval)),
            phase_name="final",
        )

    test_prompts = prepare_prompts(
        sources=[_clean_text(source) for source in dataset.test_source_texts],
        drafts=test_drafts,
    )
    transformer_test_predictions = generate_predictions(
        model=final_model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        device=device,
        batch_size=final_config.eval_batch_size,
        num_beams=final_config.generation_beams,
        max_new_tokens=final_config.generation_max_new_tokens,
        progress=bool(args.progress),
        progress_desc="test generation",
    )

    use_transformer = (
        transformer_holdout_metrics["sequence_ratio"]
        >= draft_holdout_metrics["sequence_ratio"] + 1.0e-4
    )
    strategy = "transformer_editor" if use_transformer else "retrieval_draft"
    final_predictions = (
        transformer_test_predictions if use_transformer else test_drafts
    )

    empty_replacements = 0
    for idx in range(len(final_predictions)):
        candidate = _clean_text(final_predictions[idx])
        fallback = _clean_text(test_drafts[idx])
        if not candidate:
            final_predictions[idx] = fallback
            empty_replacements += 1
            continue
        if len(candidate.split()) < max(2, int(0.30 * max(1, len(fallback.split())))):
            final_predictions[idx] = fallback
            empty_replacements += 1

    submission = pd.DataFrame(
        {
            dataset.id_column: dataset.test_frame[dataset.id_column].tolist(),
            dataset.target_column: final_predictions,
        }
    )
    submission = dataset.sample_submission[[dataset.id_column]].merge(
        submission,
        on=dataset.id_column,
        how="left",
    )
    if submission[dataset.target_column].isna().any():
        raise ValueError("Submission contains missing rows after id merge.")

    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    submission.to_csv(args.submission_path, index=False)
    final_model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    report = {
        "selected_strategy": strategy,
        "use_transformer_editor": bool(use_transformer),
        "empty_or_short_replacements": int(empty_replacements),
        "test_rows": int(len(dataset.test_frame)),
        "test_draft_scores": test_draft_scores,
        "holdout_train_indices": train_idx_list,
        "holdout_val_indices": val_idx_list,
        "model_dir": str(args.model_dir),
    }
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics_payload = {
        "competition": COMPETITION_SLUG,
        "device_used": str(device),
        "model_name": args.model_name,
        "data": {
            "train_rows": int(len(dataset.train_frame)),
            "test_rows": int(len(dataset.test_frame)),
            "source_columns": list(dataset.source_columns),
            "target_column": dataset.target_column,
            "id_column": dataset.id_column,
            "supplemental_sentence_memory_pairs": int(len(sentence_pairs)),
        },
        "retrieval_draft_holdout_metrics": draft_holdout_metrics,
        "transformer_holdout_metrics": transformer_holdout_metrics,
        "transformer_holdout_best_metrics": holdout_val_metrics,
        "holdout_history": holdout_history,
        "final_history": final_history,
        "selected_strategy": strategy,
        "config": {
            "holdout": holdout_config.__dict__,
            "final": final_config.__dict__,
        },
        "draft_score_stats": {
            "train": {
                "mean": float(np.mean(train_draft_scores)) if train_draft_scores else 0.0,
                "min": float(np.min(train_draft_scores)) if train_draft_scores else 0.0,
                "max": float(np.max(train_draft_scores)) if train_draft_scores else 0.0,
            },
            "val": {
                "mean": float(np.mean(val_draft_scores)) if val_draft_scores else 0.0,
                "min": float(np.min(val_draft_scores)) if val_draft_scores else 0.0,
                "max": float(np.max(val_draft_scores)) if val_draft_scores else 0.0,
            },
            "test": {
                "mean": float(np.mean(test_draft_scores)) if test_draft_scores else 0.0,
                "min": float(np.min(test_draft_scores)) if test_draft_scores else 0.0,
                "max": float(np.max(test_draft_scores)) if test_draft_scores else 0.0,
            },
        },
    }
    args.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved submission: {args.submission_path}")
    print(f"Saved report: {args.report_path}")
    print(f"Saved metrics: {args.metrics_path}")
    print(f"Saved model/tokenizer: {args.model_dir}")
    print(f"Selected final strategy: {strategy}")
    print("Submission preview:")
    print(submission.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
