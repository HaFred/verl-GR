"""MiniOneRec dataset adapter following the original SidDataset behavior."""

from __future__ import annotations

import copy
import functools
import logging
import os
from typing import Any, Optional

import datasets
import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl_gr.recipes.minionerec.minionerec_format import build_sid_prompt, parse_maybe_list

logger = logging.getLogger(__name__)

MINIONEREC_SOURCE = "minionerec"

def extract_minionerec_prompt_fields(row: dict[str, Any], *, prompt_key: str) -> dict[str, Any]:
    """Build prompt/reward fields compatible with verl reward routing."""

    history_item_sid = parse_maybe_list(row.get("history_item_sid"))
    if not history_item_sid:
        raise ValueError("MiniOneRec sample has empty history_item_sid.")

    target_sid = str(row.get("item_sid", "")).strip()
    if not target_sid:
        raise ValueError("MiniOneRec sample has empty item_sid.")

    prompt, history_key = build_sid_prompt(history_item_sid)
    target = f"{target_sid}\n"
    history_item_ids = parse_maybe_list(row.get("history_item_id"))
    last_history_item_id = history_item_ids[-1] if history_item_ids else None
    dedup = str(row.get("item_id")) == str(last_history_item_id) if last_history_item_id is not None else False

    row[prompt_key] = prompt
    row["reward_model"] = {"ground_truth": target, "style": "rule"}
    row["source"] = row.get("source", MINIONEREC_SOURCE)
    row["data_source"] = row.get("data_source", row["source"])
    row["extra_info"] = {
        **(row.get("extra_info") or {}),
        "history_key": history_key,
        "target_sid": target_sid,
        "dedup": dedup,
    }
    return row


class MiniOneRecDataset(Dataset):
    """Dataset adapter for MiniOneRec CSV/parquet files.

    The original MiniOneRec trainer tokenizes plain prompt strings directly
    rather than applying chat templates. This adapter preserves that behavior
    and also exposes `raw_prompt_text` for the custom async agent loop.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        if processor is not None:
            logger.warning("MiniOneRecDataset ignores processor/chat-template multimodal handling.")
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        if self.num_workers is not None:
            self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet: bool = False) -> None:
        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, data_file in enumerate(target_files):
            local_path = copy_to_local(src=data_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path
        if use_origin_parquet:
            self.data_files = target_files

    def _load_file(self, data_file: str) -> datasets.Dataset:
        suffix = os.path.splitext(str(data_file))[1].lower()
        if suffix == ".parquet":
            return datasets.load_dataset("parquet", data_files=data_file)["train"]
        if suffix == ".csv":
            return datasets.load_dataset("csv", data_files=data_file)["train"]
        raise ValueError(f"Unsupported MiniOneRec data file type: {data_file}")

    def _read_files_and_tokenize(self) -> None:
        dataframes = [self._load_file(data_file) for data_file in self.data_files]
        self.dataframe = datasets.concatenate_datasets(dataframes)
        logger.info("MiniOneRec dataset len: %s", len(self.dataframe))

        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(len(self.dataframe), size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())

        extract_fn = functools.partial(extract_minionerec_prompt_fields, prompt_key=self.prompt_key)
        self.dataframe = self.dataframe.map(
            extract_fn,
            num_proc=self.num_workers,
            desc="Extract MiniOneRec prompts and reward annotations",
        )
        if self.filter_overlong_prompts:
            self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        logger.info("MiniOneRec processed dataset len: %s", len(self.dataframe))

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        def doc_length(doc: dict[str, Any]) -> int:
            return len(tokenizer.encode(doc[prompt_key], add_special_tokens=False))

        filtered = dataframe.filter(
            lambda doc: doc_length(doc) <= self.max_prompt_length - 10,
            num_proc=self.num_workers,
            desc=f"Filtering MiniOneRec prompts longer than {self.max_prompt_length - 10} tokens",
        )
        logger.info("MiniOneRec filtered dataset len: %s", len(filtered))
        return filtered

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("resume with serialized dataloader, consider restarting from scratch for better perf")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.dataframe[index])
        raw_prompt = str(row[self.prompt_key])
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)
        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]
        row["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)[-self.max_prompt_length :]
        row["raw_prompt_text"] = raw_prompt
        row["index"] = (row.get("extra_info") or {}).get("index", index)
        row["tools_kwargs"] = {}
        row["interaction_kwargs"] = {}
        if "uid" not in row:
            row["uid"] = str(row["index"])
        return row

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()
