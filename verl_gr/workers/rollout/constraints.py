"""Prefix constraints used by Python-side beam search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_TRIE_CACHE: dict[tuple[str, str | None, int | None, bool, int], "PrefixTrieConstraint"] = {}


def get_hash(token_ids: list[int]) -> str:
    return "-".join(str(token_id) for token_id in token_ids)


@dataclass(slots=True)
class PrefixTrieConstraint:
    """MiniOneRec-compatible prefix trie for constrained SID decoding."""

    hash_dict: dict[str, list[int]]
    prefix_index: int
    eos_token_id: int | None = None
    fallback_to_eos: bool = True

    @classmethod
    def from_info_file(
        cls,
        *,
        info_file: str,
        tokenizer,
        base_model: str | None = None,
        eos_token_id: int | None = None,
        fallback_to_eos: bool = True,
    ) -> "PrefixTrieConstraint":
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        prefix_index = 4 if base_model and "gpt2" in base_model.lower() else 3
        hash_dict: dict[str, set[int]] = {}

        with open(info_file, "r", encoding="utf-8") as f:
            semantic_ids = [line.split("\t")[0].strip() + "\n" for line in f if line.strip()]
        formatted = [f"### Response:\n{semantic_id}" for semantic_id in semantic_ids]
        for text in formatted:
            token_ids = tokenizer(text).input_ids
            if base_model and "llama" in base_model.lower():
                token_ids = token_ids[1:]
            token_ids = list(token_ids)
            if eos_token_id is not None:
                token_ids.append(eos_token_id)
            for i in range(prefix_index, len(token_ids)):
                key_ids = token_ids[:i] if i == prefix_index else token_ids[prefix_index:i]
                hash_dict.setdefault(get_hash(key_ids), set()).add(int(token_ids[i]))

        return cls(
            hash_dict={key: sorted(values) for key, values in hash_dict.items()},
            prefix_index=prefix_index,
            eos_token_id=eos_token_id,
            fallback_to_eos=fallback_to_eos,
        )

    def allowed_tokens(self, prompt_token_ids: list[int], generated_token_ids: list[int]) -> list[int]:
        if generated_token_ids:
            key_ids = generated_token_ids
        else:
            key_ids = prompt_token_ids[-self.prefix_index :]
        allowed = self.hash_dict.get(get_hash(list(key_ids)), [])
        if not allowed and self.fallback_to_eos and self.eos_token_id is not None:
            return [int(self.eos_token_id)]
        return list(allowed)


def build_constraint_from_config(config: dict[str, Any] | None, *, tokenizer) -> Callable[[list[int], list[int]], list[int]] | None:
    """Build an allowed-token callback from rollout config."""

    if not config:
        return None
    constraint_type = str(config.get("type", "")).lower()
    if constraint_type not in {"prefix_trie", "minionerec_prefix_trie"}:
        return None
    info_file = config.get("info_file")
    if not info_file:
        raise ValueError("prefix_trie constraint requires constraint.info_file")
    base_model = config.get("base_model")
    eos_token_id = config.get("eos_token_id", tokenizer.eos_token_id)
    fallback_to_eos = bool(config.get("fallback_to_eos", True))
    cache_key = (str(info_file), str(base_model) if base_model is not None else None, eos_token_id, fallback_to_eos, id(tokenizer))
    trie = _TRIE_CACHE.get(cache_key)
    if trie is None:
        trie = PrefixTrieConstraint.from_info_file(
            info_file=str(info_file),
            tokenizer=tokenizer,
            base_model=base_model,
            eos_token_id=eos_token_id,
            fallback_to_eos=fallback_to_eos,
        )
        _TRIE_CACHE[cache_key] = trie
    return trie.allowed_tokens
