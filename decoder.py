"""Constrained decoding: the model only chooses among valid tokens."""
import json
import os
import re
import string
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.models import FunctionDef
from src.vocab import TokenTable

MAX_NUMBER_LEN = 20
MAX_STRING_TOKENS = 120
ESCAPE_START = '"\\/bfnrtu' + string.hexdigits

NUMBER_PREFIX = re.compile(r"^-?((0|[1-9][0-9]*)(\.[0-9]*)?)?$")
NUMBER_DONE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$")
INTEGER_PREFIX = re.compile(r"^-?(0|[1-9][0-9]*)?$")
INTEGER_DONE = re.compile(r"^-?(0|[1-9][0-9]*)$")

EXAMPLE = (
    "Examples\n"
    'Functions: [{"name":"fn_multiply","description":"Multiply two '
    'numbers.","parameters":{"x":"number","y":"number"}},'
    '{"name":"fn_welcome","description":"Welcome a person by name.",'
    '"parameters":{"person":"string"}}]\n'
    "Request: Multiply 6 by 7\n"
    'Function call: {"name":"fn_multiply",'
    '"parameters":{"x":6,"y":7}}\n'
    "Request: Welcome Maria\n"
    'Function call: {"name":"fn_welcome",'
    '"parameters":{"person":"Maria"}}\n\n'
)


def scan_string(body: str) -> tuple[bool, bool]:
    """Check the content of a JSON string (without the quotes).

    Args:
        body: Text written so far inside the string.

    Returns:
        (is_valid_prefix, ends_inside_an_escape_sequence)
    """
    i = 0
    while i < len(body):
        char = body[i]
        if char == '"' or ord(char) < 0x20:
            return False, False
        if char == "\\":
            if i + 1 >= len(body):
                return True, True
            nxt = body[i + 1]
            if nxt in '"\\/bfnrt':
                i += 2
                continue
            if nxt == "u":
                digits = body[i + 2:i + 6]
                if any(d not in string.hexdigits for d in digits):
                    return False, False
                if len(digits) < 4:
                    return True, True
                i += 6
                continue
            return False, False
        i += 1
    return True, False


def build_prompt(request: str, functions: list[FunctionDef]) -> str:
    """Build the context text shown to the model (format is not trusted)."""
    schema = json.dumps(
        [{"name": f.name,
          "description": f.description,
          "parameters": {k: v.type for k, v in f.parameters.items()}}
         for f in functions],
        separators=(",", ":"))
    return (
        "Convert the user request into one function call written as JSON. "
        "Choose the function whose description matches the request and "
        "copy numbers and text exactly from the request.\n\n"
        f"{EXAMPLE}"
        f"Functions: {schema}\n"
        f"Request: {request}\n"
        "Function call: "
    )


class FunctionCaller(BaseModel):
    """Generates one function call per request with constrained decoding."""

    model: Any
    table: TokenTable
    functions: list[FunctionDef]
    prompt: str = ""
    text: str = ""
    literal_cache: dict[tuple[str, ...], list[int]] = {}

    def _emit(self, piece: str) -> None:
        """Append text written by our code or chosen by the model."""
        self.text += piece

    def _pick(self, base: Any, extra: list[int]) -> int:
        """Ask the model for logits, forbid invalid tokens, take the best."""
        ids = self.model.encode(self.prompt + self.text).tolist()[0]
        logits = np.asarray(
            self.model.get_logits_from_input_ids(ids), dtype=np.float64)
        allowed = np.zeros(len(logits), dtype=bool)
        if base is not None:
            allowed |= base
        if extra:
            allowed[extra] = True
        if not allowed.any():
            raise RuntimeError("no valid token available")
        return int(np.argmax(np.where(allowed, logits, -np.inf)))

    def _starting_with(self, char: str) -> list[int]:
        """Ids of tokens whose text starts with `char` (cached)."""
        key = ("starts", char)
        if key not in self.literal_cache:
            self.literal_cache[key] = [
                i for i, s in enumerate(self.table.strings)
                if s.startswith(char)]
        return self.literal_cache[key]

    def _literal_ids(self, cands: list[str]) -> list[int]:
        """Ids of tokens that are a piece of at least one candidate."""
        key = ("lit",) + tuple(cands)
        if key not in self.literal_cache:
            self.literal_cache[key] = [
                i for i, s in enumerate(self.table.strings)
                if s and any(s in c for c in cands)]
        return self.literal_cache[key]

    def _choose_literal(self, cands: list[str]) -> str:
        """Let the model pick one candidate, token by token."""
        strings = self.table.strings
        done = ""
        while done not in cands:
            matches = [c for c in cands if c.startswith(done)]
            common = os.path.commonprefix(matches)
            if len(common) > len(done):
                self._emit(common[len(done):])
                done = common
                continue
            allowed = [
                i for i in self._literal_ids(cands)
                if any(c.startswith(done + strings[i]) for c in matches)]
            piece = strings[self._pick(None, allowed)]
            self._emit(piece)
            done += piece
        return done

    def _choose_number(self, integer: bool, more: bool) -> None:
        """Write a valid JSON number; the model decides when to stop."""
        prefix_re = INTEGER_PREFIX if integer else NUMBER_PREFIX
        done_re = INTEGER_DONE if integer else NUMBER_DONE
        stop_ids = self._starting_with("," if more else "}")
        stop_set = set(stop_ids)
        strings = self.table.strings
        value = ""
        for _ in range(MAX_NUMBER_LEN * 2):
            complete = done_re.match(value) is not None
            allowed: list[int] = list(stop_ids) if complete else []
            if not (complete and len(value) >= MAX_NUMBER_LEN):
                allowed.extend(
                    i for i in self.table.number_ids
                    if prefix_re.match(value + strings[i]))
            token = self._pick(None, allowed)
            if complete and token in stop_set:
                break
            value += strings[token]
            self._emit(strings[token])
        if not done_re.match(value):
            self._emit("0")

    def _choose_string(self) -> None:
        """Write a valid JSON string; the model decides when to close."""
        table = self.table
        close_ids = self._starting_with('"')
        close_set = set(close_ids)
        self._emit('"')
        body = ""
        steps = 0
        while True:
            steps += 1
            if steps > MAX_STRING_TOKENS * 2:
                raise RuntimeError("string value too long")
            pending = scan_string(body)[1]
            if pending:
                base = None
                extra = [i for i in table.nocontrol_ids
                         if table.strings[i][0] in ESCAPE_START
                         and scan_string(body + table.strings[i])[0]]
            else:
                base = table.plain
                extra = [i for i in table.escape_ids
                         if scan_string(body + table.strings[i])[0]]
                extra.extend(close_ids)
                if steps > MAX_STRING_TOKENS:
                    base, extra = None, list(close_ids)
            token = self._pick(base, extra)
            if not pending and token in close_set:
                break
            body += table.strings[token]
            self._emit(table.strings[token])
        self._emit('"')

    def _choose_value(self, kind: str, more: bool) -> None:
        """Write one parameter value according to its declared type."""
        if kind == "string":
            self._choose_string()
        elif kind in ("number", "integer"):
            self._choose_number(kind == "integer", more)
        elif kind == "boolean":
            self._choose_literal(["true", "false"])
        else:
            self._choose_literal(["null"])

    def call(self, request: str) -> dict[str, Any]:
        """Turn a request into {"name": ..., "parameters": {...}}.

        Args:
            request: The natural-language prompt.

        Returns:
            The parsed function call.

        Raises:
            RuntimeError: If no valid continuation exists.
        """
        self.text = ""
        self.prompt = build_prompt(request, self.functions)
        self._emit('{"name":"')
        done = self._choose_literal([f.name + '"' for f in self.functions])
        func = next(f for f in self.functions if f.name == done[:-1])
        self.prompt = build_prompt(request, [func])
        self._emit(',"parameters":{')
        last = len(func.parameters) - 1
        for index, (key, spec) in enumerate(func.parameters.items()):
            self._emit(json.dumps(key) + ":")
            more = index != last
            self._choose_value(spec.type, more)
            if more:
                self._emit(",")
        self._emit("}}")
        result: dict[str, Any] = json.loads(self.text)
        return result