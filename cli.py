"""Əmr sətri interfeysi: interaktiv REPL loop."""

from __future__ import annotations

import argparse
import sys

from .config import CONFIG
from .exceptions import LLMConnectionError, LLMResponseError, WebAgentError
from .history import ConversationHistory
from .logging_setup import log
from .pipeline import answer, format_sources

HELP_TEXT = (
    "Əmrlər:\n"
    "  exit / quit / çıxış   — proqramdan çıx\n"
    "  clear                 — söhbət tarixçəsini təmizlə\n"
    "  help                  — bu mesajı göstər\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Veb axtarışı ilə gücləndirilmiş LLM agenti")
    parser.add_argument("--no-stream", action="store_true", help="Streaming cavabı deaktiv et")
    parser.add_argument("--no-fetch", action="store_true", help="Tam səhifə çəkməni deaktiv et (sürətli, daha az dəqiq)")
    parser.add_argument("--top-k", type=int, default=None, help="Rerank sonrası neçə mənbə saxlansın")
    parser.add_argument("--model", type=str, default=None, help="LLM model adı (server tərəfindən dəstəklənməlidir)")
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    """CLI arqumentləri qlobal CONFIG-i override edir (dataclass frozen olduğu üçün object.__setattr__)."""
    if args.no_fetch:
        object.__setattr__(CONFIG, "fetch_enabled", False)
    if args.top_k is not None:
        object.__setattr__(CONFIG, "rerank_top_k", args.top_k)
    if args.model is not None:
        object.__setattr__(CONFIG, "model_name", args.model)


def run_repl(stream: bool) -> None:
    history = ConversationHistory()
    print("🔎 Web axtarış agenti hazırdır. Əmrlər üçün 'help' yazın.\n")

    while True:
        try:
            user_input = input("Sual > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıxılır...")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in ("exit", "quit", "çıxış"):
            break
        if lowered == "help":
            print(HELP_TEXT)
            continue
        if lowered == "clear":
            history.clear()
            print("Tarixçə təmizləndi.\n")
            continue

        try:
            if stream:
                print()
                gen, sources = answer(user_input, history, stream=True)
                full_text_parts = []
                for token in gen:
                    print(token, end="", flush=True)
                    full_text_parts.append(token)
                full_text = "".join(full_text_parts)
                print()  # sətir sonu

                src_block = format_sources(sources)
                if src_block:
                    print(f"\n{src_block}")

                history.add_user(user_input)
                history.add_assistant(full_text)
            else:
                content, sources = answer(user_input, history, stream=False)
                print(f"\n{content}\n")
                src_block = format_sources(sources)
                if src_block:
                    print(src_block)
                history.add_user(user_input)
                history.add_assistant(content)

        except LLMConnectionError as e:
            print(f"\n[Xəta] {e}\n")
        except LLMResponseError as e:
            print(f"\n[Xəta] {e}\n")
        except WebAgentError as e:
            print(f"\n[Xəta] {e}\n")
        except Exception as e:
            log.exception("Gözlənilməz xəta")
            print(f"\n[Gözlənilməz xəta] {e}\n")

        print()


def main() -> None:
    args = parse_args()
    apply_overrides(args)
    stream = CONFIG.stream and not args.no_stream
    run_repl(stream=stream)


if __name__ == "__main__":
    main()
