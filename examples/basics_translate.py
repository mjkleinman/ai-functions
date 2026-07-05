"""Translate one sentence into multiple languages concurrently with async AI functions."""

import asyncio

from _utils import display

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult

model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


# Post-condition guarding a common failure: the model transliterating
# non-Latin scripts instead of using the native script.
@ai_function(model=model)
def check_translation(text: str) -> PostConditionResult:
    """
    Check that the following text is written in the native script of the language and does not contain any romanization.
    ```
    {text}
    ```
    Answer immediately without explaining your thinking.
    """


# An async AI function is awaited like any other coroutine.
@ai_function(model=model, post_conditions=[check_translation])
async def translate_text(text: str, lang: str) -> str:
    """
    Translate the text below to the following language: `{lang}`.
    ```
    {text}
    ```
    """


async def main():
    text = "It was the best of times, it was the worst of times"
    languages = ["fr", "ja", "it", "zh"]
    # gather() fans out one translation per language and awaits them together.
    translations = await asyncio.gather(*(translate_text(text, lang) for lang in languages))

    body = "\n".join(f"({lang}) {translation}" for lang, translation in zip(languages, translations, strict=True))
    display(text, body)


if __name__ == "__main__":
    asyncio.run(main())
