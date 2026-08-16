"""Derive task referents and manipulation roles from an instruction."""

from __future__ import annotations

import difflib
import json
import os
import re

_PROMPT = """A robot is given this instruction:

    "{instruction}"

List ONLY the physical objects the instruction refers to. Reply with JSON:

{{"referents": [{{"name": "...", "detector": "...", "role": "grasped|destination|other"}}]}}

- "name": one short lowercase word identifying the object.
- "detector": the object's plain noun for an open-vocabulary detector (a coffee maker, a teapot). Add
  a visual attribute ONLY if the instruction itself states one (put the RED cup down gives red cup).
  Never invent appearance the instruction does not give: a guessed colour mis-detects the real object.
- "role": "grasped" if the robot must pick it up or move it; "destination" if something is placed,
  poured or put into/onto it; otherwise "other".
- Do NOT invent objects the instruction does not mention. Do NOT list distractors, furniture, or the
  robot itself. A part named separately from its whole (a lid of a pot) is its own referent.
"""


def _chat(prompt: str) -> str:
    """Request a JSON response from GPT-4o."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    reply = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.choices[0].message.content


def align(name: str, world_names) -> str:
    """Match a language referent to a scene-entity name."""
    if not world_names:
        return name

    lowered = {str(world_name).lower(): world_name for world_name in world_names}
    if name in lowered:
        return lowered[name]

    contained = [
        world_name
        for lowered_name, world_name in lowered.items()
        if name in lowered_name or lowered_name in name
    ]
    if len(contained) == 1:
        return contained[0]

    close = difflib.get_close_matches(
        name,
        list(lowered),
        n=1,
        cutoff=0.85,
    )
    return lowered[close[0]] if close else name


def derive(instruction: str, world_names=(), *, chat=None) -> dict:
    """Derive detector names, grasp objects, and the destination."""
    raw = (chat or _chat)(_PROMPT.format(instruction=instruction))

    try:
        payload = json.loads(_strip_fence(raw))
        referents = payload["referents"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"[vocabulary] could not parse the referent reply: {exc}\n{raw}"
        )

    if not referents:
        raise SystemExit(
            f"[vocabulary] no referents found in instruction {instruction!r}"
        )

    objects = {}
    grasp_objs = []
    place_obj = None

    for item in referents:
        name = align(
            str(item.get("name", "")).strip().lower(),
            world_names,
        )
        if not name:
            continue

        objects[name] = str(item.get("detector") or name).strip()
        role = str(item.get("role", "other")).strip().lower()

        if role == "grasped" and name not in grasp_objs:
            grasp_objs.append(name)
        elif role == "destination" and place_obj is None:
            place_obj = name

    if not objects:
        raise SystemExit(
            f"[vocabulary] every referent was empty for instruction {instruction!r}"
        )

    return {
        "objects": objects,
        "grasp_objs": grasp_objs,
        "place_obj": place_obj,
    }


def _strip_fence(text: str) -> str:
    """Remove an optional JSON code fence."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text or "", re.S)
    return (match.group(1) if match else text or "").strip()