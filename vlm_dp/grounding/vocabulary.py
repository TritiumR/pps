"""Derive the objects a task refers to, and what each is for, from the instruction.

The detector vocabulary and manipulation roles were hand-authored per task in task_prompts.json: a
{name: detector text} dict plus grasp_obj, place_obj and grasp_objs. That is the single largest piece
of per-task configuration in the stack, and it is information the instruction already carries. The
sentence put pear and apple on the scale names the referents, says which are picked up, and says where
they go.

Deriving it also removes a class of mis-grounding. The hand-authored dict listed distractors (mango,
cabbage, board) as named objects, which invites the grounding to select one as a target (measured: 2 of
10 weight seeds grasped a cabbage). Referents need identity, obstacles need only geometry, so naming
only what the instruction refers to fixes this by construction.

The VLM does the extraction because it is already in the loop for constraint generation, and picking
the manipulable nouns and their roles out of a sentence is what a language model is for. A hand-written
parser would be a per-phrasing rule set, the thing being removed.
"""
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
    """One text-only GPT-4o call returning JSON (same client/model as the constraint generator)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    reply = client.chat.completions.create(
        model="gpt-4o", temperature=0.0, max_tokens=512,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    return reply.choices[0].message.content


def align(name: str, world_names) -> str:
    """Map a language referent onto a scene-entity name when one plainly corresponds.

    Any embodiment must link a referent in a sentence to a thing in the world. In sim the scene entity
    carries that identity, so the link is by name. Substring first (lid does not match cover, and should
    not), then a conservative fuzzy pass for plurals and spelling. An unmatched referent keeps its own
    name, which is correct on a real robot where no scene-entity list exists.
    """
    if not world_names:
        return name
    lowered = {str(w).lower(): w for w in world_names}
    if name in lowered:
        return lowered[name]
    contained = [w for low, w in lowered.items() if name in low or low in name]
    if len(contained) == 1:
        return contained[0]
    close = difflib.get_close_matches(name, list(lowered), n=1, cutoff=0.85)
    return lowered[close[0]] if close else name


def derive(instruction: str, world_names=(), *, chat=None) -> dict:
    """Instruction -> ``{"objects": {name: detector}, "grasp_objs": [...], "place_obj": name|None}``.

    Raises on an unusable reply rather than returning a partial vocabulary: a silently empty object
    set would ground onto nothing and fail much later, with no trace back to here.
    """
    raw = (chat or _chat)(_PROMPT.format(instruction=instruction))
    try:
        payload = json.loads(_strip_fence(raw))
        referents = payload["referents"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"[vocabulary] could not parse the referent reply: {exc}\n{raw}")
    if not referents:
        raise SystemExit(f"[vocabulary] no referents found in instruction {instruction!r}")

    objects, grasp_objs, place_obj = {}, [], None
    for item in referents:
        name = align(str(item.get("name", "")).strip().lower(), world_names)
        if not name:
            continue
        objects[name] = str(item.get("detector") or name).strip()
        role = str(item.get("role", "other")).strip().lower()
        if role == "grasped" and name not in grasp_objs:
            grasp_objs.append(name)
        elif role == "destination" and place_obj is None:
            place_obj = name
    if not objects:
        raise SystemExit(f"[vocabulary] every referent was empty for instruction {instruction!r}")
    return {"objects": objects, "grasp_objs": grasp_objs, "place_obj": place_obj}


def _strip_fence(text: str) -> str:
    """Tolerate a ```json fenced reply (the model emits one despite response_format)."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text or "", re.S)
    return (match.group(1) if match else text or "").strip()
