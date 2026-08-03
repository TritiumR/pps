"""Generate ReKep constraints with GPT-4o.

Turns a keypoint-annotated image + instruction into per-stage constraint files
(``stage{i}_{subgoal|path}_constraints.txt``) and a ``metadata.json`` (num_stages,
grasp/release keypoints), written under a caller-given ``task_dir``. Ported from upstream ReKep.
"""

import base64
import json
import os
import time

import cv2
import numpy as np
import parse
from openai import OpenAI

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # Prompt file is selectable (env var) so alternates stay tracked as separate files, and the
        # default is unchanged. The chosen prompt is also saved per-run as prompt.txt.
        prompt_name = os.environ.get("REKEP_PROMPT_TEMPLATE", "prompt_template.txt")
        with open(os.path.join(_PROMPT_DIR, prompt_name), "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        self.task_dir = None

    def _build_prompt(self, image_path, instruction, geometry=""):
        img_base64 = encode_image(image_path)
        # geometry fills the grounded template's {geometry} slot; str.format ignores it elsewhere.
        prompt_text = self.prompt_template.format(instruction=instruction, geometry=geometry)
        # save prompt
        with open(os.path.join(self.task_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                ],
            }
        ]
        return messages

    def _parse_and_save_constraints(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                functions[name] = lines[start : end + 1]
        # group functions by stage prefix (the trailing part is the constraint index)
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w", encoding="utf-8") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")

    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        num_stages = None
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict["num_stages"] = int(num_stages["num_stages"])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        grasp_keypoints = None
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        grasp_keypoints = grasp_keypoints["grasp_keypoints"].replace("[", "").replace("]", "").split(",")
        data_dict["grasp_keypoints"] = [int(x.strip()) for x in grasp_keypoints]
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        release_keypoints = None
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        release_keypoints = release_keypoints["release_keypoints"].replace("[", "").replace("]", "").split(",")
        data_dict["release_keypoints"] = [int(x.strip()) for x in release_keypoints]
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata, task_dir, geometry=""):
        """Query GPT-4o and save the constraint program under ``task_dir`` (returns it).

        ``img`` is the (H,W,3) uint8 RGB keypoint-annotated scene; ``instruction`` the task
        text; ``metadata`` a dict merged with the parsed num_stages + grasp/release keypoints;
        ``geometry`` an optional measured-scene text block (grounded prompt templates).
        """
        self.task_dir = task_dir
        os.makedirs(self.task_dir, exist_ok=True)
        # save query image (cv2 expects BGR)
        image_path = os.path.join(self.task_dir, "query_img.png")
        cv2.imwrite(image_path, img[..., ::-1])
        messages = self._build_prompt(image_path, instruction, geometry)
        # stream back the response
        stream = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            stream=True,
        )
        output = ""
        start = time.time()
        for chunk in stream:
            print(f"[{time.time() - start:.2f}s] Querying OpenAI API...", end="\r")
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f"[{time.time() - start:.2f}s] Querying OpenAI API...Done")
        # save raw output
        with open(os.path.join(self.task_dir, "output_raw.txt"), "w", encoding="utf-8") as f:
            f.write(output)
        self._parse_and_save_constraints(output, self.task_dir)
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        return self.task_dir
