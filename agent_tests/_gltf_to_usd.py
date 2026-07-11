"""Throwaway: convert a glTF/GLB mesh to USD via Isaac Sim's asset converter.

    /isaac-sim/python.sh agent_tests/_gltf_to_usd.py <in.glb> <out.usd>
"""
import asyncio
import sys

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import omni.kit.asset_converter as asset_converter  # noqa: E402


async def _convert(in_path: str, out_path: str) -> bool:
    context = asset_converter.AssetConverterContext()
    task = asset_converter.get_instance().create_converter_task(in_path, out_path, None, context)
    return await task.wait_until_finished()


def main(in_path: str, out_path: str) -> None:
    future = asyncio.ensure_future(_convert(in_path, out_path))
    while not future.done():  # kit's update loop drives the async converter task
        _app.update()
    ok = future.result()
    print(f"[gltf2usd] {'OK' if ok else 'FAILED'} -> {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    _app.close()
