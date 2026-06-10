from huggingface_hub import snapshot_download

# This will download ONLY the 'Articulated_objects' folder and its contents
local_dir = snapshot_download(
    repo_id="x-humanoid-robomind/ArtVIP",
    repo_type="dataset",
    allow_patterns="Scenes/*",
    local_dir="./assets/ArtVIP"
)

print(f"Folder downloaded to: {local_dir}")