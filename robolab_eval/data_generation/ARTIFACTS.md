# Reference Spoon artifacts

These are the hashes of the accepted artifacts used to validate this pipeline.
They are transfer receipts, not files tracked by Git.

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `robolab_spoon_single50/source.hdf5` | 4,013,595,405 B | `8dc2bd5ad1e6002260a6b43b11dfb3d54b288a7f29f58a8459c63b3472dc5780` |
| `robolab_spoon_single50/demo_224.hdf5` | 5,485,419,652 B | `83fe1f8956a8b0d187453838da3431d1280d3d906c1fd9816b1d44ae1988a86b` |
| `awe_robolab_spoon_n40_bidir_j_v2/30000/model.safetensors` | 135,521,704 B | `bc83820d2e991d6d4d145a372d6748b6c7814392c9dae2aae58e090483d4c31e` |
| `awe_robolab_spoon_n40_bidir_j_v2/30000/action_norm_stats.json` | 11,376 B | `8e9918e6bb0a09f3ced0c93e52bf2d6365ae2e03c406664552ace0de5046853b` |
| `robolab_spoon_insertion/fk_fit.json` | 750 B | `aae3c8ffebe4bec262fe947672d851bb3ffa81e595306835a0a75b3384c4405a` |

The converted dataset validates as 50/50 successful, single-insertion demos,
26,629 total samples, H15xD8 proxy samples, 224x224 front/wrist RGB, achieved
next-joint arm labels, and continuous gripper commands. Its semantic-gripper
audit finds exactly two close/open transitions in every demo.
