# Third-party software

- The retained [SpikingResformer](https://github.com/xyshi2000/SpikingResformer) implementation is under its original MIT license, included at `experiment/direction1_budget_rotation_20260907/vendor/SpikingResformer/LICENSE`. Its existing copyright notice is preserved.
- [SmoothSpike](https://github.com/CayleyZ/SmoothSpike) is an external dependency for language-model execution. The archived snapshot did not contain a repository-wide license, so its source and tokenizer snapshot are not redistributed here. Source identities are retained for comparison after obtaining the dependency separately. The offline numerical analyses do not import it.
- Python dependencies retain their respective licenses. Analysis dependencies are listed in `requirements-analysis.txt`; experiment environment snapshots record the versions used for execution.
- The additional experiment sources under `extensions/code/` retain the SpikingResformer and Spikformer MIT notices in their corresponding `constructor/vendor/` trees. The Spikformer compatibility file preserves the upstream model body while adapting the neuron import.
- The temporal-transfer `table_spwn` reference from Temporal Flexibility in SNNs is retained with its original MIT notice under each applicable `constructor/upstream/temporal_flexibility_in_SNN-main/` directory. The adapter uses the original sum/repeat table.

No repository-wide license is assigned by this release. Existing third-party notices apply to the corresponding files; they do not license the other materials.
