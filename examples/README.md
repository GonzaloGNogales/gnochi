# Example Conditioning Poses

These JSON files are some conditioning pose examples so users can test inference without preparing their own avatar.

The folder contains:

- 5 in-distribution interaction examples, prefixed with `id_`
- 5 out-of-distribution Mixamo motion examples, prefixed with `ood_`

Included ID examples:

- `id_dance14_frame000075.json`
- `id_fight16_frame000010.json`
- `id_piggyback17_frame000103.json`
- `id_sidehug37_frame000029.json`
- `id_talk22_frame000037.json`

Included OOD examples:

- `ood_hiphop_frame000002.json`
- `ood_cumbia_frame000000.json`
- `ood_taunt_frame000000.json`
- `ood_breakdance2_frame000004.json`
- `ood_swingkettleball_frame000002.json`

Run one example:

```bash
python -m gnochi.infer --input examples/ood_hiphop_frame000002.json --num-generations 10
```

Run all examples:

```bash
python -m gnochi.infer --input-dir examples --num-generations 10 --output-dir outputs/examples
```
