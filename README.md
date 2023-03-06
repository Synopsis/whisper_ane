This project is an attempt to re-define certain architectures to be optimised to run on the ANE. More specifically, it aims to adopt this Apple's [`ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) codebase to the [Whisper](https://github.com/openai/whisper) architecture. I've been working with the commit hash `d18e9ea` (there have been some changes to return signatures of internal functions -- that may not impact us, but I haven't tested that yet)

---

## Install

```bash
pip install git+https://github.com/openai/whisper.git@d18e9ea5dd2ca57c697e8e55f9e654f06ede25d0
pip install git+https://github.com/apple/ml-ane-transformers
pip install whisper_ane
```

## Export

See `notebooks/export-decoder.ipynb`.
You can set the export arguments (arch, precision, save directory, deployment target) in the top level cell, and then run all cells in the notebook.

The notebook checks for correctness by comparing predictions between the stock model and the ANE optimised model. Note that if running on a macOS system (as I did), the absolute differences only match up when the precision is set to FLOAT32. The differences are quite large in FLOAT16, and this is because on CPU, the model can only run on fp32 in PyTorch, so the comparisons are between an fp16 and fp32 model. In practice, predictions do match up.


## Benchmarks
All benchmarks have been run on a 16 inch M1 Max MBP 2021, 64GB RAM, MacOS 13.0, and can be viewed in [this google sheet](https://docs.google.com/spreadsheets/d/1CjLJuki5Lm2lSZzpYLYWs4NyPWp9EbYIkx2OQyYdRIc/edit#gid=0).


Key observations from the XCode benchmarks:
1. All FP16 exports (base and optimised arch) run on ANE 100%
2. All FP32 exports (base and optimised arch) run on GPU 100%
3. In FP16, the optimised arch is significantly faster
4. In FP32, the base arch is significantly faster

Caveats:
* It's unclear how exactly XCode runs their benchmark. We do not know what the batch size is
* Though the FP32 base model looks the fastest per XCode, we may see a different result when running the optimised ANE model on _larger batch sizes_ as it is 100% ANE accelerated

---

## Next Steps

* Export decoder
  * A lot of work has already been done towards this. But the optimised arch cannot be exported without changing the CoreMLtools source code because it doesn't play nice with flexible shapes. See [this issue](https://github.com/apple/coremltools/issues/1763) for more info
* Is it possible to remove the additional `Cast` operations happening in the FP16 encoder? It's unclear if this is happening due to the arch definition or the CoreML conversion process
