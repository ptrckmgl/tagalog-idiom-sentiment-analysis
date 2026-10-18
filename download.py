from datasets import load_dataset
ds = load_dataset("uonlp/CulturaX", "tl", split="train")  # add streaming=True to avoid a full download
print(ds[0])
ds.to_parquet("culturax_tl.parquet")