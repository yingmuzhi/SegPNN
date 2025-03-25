

## install

1. conda


2. pypi


3. sam2

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

4. checkpoints

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```


## framework

1. put the origin data `/home/yingmuzhi/SegPNN/src/20250324/origin/JS WFA+PV 60X.tif` into origin folder.

2. run `preprocess.py` to generate folder `output`