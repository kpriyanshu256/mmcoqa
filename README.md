# Environment Setup
```bash
conda create -n mmcoqa python=3.10 -y
conda activate mmcoqa
pip install -r requirements.txt
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y
pip install sentencepiece
```