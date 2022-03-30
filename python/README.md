
Python bindings

Running:

```sh
python3 -m venv .env
source .env/bin/activate
pip install maturin numpy matplotlib scipy
maturin develop --release
```

Then `import beta` in Python and you get `beta.pdf(x,a,b)`, `beta.cdf(x,a,b)`, `beta.ppf(x,a,b)`
