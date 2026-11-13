# Tile Language Documentation

The documentation was built upon [Sphinx](https://www.sphinx-doc.org/en/master/).

## RISC-V Guides

- [Build and run on SG2044](get_started/BuildOnSG2044.md)
- [Target selection](get_started/targets.md)
- [GEMM example](../examples/riscv/gemm/README.md)
- [BOSC GEMM example](../examples/riscv/bosc/README.md)
- [RISC-V tests](../testing/python/riscv/README.md)

## Dependencies

Run the following command in this directory to install dependencies first:

```bash
pip3 install -r requirements.txt
```

## Build the Documentation

Then you can build the documentation by running:

```bash
make html
```

## View the Documentation

Run the following command to start a simple HTTP server:

```bash
cd _build/html
python3 -m http.server
```

Then you can view the documentation in your browser at `http://localhost:8000` (the port can be customized by appending `-p PORT_NUMBER` in the python command above).
