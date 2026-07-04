# tflite-builder

## *How to* guide

Firstly, you need to setup the environment by:
```bash
source setupenv
```

Then you can run `tf -h` for further information.

### Cloning

To clone a specific version of TF, run:
```bash
tf clone -b <BRANCH_OR_TAG>
```

### Building

To build an already cloned version of TF, do:
```bash
tf build -b <BRANCH_OR_TAG> [--cmake | --bazel]
```

### Installing

To collect the built files into a package, run:
```bash
tf install -b <BRANCH_OR_TAG> -s {cmake,bazel} [-i <CUSTOM_INSTALL_PATH>]
```

### Testing

To verify that the built library is correct, you can run a simple C++ program to check that:
```bash
tf run -i <INSTALL_PATH>
```

### Example

The following example uses:

- tag: `v2.21.0`
- Bazel build system

```bash
. setupenv
tf clone -b v2.21.0
tf build -b v2.21.0 --bazel
tf install -b v2.21.0 -s bazel
tf run -i builds/v2.21.0--bazel
```

> Note that `tf install` without provided `-i/--install-path` flag, will use a default installation path: `builds/<branch-tag>--<source>`.
