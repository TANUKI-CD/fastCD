# Security policy

fastCD is a numerical library with no network access, file I/O, or dynamic
code execution. Security-relevant issues are therefore memory-safety defects:
out-of-bounds reads or writes, uninitialized memory, integer overflow in size
computations, or reference-counting errors in the Python extension.

Please report such issues privately to the maintainer through the contact
details on the canonical repository,
[git.tanuki-cd.com/algebraity/fastCD](https://git.tanuki-cd.com/algebraity/fastCD),
rather than in a public issue. Include the dimension, algorithm, and a
minimal input that reproduces the problem. A fix will be released as a patch
version and noted in `CHANGELOG.md`.
