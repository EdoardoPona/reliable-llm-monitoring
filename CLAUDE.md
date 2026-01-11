**Linting and Formatting**
After all code changes, you should check formatting and linting by running `./lint.sh`

**Testing**
- Tests can be run with `pytest`, but they are very slow becuase they involve local LLM inference. Avoid running the full test suite unless explicity told to.
- You should write and run specific tests for new functionality you add. Generally, you will be told do to so.
- Tests for a given module should be placed in a corresponding test file, next to the module file. For example, tests for `claude.py` should go in `test_claude.py` (at the same location).
