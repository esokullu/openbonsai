"""``python -m openbonsai``. The stage CLI is rebuilt in phase P1 (FRAMEWORK_PLAN.md).

The legacy CLI moved to research/legacy/cli.py; tag legacy-staged-pipeline runs it.
"""
import sys


def main(argv=None):
    print("openbonsai: the framework CLI is not built yet (see FRAMEWORK_PLAN.md). "
          "The legacy CLI is at tag legacy-staged-pipeline.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
