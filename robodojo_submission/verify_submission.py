#!/usr/bin/env python3
"""Verify preserved core files and, optionally, pinned official evaluator sources."""
import argparse
import json
from pathlib import Path

from manipisa_robodojo.provenance import profile, verify_core_tree, verify_official_checkouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path)
    args = parser.parse_args()
    report = {"core": verify_core_tree(), "submission": profile(),
              "scope": "Source and configuration verification; no physics or official score"}
    if args.robodojo:
        report["official_sources"] = verify_official_checkouts(args.robodojo)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
