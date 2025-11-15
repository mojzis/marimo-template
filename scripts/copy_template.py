#!/usr/bin/env python3
"""
Simple script to copy the marimo template to a new directory.
Excludes .git directory from the copy.
"""

import argparse
import shutil
import sys
from pathlib import Path


def copy_template(target_dir: str) -> None:
    """Copy the template directory to a new location, excluding .git."""

    # Get the template directory (parent of scripts dir)
    template_dir = Path(__file__).parent.parent
    target_path = Path(target_dir).resolve()

    # Validation
    if target_path.exists():
        print(f"❌ Error: Target directory '{target_path}' already exists!")
        sys.exit(1)

    if target_path == template_dir:
        print("❌ Error: Target cannot be the same as the template directory!")
        sys.exit(1)

    # Define what to ignore
    def ignore_patterns(directory, files):
        """Ignore .git directory and __pycache__."""
        ignored = []
        if '.git' in files:
            ignored.append('.git')
        if '__pycache__' in files:
            ignored.append('__pycache__')
        return ignored

    # Copy the directory
    print(f"📁 Copying template from: {template_dir}")
    print(f"📂 To: {target_path}")
    print("⏳ Copying files...")

    try:
        shutil.copytree(
            template_dir,
            target_path,
            ignore=ignore_patterns,
            symlinks=True
        )
        print(f"✅ Template successfully copied to: {target_path}")
        print(f"\n💡 Next steps:")
        print(f"   cd {target_path}")
        print(f"   uv sync")
        print(f"   marimo edit main.py")
    except Exception as e:
        print(f"❌ Error during copy: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Copy the marimo template to a new directory"
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Target directory path (will be created)"
    )

    args = parser.parse_args()

    # If no target provided, prompt for it
    if not args.target:
        print("🎯 Marimo Template Copy Script")
        print("=" * 40)
        target = input("Enter target directory name: ").strip()

        if not target:
            print("❌ Error: Target directory cannot be empty!")
            sys.exit(1)
    else:
        target = args.target

    copy_template(target)


if __name__ == "__main__":
    main()
