#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simplified script to format code in the current directory using clang-format.

This script formats files matching the given regex (default: *.cc,*.h)
in the current folder.

Usage:
    python simplified_indent.py
    python simplified_indent.py --dry-run
    python simplified_indent.py --regex "*.cpp,*.hpp"
"""

from __future__ import print_function

import argparse
import time
import shutil
import filecmp
import fnmatch
import os
import logging
import re
import subprocess
import sys
from tempfile import mkstemp

# Compatible versions from the original script
COMPATIBLE_CLANG_VERSIONS = ["6.0.0", "6.0.1"]


def parse_arguments_simplified():
    """
    Argument parser for the simplified script.
    """
    parser = argparse.ArgumentParser(
        "Run clang-format on all files in the current directory."
    )

    parser.add_argument(
        "-b",
        "--clang-format-binary",
        metavar="PATH",
        default=shutil.which("clang-format"),
    )

    parser.add_argument(
        "--regex",
        default="*.cc,*.h",
        help="Regular expression (regex) to filter files on "
        "which clang-format is applied (e.g., '*.cpp,*.hpp').",
    )

    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="If passed, only report files not formatted correctly without changing them.",
    )

    return parser.parse_args()


def check_clang_format_version(clang_format_binary, compatible_version_list):
    """
    Check whether clang-format with a suitable version is installed.
    """
    if not clang_format_binary:
        sys.exit("\n*** No 'clang-format' program found in your PATH.")

    try:
        clang_format_version = subprocess.check_output(
            [clang_format_binary, "--version"]
        ).decode()
        version_match = re.search(
            r"Version\s*([\d.]+)", clang_format_version, re.IGNORECASE
        )
        if version_match:
            version_number = version_match.group(1)
            # The original script checked against a list of very old versions.
            # In a modern context, you might want to check for a minimum version (e.g., 10.0.0 or higher)
            # but I'm keeping the original logic for compatibility with the source.
            if version_number not in compatible_version_list:
                logging.warning(
                    "Note: Found clang-format version %s which is not in the originally "
                    "specified compatible list (%s). Proceeding, but be aware of potential issues.",
                    version_number,
                    ", ".join(compatible_version_list),
                )
        else:
            logging.warning(
                "Could not determine clang-format version. Proceeding anyway."
            )

    except subprocess.CalledProcessError as subprocess_error:
        raise SystemExit(subprocess_error)
    except OSError as os_error:
        # This catches errors like "No such file or directory" for the binary path
        sys.exit(
            f"\n*** Error executing clang-format at path '{clang_format_binary}': {os_error}"
        )


def format_file(clang_format_binary, full_file_name, dry_run):
    """
    Applies clang-format to a single file and handles dry-run/overwriting.
    Returns True if the file was modified or reported as incorrect, False otherwise.
    """
    # Create a temporary file to store the formatted content using the system's safe temp location
    # We use delete=False to keep the file until we explicitly delete it later
    # The suffix ensures we get a unique, non-clashing name
    temp_file = mkstemp(
        prefix=os.path.basename(full_file_name) + ".", suffix=".tmp", text=True
    )
    temp_fd, temp_file_name = temp_file

    # We must close the file descriptor immediately after creation
    os.close(temp_fd)

    # 1. Copy the contents of the original file into the newly created temporary file
    shutil.copyfile(full_file_name, temp_file_name)

    try:
        # 2. Run clang-format on the temporary file IN PLACE
        # This is the standard, safer way to use clang-format for formatting: -i (in-place)
        subprocess.check_call(
            [clang_format_binary, "-i", temp_file_name],  # Note the '-i' flag here
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        # Clean up the temp file if the command failed
        os.remove(temp_file_name)
        logging.error(f"Error formatting {full_file_name}: {e.stderr.decode().strip()}")
        return True  # Report an issue

    # 3. Compare the original file and the formatted temporary file
    files_differ = not filecmp.cmp(full_file_name, temp_file_name)

    if files_differ:
        if dry_run:
            print(full_file_name)  # Print file path for dry-run
            os.remove(temp_file_name)
        else:
            # Overwrite the original file with the formatted content
            shutil.move(temp_file_name, full_file_name)
            logging.info(f"Formatted: {full_file_name}")
    else:
        # Clean up the temporary file if no changes were needed
        os.remove(temp_file_name)

    return files_differ


def process_directories(arguments):
    """
    Collects all files recursively starting from the current directory and formats them.
    """
    root_dir = "."  # Start from the current directory
    files_to_format = []
    patterns = [pattern.strip() for pattern in arguments.regex.split(",")]

    # Use os.walk to recursively traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prevent traversal into .git directories to speed things up
        if ".git" in dirnames:
            dirnames.remove(".git")

        for filename in filenames:
            # Check against all patterns in the regex list
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    # Construct the full file path
                    full_file_path = os.path.join(dirpath, filename)
                    files_to_format.append(full_file_path)
                    break
    
    if not files_to_format:
        logging.warning(
            f"No files found matching regex '{arguments.regex}' in {root_dir} or its subdirectories."
        )
        return

    # Process all collected files sequentially
    incorrectly_formatted_count = 0
    total_files = len(files_to_format)
    logging.info(f"Found {total_files} file(s) to process.")
    
    for i, full_file_name in enumerate(files_to_format):
        logging.debug(f"Processing file {i+1}/{total_files}: {full_file_name}")
        if format_file(
            arguments.clang_format_binary, full_file_name, arguments.dry_run
        ):
            incorrectly_formatted_count += 1

    if arguments.dry_run and incorrectly_formatted_count > 0:
        sys.exit(
            f"\n--- Found {incorrectly_formatted_count} incorrectly formatted file(s) out of {total_files} total."
        )
    elif arguments.dry_run and incorrectly_formatted_count == 0:
        logging.info("All files are correctly formatted.")

if __name__ == "__main__":
    START = time.time()
    PARSED_ARGUMENTS = parse_arguments_simplified()

    # Configure logging
    if PARSED_ARGUMENTS.dry_run:
        # For dry-run, only show warnings/errors and the list of incorrect files (printed by format_file)
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    else:
        # For actual run, show info messages (like 'Formatted: ...')
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Ensure clang-format is available
    check_clang_format_version(
        PARSED_ARGUMENTS.clang_format_binary, COMPATIBLE_CLANG_VERSIONS
    )

    # Check if the binary path is absolute or relative
    if (
        not os.path.isabs(PARSED_ARGUMENTS.clang_format_binary)
        and os.path.dirname(PARSED_ARGUMENTS.clang_format_binary) == ""
    ):
        # If it's just the name (e.g., 'clang-format'), we found it via shutil.which(), so it's fine.
        pass
    elif not os.path.exists(PARSED_ARGUMENTS.clang_format_binary):
        sys.exit(
            f"Error: clang-format binary not found at '{PARSED_ARGUMENTS.clang_format_binary}'."
        )

    logging.info("Starting recursive code formatting...")

    # The main processing function is now recursive
    process_directories(PARSED_ARGUMENTS)

    FINISH = time.time()

    # Only report total time if not in a dry-run where output should be clean
    if not PARSED_ARGUMENTS.dry_run:
        logging.info("Finished code formatting in: %f seconds.", (FINISH - START))