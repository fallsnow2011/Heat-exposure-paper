#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Information Figures Verification Script
======================================================
Verifies that all required SI figures are present and match LaTeX references.

Usage: python verify_si_figures.py
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check_mark(condition: bool) -> str:
    """Return colored checkmark or X based on condition."""
    return f"{GREEN}[OK]{RESET}" if condition else f"{RED}[FAIL]{RESET}"

def extract_figure_references(tex_file: Path) -> List[Tuple[int, str]]:
    """Extract all \includegraphics references from LaTeX file."""
    references = []
    with open(tex_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            match = re.search(r'\\includegraphics.*?\{(Supplementary_Fig_S\d+.*?\.pdf)\}', line)
            if match:
                references.append((line_num, match.group(1)))
    return references

def extract_figure_labels(tex_file: Path) -> List[Tuple[int, str]]:
    """Extract all \label{fig:S*} references from LaTeX file."""
    labels = []
    with open(tex_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            match = re.search(r'\\label\{(fig:S\d+)\}', line)
            if match:
                labels.append((line_num, match.group(1)))
    return labels

def get_file_size(file_path: Path) -> str:
    """Get human-readable file size."""
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def main():
    """Main verification function."""
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}Supplementary Information Figures Verification{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    # Get current directory
    current_dir = Path.cwd()
    print(f"Working directory: {BLUE}{current_dir}{RESET}\n")

    # Check if supplementary_information.tex exists
    tex_file = current_dir / "supplementary_information.tex"
    if not tex_file.exists():
        print(f"{RED}ERROR: supplementary_information.tex not found!{RESET}")
        print(f"Please run this script from the paper-draft-tex directory.\n")
        return 1

    print(f"{check_mark(True)} Found supplementary_information.tex\n")

    # Extract figure references from LaTeX
    print(f"{BOLD}[1] Extracting LaTeX figure references...{RESET}\n")
    references = extract_figure_references(tex_file)
    print(f"Found {len(references)} \\includegraphics references:\n")

    # Check if each referenced file exists
    all_files_exist = True
    file_info: Dict[str, Dict] = {}

    for line_num, filename in references:
        file_path = current_dir / filename
        exists = file_path.exists()
        all_files_exist = all_files_exist and exists

        # Also check for PNG version
        png_filename = filename.replace('.pdf', '.png')
        png_path = current_dir / png_filename
        png_exists = png_path.exists()

        file_info[filename] = {
            'line': line_num,
            'exists': exists,
            'path': file_path,
            'png_exists': png_exists,
            'png_path': png_path
        }

        status = check_mark(exists)
        print(f"  {status} Line {line_num:3d}: {filename}")
        if exists:
            print(f"      Size: {get_file_size(file_path)}")
        if png_exists:
            print(f"      PNG:  {get_file_size(png_path)}")
        print()

    # Check figure labels
    print(f"\n{BOLD}[2] Verifying figure labels...{RESET}\n")
    labels = extract_figure_labels(tex_file)
    print(f"Found {len(labels)} figure labels:\n")

    expected_labels = [f"fig:S{i}" for i in range(1, len(references) + 1)]
    actual_labels = [label for _, label in labels]

    labels_correct = expected_labels == actual_labels

    for i, (line_num, label) in enumerate(labels, 1):
        expected = f"fig:S{i}"
        correct = label == expected
        status = check_mark(correct)
        print(f"  {status} Line {line_num:3d}: {label} (expected: {expected})")

    # Summary
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}Verification Summary{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    print(f"Total figures referenced: {len(references)}")
    print(f"Files exist:              {check_mark(all_files_exist)} {'All present' if all_files_exist else 'Some missing'}")
    print(f"Labels correct:           {check_mark(labels_correct)} {'Sequential S1-S{}'.format(len(references)) if labels_correct else 'Issues found'}")

    # Count PNG versions
    png_count = sum(1 for info in file_info.values() if info['png_exists'])
    print(f"PNG versions available:   {png_count}/{len(references)}")

    # Calculate total size
    total_pdf_size = sum(info['path'].stat().st_size for info in file_info.values() if info['exists'])
    total_png_size = sum(info['png_path'].stat().st_size for info in file_info.values() if info['png_exists'])

    print(f"\nTotal PDF size: {total_pdf_size / (1024*1024):.2f} MB")
    print(f"Total PNG size: {total_png_size / (1024*1024):.2f} MB")

    # Final status
    print(f"\n{BOLD}{'='*80}{RESET}")
    if all_files_exist and labels_correct:
        print(f"{GREEN}{BOLD}[OK] VERIFICATION PASSED{RESET}")
        print(f"{GREEN}All figures are present and correctly referenced.{RESET}")
        print(f"{GREEN}Ready for LaTeX compilation!{RESET}")
        print(f"\nRun: {BLUE}compile_si.bat{RESET} (Windows) or {BLUE}bash compile_si.sh{RESET} (Linux/Mac)")
        return_code = 0
    else:
        print(f"{RED}{BOLD}[FAIL] VERIFICATION FAILED{RESET}")
        if not all_files_exist:
            print(f"{RED}Some figure files are missing.{RESET}")
        if not labels_correct:
            print(f"{RED}Figure labels are not sequential.{RESET}")
        return_code = 1

    print(f"{BOLD}{'='*80}{RESET}\n")

    return return_code

if __name__ == "__main__":
    exit(main())
