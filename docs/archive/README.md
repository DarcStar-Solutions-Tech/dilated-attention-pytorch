# Archive Directory

This directory contains historical documentation that has been superseded by newer versions but is preserved for reference.

## Purpose
- Preserve historically significant documents
- Keep superseded reports for reference
- Document the evolution of the project
- Maintain audit trail of major changes

## What Goes Here
Documents are moved to archive when:
- A newer, more comprehensive version exists
- The implementation has been significantly changed
- The analysis is no longer relevant to current code
- It documents a major milestone or breakthrough

## What Stays Here
Keep documents that:
- Document major breakthroughs or turning points
- Show the evolution of key implementations
- Have historical significance to the project
- Provide context for major architectural decisions

## Directory Structure
```
archive/
├── reports/           # Archived technical reports
│   └── archived/      # Older archived reports
├── *.md               # Individual archived documents
└── README.md          # This file
```

## Finding Archived Documents
Documents retain their original timestamps for tracking:
```bash
# Find all June 2025 archives
ls *2025-06*.md

# Find ring attention archives
ls *ring-attention*.md
```

## Important Historical Documents
Some key documents preserved here:
- Ring Attention breakthrough (June 27, 2025)
- Major refactoring summaries
- Original implementation analyses

## Note
Before deleting any archived document, consider its historical value. Some documents that seem redundant may provide important context for understanding the project's evolution.