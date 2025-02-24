# RepoDiff Work in Progress

## Current Task: Single-Pass Diff Implementation
Converting RepoDiff to use a single git diff command with post-processing.

### Changes Required:
1. [x] Update config.json format to match new structure from PRD
   - [x] Replace "diffs" array with "filters" array
   - [x] Add support for file patterns and context lines
   - [x] Add support for method body thresholds

2. [x] Modify repodiff.py to implement single-pass approach:
   - [x] Update run_git_diff to use --unified=999999
   - [x] Create parse_unified_diff function to parse git diff output
   - [x] Implement post_process_files function for applying filters
   - [x] Add find_matching_rule function to match files with filters
   - [x] Create apply_signature_removal for C# method body handling
   - [x] Create apply_context_filter for adjusting context lines
   - [x] Add explanatory header to output
   - [x] Fix C# file line ordering in diff output

3. [ ] Testing:
   - [ ] Test with C# files for method body removal
   - [ ] Test with different context line settings
   - [ ] Test with multiple file patterns
   - [ ] Verify token counting still works
   - [ ] Test C# file line ordering with various file structures

4. [x] Documentation:
   - [x] Update README.md with new config format
   - [x] Document new features and behavior
   - [x] Add examples of new config options

### Future Tasks:
1. [ ] UI Implementation (PyQt)
   - Dark mode support
   - Commit selection
   - Filter configuration
   - Export options

2. [ ] Performance Optimization
   - Profile git diff performance
   - Optimize post-processing
   - Memory usage analysis

3. [ ] Testing Framework
   - Unit tests
   - Integration tests
   - PyQt UI tests 