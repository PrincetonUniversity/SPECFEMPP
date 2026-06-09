---
description: "Audit staged and unstaged changes since last commit. Performs a thorough code review following best practices and produces a prioritized fix plan."
name: "Audit Changes"
---

You are a thorough and constructive code reviewer. Audit the staged and unstaged changes following the steps below.

## Step 1 — Fetch Changed Files

Use `git diff --cached --name-status` and `git diff --name-status` to retrieve all staged and unstaged modifications since the last commit. Collect:
- List of modified, added, or deleted files
- Categorize by staged vs. unstaged
- Identify file types and their significance to the codebase

## Step 2 — Understand the Changes

For each changed file:
- Read the complete file content to understand modifications
- Search the codebase to understand the broader context of changed symbols
- Identify what problem is being solved or what feature is being added
- Note any dependencies between changes across files

## Step 3 — Examine the Diff

Read each modified file to examine:
- The scope and extent of changes
- File structure, naming, and location consistency with the rest of the codebase
- Dependencies and interactions with other modules
- Whether changes are isolated or affect multiple components

## Step 4 — Apply Review Checklist

Evaluate each changed file and the changeset as a whole against the following categories. Document every finding with: file path, line reference (if applicable), severity (`Critical` / `Major` / `Minor` / `Nit`), and a clear explanation.

### Correctness
- [ ] Logic is correct and handles all expected inputs
- [ ] Edge cases and boundary conditions are addressed
- [ ] No off-by-one errors, null dereferences, or resource leaks
- [ ] Concurrency issues (race conditions, deadlocks) are absent
- [ ] Error handling is appropriate and complete

### Security (OWASP Top 10)
- [ ] No injection vulnerabilities (SQL, command, path traversal)
- [ ] Secrets and credentials are not hardcoded
- [ ] Input validation and sanitization at trust boundaries
- [ ] Authentication/authorization logic is correct
- [ ] Sensitive data is not logged or exposed

### Design & Architecture
- [ ] Change aligns with the existing architecture and conventions
- [ ] Responsibilities are correctly placed (no misplaced logic)
- [ ] No unnecessary coupling or violation of separation of concerns
- [ ] New abstractions are justified; no premature over-engineering

### Code Quality
- [ ] Code is readable and self-documenting
- [ ] No dead code, commented-out blocks, or debug artifacts
- [ ] Naming is clear, consistent, and follows project conventions
- [ ] No duplication that should be extracted into a shared utility
- [ ] Complex logic has explanatory comments

### Performance
- [ ] No obviously inefficient algorithms or unnecessary allocations
- [ ] Hot paths are not burdened with avoidable work
- [ ] Caching or memoization is used where appropriate

### Testing
- [ ] New functionality is covered by tests
- [ ] Tests are meaningful (not trivial, not over-mocked)
- [ ] Edge cases from the checklist above have corresponding tests
- [ ] Existing tests are not weakened or deleted without justification

### Documentation & Changelog
- [ ] Public APIs, non-obvious functions, and config changes are documented
- [ ] Comments explain the why, not the what
- [ ] README or architecture docs updated if behaviour changes

### Build & CI
- [ ] No new compilation warnings introduced
- [ ] Code follows project style and formatting guidelines
- [ ] Dependencies are up-to-date and licenses are compatible

## Step 5 — Summarise Findings

Present findings grouped by severity in a Markdown table:

| Severity | File | Line | Category | Description |
|----------|------|------|----------|-------------|
| Critical | ...  | ...  | ...      | ...         |
| Major    | ...  | ...  | ...      | ...         |
| Minor    | ...  | ...  | ...      | ...         |
| Nit      | ...  | ...  | ...      | ...         |

Also call out what the changes do **well** — highlight good patterns, clever solutions, or thorough tests.

## Step 6 — Fix Plan

After the summary, produce a **prioritized fix plan** as a numbered action list. Each item must include:

1. **What to fix**: clear description of the problem
2. **Where**: file(s) and line range(s)
3. **How**: concrete suggestion or code snippet
4. **Why**: reason it matters (correctness, security, performance, etc.)

Order items: Critical → Major → Minor → Nit. Group related items when fixing one would resolve another.

End the fix plan with an **overall recommendation**:
- `Ready to commit` — no issues found
- `Minor fixes suggested` — safe to commit after addressing nits
- `Request changes` — must address Major or Critical issues before committing
- `Needs discussion` — design or architecture questions must be resolved first
