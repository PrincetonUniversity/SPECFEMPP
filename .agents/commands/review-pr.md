---
argument-hint: <PR-number>
---

You are a code reviewer for SPECFEM++. Review PR #$ARGUMENTS from the origin repository.

## Step 1: Fetch PR information

Run all of these to gather context:

```bash
gh pr view $ARGUMENTS --repo PrincetonUniversity/SPECFEMPP --json number,title,body,state,author,baseRefName,headRefName,reviewDecision
gh pr diff $ARGUMENTS --repo PrincetonUniversity/SPECFEMPP
gh api repos/PrincetonUniversity/SPECFEMPP/pulls/$ARGUMENTS/reviews
gh api repos/PrincetonUniversity/SPECFEMPP/pulls/$ARGUMENTS/comments
```

## Step 2: Analyze the diff

For each changed file, evaluate against these SPECFEM++ conventions:

### Style checks
- [ ] `snake_case` for classes, functions, variables, namespaces, files
- [ ] `CamelCase` only for template parameters
- [ ] No `using namespace` at file or namespace scope (exception: `specfem::units::unit_symbols` inside function bodies)
- [ ] No anonymous namespaces (unity build compatibility)
- [ ] `#pragma once` for header guards
- [ ] All member variables initialized
- [ ] Namespace depth <= 3 levels (`specfem::X::Y`)
- [ ] Utilities in `specfem::utilities`, not in specific `_impl` namespaces
- [ ] Fully qualified names (explicit `std::`, `specfem::`, etc.)
- [ ] Include order: project, third-party, standard library

### Architecture checks
- [ ] Public API additions have corresponding unit tests
- [ ] No backward-incompatible API removals without deprecation
- [ ] Input validation with clear error messages at system boundaries
- [ ] Kokkos views not captured by value in hot-path lambdas
- [ ] Return view-containing structs by `const&`, not by value
- [ ] No nested lambdas inside `KOKKOS_LAMBDA`
- [ ] Functions called from `KOKKOS_LAMBDA` marked `KOKKOS_INLINE_FUNCTION`

### Documentation checks
- [ ] New public APIs documented with `@brief`, `@param`, `@return`, `@tparam`
- [ ] RST API page exists or is updated for new public classes
- [ ] No `@file` directives added
- [ ] Documentation references match actual function/class names

## Step 3: Also read any existing review comments

Check if reviewers have already flagged issues. Incorporate their feedback into your
analysis to avoid duplicating concerns and to validate whether suggested fixes have
been addressed.

## Step 4: Categorize findings

Group issues by severity:
- **Critical**: Bugs, correctness issues, ODR violations, uninitialized state
- **Major**: Convention violations, missing tests, API breakage, `using namespace`
- **Minor**: Style inconsistencies, documentation gaps, suboptimal patterns
- **Nit**: Formatting preferences, comment improvements, naming suggestions

## Step 5: Present review

Produce a structured review:

| Severity | File | Line | Issue | Suggestion |
|----------|------|------|-------|------------|
| ... | ... | ... | ... | ... |

Include praise for well-done aspects. End with an overall recommendation:
- **Approve** -- no issues or only nits
- **Request changes** -- major or critical issues found
- **Needs discussion** -- design questions that require maintainer input
