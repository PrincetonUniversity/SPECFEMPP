---
description: "Scan the codebase for inconsistencies against declared naming and organization conventions."
name: "Review Conventions"
---

You are a code conventions auditor for SPECFEM++. Scan the codebase for inconsistencies
against the project's declared conventions and produce a structured report.

## Audit categories

### 1. Class/struct naming (snake_case vs CamelCase)

Search for all class and struct declarations in `core/specfem/`. Flag any that use
CamelCase when the project convention is `snake_case`.

Note: Template parameters should be `CamelCase` -- that is correct and should NOT
be flagged. GoogleTest test fixtures are also `CamelCase` by gtest convention.

Use this grep pattern to find declarations:
```bash
grep -rn "^\(class\|struct\) [A-Z]" core/specfem/ --include="*.hpp" --include="*.cpp" --include="*.tpp"
```

Filter out template parameters and forward declarations.

### 2. `using namespace` violations

Search for `using namespace` in all `.hpp`, `.cpp`, `.tpp` files:
```bash
grep -rn "using namespace" core/ tests/ --include="*.hpp" --include="*.cpp" --include="*.tpp"
```

Flag any at file/namespace scope. The only acceptable usage is
`using namespace specfem::units::unit_symbols;` inside function bodies.

### 3. Anonymous namespaces

Search for `namespace {` patterns:
```bash
grep -rn "^namespace {" core/ --include="*.hpp" --include="*.cpp" --include="*.tpp"
```

All occurrences are violations (unity build incompatibility). Each should be
converted to a `_impl` suffix namespace.

### 4. Namespace depth > 3

Search for namespace declarations exceeding `specfem::X::Y::Z` (more than 3 `::` separators).

### 5. Missing Doxygen on public classes

Search for public class/struct declarations without a preceding `@brief`:
```bash
grep -B5 "^\(class\|struct\) " core/specfem/ --include="*.hpp" -r
```

Check if the 5 lines before each declaration contain `@brief`.

### 6. `@file` directives

Search for `@file` in headers:
```bash
grep -rn "@file" core/ --include="*.hpp"
```

These should be removed for consistency (project convention: no `@file` directives).

## Output format

Present findings as a structured markdown questionnaire:

```
# SPECFEM++ Convention Audit

## 1. CamelCase classes (should be snake_case)

For each item, choose: [ ] Rename | [ ] Keep | [ ] Defer

| Current Name | File | Suggested Name |
|---|---|---|

## 2. `using namespace` violations

| File:Line | Context | Recommended Fix |
|---|---|---|

## 3. Anonymous namespaces

| File:Line | Contents | Replace With |
|---|---|---|

## 4. Namespace depth > 3

| Namespace | File | Suggested Restructure |
|---|---|---|

## 5. Missing Doxygen

| Class/Struct | File |
|---|---|

## 6. Stale `@file` directives

| File:Line |
|---|
```

Group items by module/directory for easier batch processing.
