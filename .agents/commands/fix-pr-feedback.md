---
argument-hint: <PR-number>
description: "Fetch review comments and requested changes on your PR, summarize them, and produce an actionable fix plan."
name: "Fix PR Feedback"
---

You are helping the PR author address reviewer feedback on PR #$ARGUMENTS.

## Step 1: Fetch PR metadata and review feedback

Run all of these in parallel to gather context:

```bash
gh pr view $ARGUMENTS --json number,title,body,state,author,baseRefName,headRefName,reviewDecision
gh api repos/PrincetonUniversity/SPECFEMPP/pulls/$ARGUMENTS/reviews
gh api repos/PrincetonUniversity/SPECFEMPP/pulls/$ARGUMENTS/comments
gh pr diff $ARGUMENTS --repo PrincetonUniversity/SPECFEMPP
```

Also fetch inline review comments (these are different from PR-level comments):

```bash
gh api repos/PrincetonUniversity/SPECFEMPP/pulls/$ARGUMENTS/comments --paginate
```

And top-level issue-style comments on the PR:

```bash
gh api repos/PrincetonUniversity/SPECFEMPP/issues/$ARGUMENTS/comments --paginate
```

## Step 2: Parse and group feedback

For each review and comment, extract:
- **Who** left the feedback (reviewer username)
- **Where** it applies (file path and line, or general)
- **What** they asked for (quote the key sentence)
- **Status**: is it part of a "changes requested" review, a comment-only review, or a standalone comment?

Group feedback into categories:
1. **Must fix** -- from "changes requested" reviews or explicit blockers
2. **Should fix** -- suggestions from approvers or non-blocking reviewers
3. **Discussion** -- open questions or design debates needing a decision
4. **Resolved** -- threads already marked resolved or addressed in subsequent commits

Ignore bot comments (CI status, auto-labelers, etc.) unless they report a failure.

## Step 3: Understand the current code state

For each piece of feedback that references specific files or lines:
- Read the current version of the file to understand the present state
- Check if the feedback has already been addressed by a subsequent commit
- Note if the line numbers have shifted due to later changes

## Step 4: Present the summary

Produce a structured summary:

### Review Status
- **Overall decision**: (changes requested / approved with comments / pending)
- **Reviewers**: list each reviewer and their verdict

### Feedback Summary

| # | Reviewer | File | Line(s) | Category | Summary | Already addressed? |
|---|----------|------|---------|----------|---------|-------------------|
| 1 | ...      | ...  | ...     | Must fix | ...     | No                |
| 2 | ...      | ...  | ...     | Should fix | ...  | Yes (commit abc)  |
| ... | | | | | | |

For each item in the table, include the reviewer's exact quote (abbreviated if long) so the author can understand the intent without going back to GitHub.

## Step 5: Produce a fix plan

Create a numbered, prioritized action plan:

1. **What to fix**: clear description of what the reviewer wants changed
2. **Where**: file(s) and line range(s) in the current code
3. **How**: concrete approach or code snippet to address the feedback
4. **Reviewer**: who requested it, so we can tag them if needed

Order: Must fix first, then Should fix, then Discussion items (with proposed answers).

Skip items already addressed. For Discussion items, propose an answer or direction rather than leaving them open.

## Step 6: Wait for approval

Present the plan and ask:
- "Does this plan look right? Any items you want to skip, reorder, or handle differently?"
- "For the Discussion items, do you agree with the proposed direction?"

Do NOT start making changes until the user explicitly approves the plan.
