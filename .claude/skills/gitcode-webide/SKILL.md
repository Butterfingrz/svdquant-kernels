---
name: gitcode-webide
description: Drive a GitCode CodeArts WebIDE session (910B3 NPU) from chrome MCP — push commands into the in-browser xterm terminal, capture output via a git transit (commit on a scratch branch → push to gitcode → fetch + git show locally). Use whenever the user wants kernel/build output from the WebIDE pulled back with byte-level fidelity. Requires chrome-devtools MCP attached to the user's browser with the WebIDE instance tab open. Sibling skill to `gitcode-space-iter` (Space/Gradio form factor) but for the IDE/terminal form factor.
---

# GitCode CodeArts WebIDE — terminal driver + git transit

GitCode's **CodeArts WebIDE** (`devstation.connect.huaweicloud.com/devdesktop/webide`)
is a browser-hosted VSCode that gives direct shell access to a 910B3 NPU box.
This skill runs commands in that shell and pulls their output back to the
local machine with **byte-level fidelity** by treating git as the transit.

The WebIDE's terminal is xterm.js rendered onto a `<canvas>` — its text is
**not in the DOM**, so chrome MCP `take_snapshot` / `wait_for` / DOM scraping
all fail for command output. Screenshot OCR is also unreliable (numerals
silently flip: `3203 / 65536` was OCR'd as `0 / 65536` in one validation run).
The only safe channel is: **write output to a file, commit, push to gitcode,
read locally via `git show`**.

## When to use

- User has the WebIDE open at `https://devstation.connect.huaweicloud.com/devdesktop/webide/instance/<uuid>?folder=...`
- They want to run something on the 910B3 (build, test, kernel smoke, npu-smi, dump...) and want **precise output** locally
- The same NPU repo is checked into `gitcode.com/<user>/<repo>`

Skip this skill if:
- The task doesn't need precise output (a yes/no "did it run" from a screenshot is enough)
- The Space form factor (Gradio + auto-rebuild) fits better — use `gitcode-space-iter` instead

## Required browser state

`mcp__chrome-devtools__list_pages` must show the WebIDE instance URL. If
multiple WebIDE tabs are open, `select_page` to the one whose `folder=`
matches the target repo.

## The transit pattern

### Step 1 — focus the terminal

`take_snapshot`, find the element whose name starts with `终端 1，bash`
(role textbox), click it. Verify focus by seeing it appear as
`focusable focused` in the next snapshot. If the terminal panel is not
visible, the snapshot will be missing it — open it via the file-tree
status bar tab named `终端 (Ctrl+\`)`.

### Step 2 — first-time setup (once per WebIDE session)

```bash
cd /mnt/workspace/gitCode/<user>/<repo>
git remote add gitcode https://gitcode.com/<user>/<repo>.git
git config user.email dev@webide.local
git config user.name webide
git remote -v   # verify both origin (github) + gitcode show up
```

Do **`git remote add` as a standalone command**, not chained with a later
`git push`. The credential dialog (next step) can interrupt a `&&` chain,
making `remote add` silently fail while only the push error is visible.

### Step 3 — branch from clean main

```bash
git fetch origin main
git checkout -B scratch/<topic> origin/main
```

`origin/main` (not local `HEAD`) — the WebIDE drops 2 tooling-config files
(`.arts/`, `.opencode/`) into the working copy that show up as pending
changes. Branching from `origin/main` skips them so the scratch branch is
clean.

### Step 4 — run the command, redirect to a tracked path

```bash
mkdir -p tmp/<topic>
<cmd> > tmp/<topic>/out.log 2>&1
# Or for multi-output: redirect each artifact to its own file
npu-smi info > tmp/<topic>/npu_smi.txt 2>&1
nm build/libfoo.so > tmp/<topic>/symbols.txt 2>&1
```

### Step 5 — commit and push

```bash
git add -f tmp/<topic>/                         # tmp/ is in .gitignore — need -f
git commit -m '<topic>: run output'
git push -u gitcode scratch/<topic>
```

**Credentials**: the WebIDE does **not** pre-cache git credentials for
either gitcode or github. The very first `git push` of a session triggers
a dialog at the top of the WebIDE window asking for Username + Token (PAT).
**The user must enter it manually** — do not attempt to type the token
through chrome MCP `type_text` (it's their secret, and any failed attempt
leaves it in tool-call telemetry).

Tell the user what's pending ("WebIDE is asking for your PAT for
`https://gitcode.com` — please enter it in the dialog at the top of the
window") and wait for them to confirm before continuing. After they
authenticate once, the credential helper caches the token for the rest of
that WebIDE session.

### Step 6 — fetch and read locally

```bash
git fetch gitcode scratch/<topic>
git show gitcode/scratch/<topic>:tmp/<topic>/out.log
git show gitcode/scratch/<topic>:tmp/<topic>/npu_smi.txt
```

`git show <ref>:<path>` reads the blob directly without a working-tree
checkout — perfectly safe even if the local working tree is dirty or on a
different branch.

## What "done" looks like

After `git push`, the WebIDE terminal shows (visible in a screenshot, no
need to OCR exact text — pattern-match the shape):

```
remote: To create a merge request for scratch/<topic>, visit:
remote:   https://gitcode.com/<user>/<repo>/merge_requests/new?source_branch=scratch/<topic>
remote: Start Git Hooks Checking                          [PASSED]
To https://gitcode.com/<user>/<repo>.git
 * [new branch]      scratch/<topic> -> scratch/<topic>
Branch 'scratch/<topic>' set up to track remote branch 'scratch/<topic>' from 'gitcode'.
```

The `[PASSED]` is the gitcode-side hook gate. If you see `[FAILED]` here,
read the lines just above for the rejection reason (commit-message lint,
file-size limit, banned-pattern scan, etc.).

The local-side `git fetch gitcode` resolving to `* [new branch]
scratch/<topic> -> gitcode/scratch/<topic>` is the end-to-end success
sentinel.

## Pitfalls

1. **xterm canvas defeats DOM scraping**. `mcp__chrome-devtools__take_snapshot`
   and `wait_for` cannot see terminal text. For "did the command finish"
   signals, watch the bottom status bar (e.g. branch name `scratch/<topic>*`
   indicates uncommitted changes; the `*` dropping means a commit landed)
   or `take_screenshot` + visual recognition for coarse status. Use file
   contents via `git show` for **anything that must be exact** (numbers,
   stack traces, log paths).

2. **Don't push to `main`**. Always `scratch/<topic>`. The WebIDE will add
   `.arts/`, `.opencode/`, possibly `.vscode/` to the working tree;
   anything on main propagates back to all collaborators. Scratch branches
   can be deleted freely after the run.

3. **`tmp/` and `log/` are .gitignored** (per repo CLAUDE.md). Use
   `git add -f` to override for transit; the scratch branch is throw-away
   so polluting `tmp/` history is fine.

4. **Long-running commands**: the WebIDE web socket can drop on idle.
   Wrap multi-minute commands in `nohup ... &` and write a sentinel file
   (`echo $? > tmp/<topic>/exit.code`), then poll by `git push`-ing every
   ~30s with the latest `out.log` snapshot. Don't `tail -f` interactively
   in this skill — there's no good way to read incremental terminal text.

5. **Multi-step `&&` chains break across credential dialogs**. Split
   `remote add` and `push` into separate commands. If you see
   `fatal: 'gitcode' does not appear to be a git repository`, that's the
   smoking gun — `remote add` was eaten by an earlier prompt.

6. **Don't OCR command output for decisions**. If the next step branches
   on a number or status string, `git show` the log file. Visual
   recognition of canvas-rendered xterm text fails in subtle ways
   (`0` ↔ `3203`, etc.) that look correct.

## Related skills / memory

- `gitcode-space-iter` (same platform, Space/Gradio form factor — rebuild + log-panel scrape; complementary, not redundant)
- Memory: `gitcode_webide_channel`, `feedback_space_log_panel_blanks_on_exit`,
  `feedback_remote_exec_via_bash_script`
