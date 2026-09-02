# Interactive Programmatic Policy Lab

A local research UI for the loop: **play the environment -> record
episodes/transitions -> select evidence -> render a prompt -> call an LLM
-> get an executable policy -> run it online -> inspect the result ->
select new evidence -> repeat.**

Every policy, prompt, LLM call, evidence selection, run and episode is
persisted with enough provenance to answer, for any policy in the library:
*why does this policy exist, and what evidence/prompt/model produced it?*

## Purpose

This is a human-operated version of the same kind of programmatic-policy
research loop the repo's autonomous agents run in the training pipeline
(`main.py`/`training/trainer.py`) -- except here a person drives each step:
choosing what evidence matters, writing/tweaking the prompt, and deciding
what to try next. The backend is deliberately UI-agnostic (see
"Architecture") so the same pipeline could later be driven by an autonomous
loop instead of a browser.

**Dependency boundary.** This package intentionally depends on only two
things outside itself: the repo's `environments/` (to interact with) and
`llm/` (to receive an LLM client/session from). It does not import
`agents`, `config`, `training`, or `utils` at the repo root -- see "How it
integrates with the existing repository" below for exactly what's reused
vs. what was copied into this package to keep that boundary.

## Architecture

```
EnvironmentAdapter -> Transition -> ExperienceStore -> EvidenceSelection
      -> PromptRenderer -> LLMService -> LLMCall -> Policy -> PolicyRunner
      -> EnvironmentAdapter -> Transition (loop closes)
```

```
programmatic_interactive_lab/
    __main__.py         entry point: builds one LabContext, registers UI routes
    app.py               composition root -- LabContext wires every module below
    cli.py                argparse CLI (kept out of `config.py` to avoid a name
                          collision with the parent repo's own config.py)
    core/
        session.py        LabSession: one named research workspace
        environment.py     EnvironmentAdapter: wraps one repo env for the lab
        experience.py      Episode/Transition persistence (ExperienceStore)
        interaction.py     the ONE place env.step() is called + a transition saved
        evidence.py         EvidenceSelection / multiple named "Evidence Baskets"
        formatters.py       TransitionFormatter: evidence -> LLM-facing text
        prompts.py           PromptTemplate (versioned) + PromptRenderer
        llm.py                LLMService -> LLMCall (own LLM_PRESETS registry)
        llm_models.py          user-managed llm_models.json model picker registry
        policies.py           Policy / PolicyStore (validation + lineage)
        runs.py                Run: exploratory online policy execution
        evaluation.py          Evaluation: fixed-configuration comparisons
        training.py            automated generate/run/improve loop (Train page)
        metrics.py             per-session return-vs-(steps/tokens/time) curves
    execution/
        sandbox.py            self-contained policy compile/validate/action-check
        validation.py        parse/validate LLM-generated policy source
        worker.py            subprocess entry point (compiles + runs policy())
        policy_runner.py     PolicyRunner: process-isolated execution + timeout
    storage/
        database.py          SQLite connection + schema + generic CRUD helpers
        models.py             one dataclass per persisted entity (to_row/from_row)
        artifacts.py          filesystem layout for states/renders/policy source
        serialization.py      explicit numpy/plain-value (de)serialization
    ui/
        layout.py             shared left-nav chrome
        state.py               process-wide "current session"/"current basket" state
        components.py           small shared widgets (e.g. the basket-picker dropdown)
        pages/                 one module per nav item (play, episodes, evidence,
                               templates, prompt_studio, policies, llm_calls, runs,
                               evaluations, train, session) -- no giant callback file
    tests/                    pytest suite for the backend (see below)
    data/                     SQLite DB + per-session artifacts (git-ignored)
```

The UI (`ui/`) only ever calls into a `LabContext` (built in `app.py`); it
never touches `storage`/`execution` internals directly. That's what keeps
the backend usable by something other than NiceGUI later.

## How it integrates with the existing repository

Nothing outside `programmatic_interactive_lab/` was modified. Reused, not
duplicated:

- **Environments** (`environments/*.py`) -- `core/environment.py`'s
  `EnvironmentAdapter` wraps whatever this package's own `make_env()`
  builds. `make_env`/`ENV_CONFIGS` are a small registry *owned by this
  package* (constructor + default params per environment name), modeled on
  the root `config.py`'s `ENV_CONFIGS`/`make_env` pattern but not imported
  from it -- so `--env SimpleGridEnv --env-overrides '{...}'` still works
  exactly like the root `main.py`'s CLI, without this package depending on
  the root `config.py`. The adapter introspects each environment module's
  `ACTION_NAMES` (already exported by `simple_grid_env.py`/
  `obstacle_grid_env.py`/`rule_discovery_grid.py`) to derive human controls
  and keyboard shortcuts generically -- no per-environment adapter subclass
  needed for any of the three environments already in this repo. It also
  reads each module's `UPPER_CASE` cell-type constants (`EMPTY`/`AGENT`/
  `GOAL`/...) to build the cell-code legend in
  `observation_space_description()` (told to the LLM once per prompt), but
  deliberately does **not** use that legend to reformat individual
  observations -- `format_state_for_llm()` is purely type-based (array/
  dict/list/scalar), not environment-specific, so it shows exactly what the
  observation is for any environment, grid or otherwise.
- **Action validation** -- `execution/sandbox.is_valid_action`/
  `normalize_action`, used by `EnvironmentAdapter`. Adapted from (not
  imported from) `agents/programmatic_scientist_agent.py`'s helpers of the
  same name, copied into this package so it has no `agents` dependency.
- **LLM client/session** (`llm/client.LLMClient`, `llm/session.ChatSession`)
  -- `core/llm.LLMService` builds a `ChatSession` per call with
  `max_messages=1`, the same "stateless-feeling single-turn call" pattern
  every agent in this repo already uses. Provider/model/credential
  resolution has two paths: the legacy `core.llm.LLM_PRESETS`/
  `build_llm_client`, modeled on the root `config.py`'s `LLM_CONFIGS`/
  `make_llm_client` but not imported from it -- so `--llm GEMINI` still
  reads `GEMINI_MODEL`/`GEMINI_API_KEY`/`GEMINI_BASE_URL` the same way,
  without depending on the root `config.py` -- and the newer
  `core/llm_models.py` registry (`llm_models.json`, git-ignored), which
  lets a researcher list several models with credentials written directly
  into the file and pick among them per call in Prompt Studio, independent
  of the launch default.
- **Policy compile/validate** -- `execution/sandbox.compile_policy_source`/
  `strip_code_fences`, used by both `execution/validation.py` (pre-run
  validation) and, inside the isolated subprocess, `execution/worker.py`
  (actual execution) -- the same function both places, so "validated" and
  "runnable" can never diverge. Adapted from (not imported from)
  `agents/programmatic_scientist_agent.py`'s `_compile_policy_source`/
  `_strip_code_fences`: the AST import-rejection + restricted-builtins-
  `exec` sandboxing and the `def policy(observation): ... return action`
  entry-point convention are exactly this repo's existing convention,
  copied into `execution/sandbox.py` so this package has no `agents`
  dependency either.
- **Output/data conventions** -- `storage/artifacts.py`'s
  `data/sessions/<id>/{states,renders,policies,exports}/` layout mirrors the
  root `outputs/<env>/<config>/...` convention of predictable, inspectable
  directories with large artifacts on disk and metadata in a structured
  store (SQLite here instead of JSON files, since the lab needs relational
  querying -- filter transitions by actor/tag/episode/run -- that a flat
  `metrics.jsonl` isn't built for). `core/metrics.py`'s per-session
  performance curves (see "Core workflow" and the Session page) are
  likewise modeled on `training/trainer.py`'s `plot_training_metrics`
  (same three x-axes: environment steps, LLM tokens, wall-clock time)
  without importing `training`.

**Summary: this package imports from exactly two repo modules outside
itself -- `environments` and `llm`.** Everything else that resembles
`agents`/`config`/`training` conventions (action validation, policy
sandboxing, environment/LLM registries, performance curves) is a
self-contained adaptation living inside `programmatic_interactive_lab/`,
not an import of those modules.

The one thing genuinely new (not adapted from anywhere in the repo) is
**process isolation** for executing generated code: the existing agents
`exec()` policies in-process (fine for an unattended training loop where a
bad program just costs one episode). An interactive tool with a person
waiting on the other end needs a hung or resource-hogging generated
program to not take the UI down with it, so `execution/policy_runner.py`
runs each policy in its own OS process with a hard per-step timeout (see
"Known limitations" for the isolation model's actual scope).

## Installation

From the repo root, with the existing repo's own dependencies already
installed (`torch`, `gymnasium`, `numpy`, `openai`, `httpx` -- see the root
`requirements.txt`), add the one new dependency:

```bash
pip install -r programmatic_interactive_lab/requirements.txt
```

There are two ways to give the lab LLM credentials:

1. **The launch default** (env vars, the parent repo's own convention) --
   set the vars matching whichever `--llm` preset you pass (default
   `GEMINI`):

   ```bash
   export GEMINI_MODEL=...
   export GEMINI_API_KEY=...
   export GEMINI_BASE_URL=...
   ```

2. **The model registry** (`llm_models.json`) -- lets you list any number
   of models, each with its own name/URL/API key, and pick among them per
   call in Prompt Studio's **Model** dropdown, no relaunch needed:

   ```bash
   cp programmatic_interactive_lab/llm_models.example.json programmatic_interactive_lab/llm_models.json
   ```

   Then edit it -- one entry per model, e.g.:

   ```json
   [
     {"name": "gemini-flash", "model_name": "gemini-2.5-flash",
      "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
      "api_key": "..."}
   ]
   ```

   `name` is the label shown in the dropdown; `model_name`/`url`/`api_key`
   map directly to `llm.client.LLMClient`'s `model`/`base_url`/`api_key`.
   Optional per-entry `temperature`/`timeout`/`max_retries`/`stream`
   override the lab's defaults. This file holds API keys directly and is
   git-ignored -- never commit it.

You do not need either set up just to browse the app -- they're only read
when you click "Call LLM" in Prompt Studio.

## Launching

```bash
python -m programmatic_interactive_lab
```

Then open the printed URL (default `http://127.0.0.1:8080`) -- with no
`--env` or `--session-id` given, the first thing you see is the **Setup**
page (`ui/pages/setup.py`): pick an environment from a dropdown, adjust its
parameters (a plain input per entry in that environment's
`core.environment.ENV_CONFIGS["params"]`, type inferred from the default
value -- new environments/params need no setup-page code, just a registry
entry), optionally name the session, and click **Start session**. That's
the *only* place an environment gets chosen -- a session's environment is
fixed for its whole lifetime (every Policy is validated against a specific
observation/action space), so there's no "change environment" control
anywhere else in the app; want a different one, start another new session
instead (the **Session** page's **Start a new session** button reopens this
same Setup page any time afterward, without restarting the process -- your
old session stays put, switchable from the list below it).

Useful flags:

```bash
python -m programmatic_interactive_lab \
    --env ObstacleGridEnv \
    --env-overrides '{"size": 10, "obstacle_density": 0.2}' \
    --llm GEMINI \
    --session-name "obstacle-grid exploration" \
    --port 8080
```

- `--env` / `--env-overrides` -- optional shortcut for scripted/batch
  launches: skip the Setup page and create the session immediately, same
  convention as the root `main.py`'s `--env`/`--env-overrides`. Omit both
  to use the Setup page instead.
- `--llm` / `--llm-overrides` -- LLM preset (`GEMINI`/`VULCAN` in the root
  `config.py`) + JSON overrides.
- `--session-id <id>` -- reopen an existing session instead of creating a
  new one (also skips the Setup page, since that session's environment is
  already fixed). Every session ever created is listed (name/environment/
  created time) on the **Session** page, with a **Delete** button for old
  ones (see "Core workflow" below) -- or find the id printed at launch.
- `--session-name` -- name for a newly created session.
- `--host` / `--port` / `--reload` -- passed straight to `nicegui.ui.run`.

## Core workflow

1. **Play** -- click/keyboard-control the environment, or switch the
   Controller to a policy and step it one action at a time (or auto-play
   with a delay) to watch exactly what it does -- same render/observation
   panels either way. Every action calls `env.step()` once and persists one
   `Transition`, whether it came from a human or a policy -- see
   `core/interaction.InteractionSession`, the single funnel both paths go
   through, and `ui/state.py`'s `step_play_policy` for the policy path (an
   isolated `PolicyRunner`, one action at a time, falling back to a random
   action and recording a `PolicyExecutionError` on any failure -- identical
   handling to a background Run, just observable step-by-step). Switching
   the controller -- to a different policy, or between human and a policy --
   takes effect immediately, on the very next step, even in the middle of
   an episode; no Reset required. This works without mislabeling provenance
   because `InteractionSession.step()` accepts a per-call actor override, so
   every transition still records exactly who produced it regardless of
   what came before; the moment an episode has been driven by more than one
   controller, it's marked `actor_type="mixed"` so the Episode Browser never
   implies a single controller acted throughout (the per-transition record
   is always the accurate source of truth).
2. **Episodes** -- browse recorded episodes (multi-select whole episodes
   right from the list, or open one to tag/annotate at the episode or
   step level and select individual transitions/step ranges) into the
   **current Evidence Basket**. Episodes are deletable too, singly or in
   bulk (`ExperienceStore.delete_episode`, with a confirmation dialog) --
   this also cleans up their tags/notes and any evidence-basket items
   pointing at them.
3. **Evidence** -- a session can hold *multiple* named baskets (e.g.
   "failures only", "human demos"), not just one. Switch which basket is
   "current" (the one Play/Episodes add to), create/rename/delete named
   baskets, and review/clear whichever one is selected. Deliberately
   independent of Prompt Studio: deciding *what* evidence matters is a
   separate decision from deciding *what to ask the LLM to do with it*.
4. **Templates** -- create/edit named `PromptTemplate`s (this is where new
   ones are made -- Prompt Studio only *picks* among existing ones and edits
   are call-scoped there, never saved) and see every placeholder a template
   can use, with a description of how each is filled. Existing templates
   are laid out in a multi-column grid; each card shows its system/user
   text directly (no separate "Edit" click needed) with its own "Save
   changes" button, which always creates a new version -- nothing is ever
   overwritten. A library of 13 built-in templates (Analyze Experiment,
   Update Environment Knowledge, Explain Reward, Infer Dynamics, Find
   Patterns, Generate Hypotheses, Critique a Hypothesis, Generate a Policy,
   Generate a Policy Directly from Experience, Improve an Existing Policy,
   Critique a Policy Without Rewriting It, Summarize a Trajectory, Select
   Important Transitions) is seeded once per database, globally (available
   in every session), the first time this page or Prompt Studio loads
   (`core.prompts.ensure_builtin_templates`) -- idempotent and
   non-destructive, so editing one of them just creates your own version
   without ever being silently reset. Every built-in's *system* prompt ends
   with `{{environment_description}}`/`{{observation_space}}`/
   `{{action_space}}` (`core.prompts._with_env_context`), so the LLM always
   has grounding info about the environment's interface regardless of which
   narrow, single-purpose template is in use. This page also hosts the
   session-wide **Environment context** editor for those three placeholders
   -- pre-filled with a deliberately generic, environment-agnostic
   description by default (`core.prompts.default_environment_description`/
   `default_observation_space_description`/`default_action_space_description`):
   the environment description and observation space never reveal
   env-specific mechanics or what grid-cell values mean (that's left for the
   researcher/LLM to discover through interaction), while the action space
   does include the real action legend (e.g. `0=up, 1=down, ...`) since
   knowing what a discrete action *does* is needed just to control the agent
   at all -- not the same kind of "answer" as cell semantics or reward
   rules. Fully rewritable here regardless, and every Prompt Studio call in
   the session picks up whatever is currently saved (`core.prompts.
   resolve_environment_context`, stored on `LabSession.metadata`).
5. **Prompt Studio** -- pick a template, then assign a basket to the
   `{{transitions}}` evidence placeholder if that template's system/user
   text actually references it (shown only when it's used). A placeholder
   left on "(none)" simply renders empty. There is no separate
   `execution_error` placeholder: whenever a policy step in Play/Runs/
   Evaluations actually errors (raises, times out, or returns an invalid
   action), the fallback transition that results is auto-tagged
   `execution-error` (so it's findable via the Episodes tag filter, same as
   any other evidence) and its `metadata["execution_error"]` is rendered
   inline automatically by `TransitionFormatter` wherever that transition
   shows up as evidence -- in `{{transitions}}` -- with no separate basket
   to assign. `{{environment_description}}`/
   `{{observation_space}}`/`{{action_space}}` are session-wide, not edited
   here -- see the Templates page's **Environment context** editor. Also
   pick a parent policy, a **Model** (populated from `llm_models.json` -- see "Installation"; falls
   back to the launch default if that file isn't set up), and add notes,
   render the exact prompt that will be sent, then
   either **Call LLM and generate policy** or **Get feedback (no policy
   generated)** -- same template/evidence/model either way; the latter just
   sends the prompt and shows the model's raw answer (a critique, a
   suggestion, an explanation) without attempting to parse it as code or
   create a `Policy` from it. Both are persisted as an `LLMCall`
   (`metadata["call_kind"]` distinguishes them, also shown as a "Kind"
   column on the LLM Calls page), so a feedback exchange is just as
   inspectable/reproducible as a policy-generating one -- it just doesn't
   produce a policy. Editing the system/user text here is call-scoped only
   -- it is *never* saved as a new template version (the exact edited text
   still ends up on the `LLMCall`, so reproducibility isn't lost); the
   template itself only changes via an explicit edit on the Templates page.
6. **Policies** -- inspect source, provenance (parent policy, generating
   LLM call + **which LLM model generated it**, lineage), and act: **Run**,
   **Use as parent**, **Send to Prompt Studio**, **Duplicate/Fork**. Also
   where a researcher writes a policy **by hand**: a "Write a policy by
   hand" form (name, optional tag/description/parent) goes through
   `PolicyStore.create` -- the exact same storage/validation/execution path
   as an LLM-generated policy, just with `llm_call_id=None` recording that a
   human wrote it directly. A hand-written policy is fully interchangeable
   with an LLM-generated one everywhere: Run it, Evaluate it, use it as a
   parent for the next LLM generation, or drive Play with it -- nothing
   downstream cares where the source code came from.
7. **Runs vs. Evaluations -- what's the difference?** Both execute a policy
   online in an isolated subprocess and land every resulting transition in
   the same `ExperienceStore` (tagged `actor_type="policy"`), but they
   answer different questions:
   - **Run** -- "go execute this policy for N episodes/steps and see what
     happens." Exploratory: pick episodes/steps/seeds ad hoc each time,
     just to see how a policy behaves right now.
   - **Evaluation** -- "run *this exact* policy with *this exact, frozen*
     seed set + episode/step budget, and store the aggregate results
     (mean return, success rate, mean episode length, error count)." The
     configuration never changes once created, so two policies' evaluations
     are actually comparable apples-to-apples -- a Run's ad hoc config isn't
     guaranteed to be. Internally an Evaluation just runs a Run with that
     frozen config and aggregates the outcome; the distinction is about
     reproducible comparison, not a different execution mechanism.

   In short: use **Runs** to explore a policy, use **Evaluations** to
   compare policies. The Evaluations page also has a basic side-by-side
   policy comparison (latest evaluation stats + a source diff).
8. **Train** -- automates the generate/run/improve loop above instead of
   doing it by hand. The first iteration always calls **Generate a Policy**
   (no parent policy yet). Two independent dropdowns configure every
   iteration after that (`core/training.py`'s `TrainConfig`): **Search
   method** and **Edge type** are separate axes, not one combined "method"
   name -- any pairing of the two is valid (e.g. Direct + Greedy,
   Critique-Guided + Hill Climbing).
   - **Search method** -- three peers, not a modifier on top of one
     another (**Greedy** is not "Hill Climbing without comparison" as a
     sub-option -- it's its own top-level choice):
     - **Greedy** -- every generated candidate is unconditionally accepted
       as the new current policy, whether or not it actually performed
       better -- a continuously-refining single chain.
     - **Hill Climbing** -- each candidate is run for one evaluation, then
       compared against the *current* policy on **average reward per
       step** (`total_reward / num_steps` over the candidate's own run --
       e.g. -50 return over 50 steps is -1/step, which loses to 0 return
       over 20 steps, 0/step). The candidate is accepted (becomes the new
       current policy) only if its metric is `>=` the current policy's;
       otherwise it's rejected and the current policy stays unchanged for
       the next iteration. The very first-ever policy has no parent to
       compare against, so it's always accepted. A rejected candidate's
       Policy/Run/LLMCall rows are still fully persisted (nothing is
       hidden), just not carried forward as the chain's parent, and its
       run's steps/episodes still count against the total budget. The
       *next* iteration's prompt gets an appended note naming both metrics
       and asking for "a meaningfully different change, not a minor
       variation of the rejected one" -- so the loop doesn't keep retrying
       near-identical rejected candidates.
     - **MCTS** -- an actual tree search, described further below.

     Greedy and Hill Climbing both build one chain of Policy rows via the
     same `run_training_loop` (only their accept/reject rule differs;
     Greedy always advances, Hill Climbing's chain only advances on
     accepted candidates -- rejected ones branch off but aren't extended).
   - **Edge type** -- how the next candidate is *generated*, shared across
     all three search methods:
     - **Direct** -- feeds that iteration's Run's transitions straight into
       **Improve an Existing Policy** as `{{transitions}}`, with
       `{{current_policy}}` set to the previous iteration's policy.
     - **Critique-Guided** -- first asks **Critique a Policy Without
       Rewriting It** (given the previous policy + that run's transitions)
       for a critique, then asks **Improve an Existing Policy** to act on
       *that critique* instead: `{{transitions}}` is left empty and the
       critique's text fills `{{custom_notes}}` ("suggestions") instead.
       The critique call is not repeated on a retry -- only the final
       improvement call is, reusing the same critique.

   A shared **evaluation budget** configures every search method on this
   page -- Greedy/Hill Climbing above and MCTS below alike -- since all
   three are, at bottom, "generate a candidate, then *evaluate* it" in a
   loop: an **Evaluation budget unit** (steps or episodes), an
   **Evaluation amount** (how much of that budget *one* evaluation runs
   for, always identical across evaluations), and a **Total budget** (the
   whole search's cap -- the sum of every evaluation's own amount, checked
   only *between* evaluations, never truncating one mid-way -- so an
   evaluation amount of 10 episodes and a total budget of 10 means exactly
   one evaluation and the search is already done). "One evaluation" is
   method-specific: Greedy evaluates its latest-generated candidate, Hill
   Climbing evaluates its latest candidate (whether or not it goes on to
   be accepted), and MCTS evaluates whichever node its selection step
   lands on (a brand-new child it just expanded, or an existing node
   chosen for re-evaluation). This is distinct from **Max steps/episode**,
   a per-*episode* safety cap independent of the evaluation budget. Seeds
   are fresh/random every evaluation (not fixed), and if the LLM's
   response doesn't produce a valid policy, the *same* generation step is
   retried (up to **Max attempts/iteration**) with the previous error
   appended to the prompt so the model can see what went wrong and correct
   it, rather than silently continuing with a broken policy (and, for
   MCTS, a generation failure that exhausts every retry ends the whole
   search, the same as Hill Climbing) -- **Step timeout** separately
   bounds how long the running policy itself may take per step (same
   mechanism Play/Runs use, auto-restarting the isolated subprocess on a
   hang). Optional starting "initial knowledge" seeds the very first
   generation's `{{custom_notes}}`; an optional **Model** picks from
   `llm_models.json`, same as Prompt Studio. Greedy and Hill Climbing need
   no parameters beyond these shared ones -- only MCTS has its own
   additional settings (below).

   While it runs you get a **live view** -- four cards side by side (Live
   environment, Episode status, Training progress, Currently running
   policy) so everything is visible at once, no scrolling between them:
   - **Live environment**: the environment's current render.
   - **Episode status**: index/step/cumulative return/terminated/truncated
     (same fields as Play's), plus **Proposed action** vs. **Action
     taken**, which differ exactly when the policy's proposal errored/was
     invalid and a random fallback got substituted instead, and a
     **"Last step's execution error"** line, same as Play's -- all sourced
     from the `metadata.proposed_action`/`metadata.execution_error` that
     `core.runs.RunManager.run_policy` already records on the transition.
   - **Training progress**: the running step/episode count against both
     budgets, and reward so far.
   - **Currently running policy**: the actual source of the policy
     currently executing, shown the moment it's generated (`on_policy_ready`
     fires right after a valid policy is produced but *before* it starts
     running) -- not just after an iteration finishes.

   Below that, "This training run's program tree" renders the run as an
   actual tree of `ProgramNode`s (`core/program_tree.py`), not a flat
   iteration-ordered list -- because Hill Climbing's rejections mean the
   chain isn't always linear. Each node is one iteration's policy: its
   source code (click to expand), its avg reward/step metric (`n` = steps
   that run actually ran for), a badge naming the edge from its parent
   ("direct improve" or "critique -> improve", plus the critique text
   itself for the latter), and a collapsible unified diff against its
   *actual parent* (not just "the previous iteration" -- these can differ
   once a rejection has happened). A rejected candidate's card is visibly
   flagged ("REJECTED -- parent kept", red-tinted) and, since it never
   became anyone's parent, has no children -- while the next iteration's
   candidate branches off the same still-current parent alongside it,
   rendered as a sibling one level in. Direct/Critique-Guided Greedy (and
   every accepted Hill Climbing iteration) never branch, so those runs
   just look like a plain vertical chain -- the tree shape only becomes
   visible when there's something to show.

   `ProgramNode`/`build_program_tree` are a read-only reconstruction --
   `parent`/`children` mirror `Policy.parent_policy_id` exactly, no new
   table or schema change -- built purely from already-persisted
   Policy/Run/LLMCall rows tagged with a shared `train_run_id` +
   `train_iteration` in each one's `metadata` (Hill Climbing also tags
   `accepted`). The same `build_program_tree` call (and the same rendering
   function) backs both this live view and the "View a past training run"
   section below, so a still-running tree and a long-finished one render
   identically -- and, being general (one node per generated program, an
   explicit edge type, parent/children links), it's meant to be reusable
   later for actual program tree search, not just this display.

   A **delay between steps** field (like Play's "Auto-play" delay) paces
   the loop so it's actually watchable instead of finishing an episode in
   a fraction of a second -- set it to `0` to run at full speed with no
   pacing. The loop itself is a background task tied to the open Train tab
   -- navigating away or reloading stops it -- but every Policy/Run/LLMCall
   it produces is saved through the exact same mechanisms as anything else
   in this app, so a training run's whole tree stays fully inspectable
   (via `list_training_run_ids`/`get_training_run_policies` under the
   hood) even long after the live view that ran it is gone.

   Selecting **MCTS** as the Search method (`core/mcts.py`'s
   `run_mcts_search`) runs an actual tree search over programs, not a
   chain -- selection/progressive-widening decide what stays in the tree,
   a structurally different mechanism from Greedy/Hill Climbing's
   accept-or-reject rule. It still reuses their machinery throughout:
   candidate generation (`core.training.generate_candidate_policy`, the
   same template selection / critique call / validation-retry loop, using
   the exact same **Edge type** selector above -- there's no separate
   MCTS-only edge type setting), evaluation (`RunManager.run_policy`, using
   the exact same shared **Evaluation budget unit**/**Evaluation amount**/
   **Total budget**/**Max steps per episode**/**Step timeout** fields --
   there's no separate MCTS iteration count either; the search simply
   keeps evaluating, root included, until that same total budget is
   spent), and persistence (every node a real Policy row, every evaluation
   a real Run row, tagged with the same `train_run_id`/`train_iteration`
   convention -- so the *same* program tree view above renders an MCTS
   search's branching tree exactly as it would Hill Climbing's, no
   separate viewer needed). The root is generated and evaluated first
   (logged as "iteration 0"), then each further iteration:
   1. **Selection** -- from the root, repeatedly choose between the
      current node and each child via a UCT-style score (**UCT
      exploration C** controls the explore/exploit balance) until some
      node picks itself; a child's score uses its subtree's best
      performance found so far, the self option uses the node's own.
   2. **Progressive widening** -- `|children| < widening k * n_visits ^
      widening alpha` decides whether the selected node gets a brand new
      child (generated via the edge type above) or is simply evaluated
      again, accumulating more evidence into the same node.
   3. **Evaluation** -- run the node's program for one evaluation budget;
      rewards/steps accumulate across every time a node is evaluated,
      never discarded.
   4. **Backpropagation** -- along the path from root to the evaluated
      node, increment visit counts and recompute each node's best
      discovered subtree performance, bottom-up.

   Every node's live search stats (visit/self-selection counts, its own
   performance, its subtree's best) show on its program-tree card, and a
   full per-iteration log (selection path, expand-vs-reevaluate decision,
   evaluation result) is written to `data/sessions/<id>/exports/
   mcts_<train_run_id>.jsonl` -- one JSON line per iteration. At the end,
   **the returned program is whichever evaluated node scored highest on
   its own performance** -- not the root, not the most-visited node (the
   root sits on every selection path, so it's always at least tied for
   most-visited), not the node with the best subtree score (which can
   belong to a descendant instead).
9. Loop back to **Episodes**/**Evidence** with the new transitions (and any
   `PolicyExecutionError`s, visible on the policy/run detail pages) to
   generate the next policy -- or let **Train** do this loop automatically.

The **Session** page also shows session-wide performance curves --
episode return vs. cumulative environment steps, cumulative LLM tokens, and
elapsed wall-clock time (`core/metrics.py`) -- across every episode in the
session regardless of actor, the same three x-axes `training/trainer.py`'s
`plot_training_metrics` plots for the autonomous training pipeline. It also
lists every session that has ever been created (not just the active one)
with **Load** and **Delete** buttons per row. **Load** switches the active
session right there -- same effect as relaunching with `--session-id`, but
without leaving the app (`app.build_context` + `ui.state.set_context`,
disabled for the row that's already active). **Delete** permanently removes
a session's episodes/transitions/tags/evidence/policies/LLM calls/runs/
evaluations and its on-disk artifact directory (`SessionManager.delete` +
`app.delete_session`; there's a confirmation dialog first, and disabled for
the currently active session -- load a different one first if you want to
delete it). For the *active* session there's a separate **Reset this
session** button instead: it wipes the same content (and artifacts) but
keeps the session row itself (id/name/environment/notes), so you keep
working in the same session, just empty, without needing to relaunch into a
different one (`SessionManager.reset` + `app.reset_session`; also a
confirmation dialog first, irreversible).

## How data is stored

- **SQLite** (`data/database.sqlite`) holds all structured metadata:
  sessions, episodes, transitions, tags/annotations, evidence selections
  (+ items), prompt templates (versioned), LLM calls, policies, runs,
  evaluations, and policy execution errors. See `storage/database.SCHEMA`
  for the exact table definitions and `storage/models.py` for the
  corresponding dataclasses.
- **Filesystem** (`data/sessions/<session-id>/`) holds large/raw
  artifacts: `states/<episode>/<step>_{state,next_state}.json` (numpy
  arrays serialized explicitly via `storage/serialization.py`, not pickled),
  `renders/<episode>/<step>.txt` (the environment's own `render()` text at
  that step), and `policies/<policy-id>.py` (a plain-text copy of every
  policy's source, alongside the copy in SQLite).

## Known limitations

- **Single session, single user.** `ui/state.py` holds one active
  `LabContext` as a process-wide singleton -- this is a local research tool
  launched with one environment/session already chosen on the command
  line, not a multi-tenant server. Reopening a different session means
  relaunching with `--session-id`.
- **Process isolation, not a security sandbox.** `PolicyRunner` isolates
  *crashes, hangs, and runaway loops* (a hard per-step timeout terminates
  the worker process) and restricts *accidental* misuse (no imports, a
  small builtins whitelist -- reused from `agents/
  programmatic_scientist_agent.py`). It is not hardened against a
  deliberately malicious program (no seccomp/namespaces/memory cgroups).
  Treat generated code as you would in that agent's existing training
  loop, not as arbitrary untrusted input from the internet.
- **Timeouts auto-recover, up to a point.** A policy that hangs doesn't
  kill the whole rest of a Run/Evaluation/Play session: `PolicyRunner`
  respawns a fresh subprocess from the same source after each timeout, so
  the next step gets a clean chance (a random action is still substituted,
  and a `PolicyExecutionError` still recorded, for the step that actually
  timed out). Consecutive timeouts are capped (`max_consecutive_restarts`,
  default 3) before the runner gives up and stays not-ready for the rest of
  that run -- so a policy that *always* hangs can't consume unbounded
  wall-clock time on repeated restarts; the counter resets after any
  successful step, so occasional, isolated timeouts in an otherwise-healthy
  policy always get a fresh budget. A runtime exception or invalid action
  never kills the subprocess in the first place (only a timeout does), so
  neither of those triggers a restart.
- **POSIX only.** `PolicyRunner` uses `multiprocessing`'s `"fork"` start
  method rather than `"spawn"` specifically because the app's own
  `__main__.py` needs the NiceGUI-required `__name__ in {"__main__",
  "__mp_main__"}` guard -- with `"spawn"`, every policy-execution subprocess
  would re-execute and re-launch the whole app inside itself. `fork` isn't
  available on Windows; this matches the rest of the repo's Linux/SLURM
  orientation.
- **No environment state checkpointing yet.** `EnvironmentAdapter.
  save_checkpoint`/`restore_checkpoint` are defined but raise
  `NotImplementedError` -- none of the three current environments expose
  state cloning. Adding it later (for "branch a human/policy A/policy B
  continuation from the same saved state") doesn't require changing any
  caller, just implementing those two methods for a specific environment.
- **Policy lineage graph visualization** exists for Train's own runs (the
  program tree above, `core/program_tree.py`) but not yet for arbitrary
  manual forking/lineage outside of Train -- the policy detail page still
  shows `PolicyStore.lineage`/`children` as plain links, no diagram.
- **UI-level tests are smoke-level only** (manual page-load + integration
  script checks during development, not an automated browser test suite);
  the pytest suite covers backend logic thoroughly (see below).

## Tests

```bash
pip install pytest  # if not already installed
pytest programmatic_interactive_lab/tests
```

Covers: state (de)serialization round-tripping, episode/transition
persistence and filtering, tags/annotations (including episode-level ones),
evidence basket resolution (transition/range/episode selection, dedup,
removal, multiple named baskets, rename/delete guards on the default
basket), prompt template versioning and placeholder rendering, explicit
per-placeholder evidence assembly (`build_prompt_context` -- no automatic
success/failure derivation, each placeholder independently either filled
with exactly what's passed in or left empty), policy source validation
(syntax/import/entry-point checks) via this package's own
`execution/sandbox.py`, policy creation/forking/lineage, the isolated
`PolicyRunner` (valid
execution, compile errors, runtime exceptions, timeouts), a full online run
producing episodes/transitions/execution-error records, LLM-call
provenance persistence (mocked client, no network needed), per-session
performance-curve computation (`core/metrics.py`), `EnvironmentAdapter`
behavior (human controls, action validation, LLM-formatted state, space
descriptions) against the repo's real `SimpleGridEnv`/`ObstacleGridEnv`, and
the automated training loop (`core/training.py` -- linear parent chaining,
train-run-id/iteration tagging on Policy/Run/LLMCall metadata and its
reconstruction back into a chain, between-iterations-only budget stopping,
episodes vs. steps budget units, per-step callbacks, the
retry-with-the-previous-error-appended behavior on an invalid/failed
generation including giving up and reporting after the configured max
attempts, both edge types -- direct and critique-guided (the
critique-then-improve-with-empty-transitions behavior) -- that a retry
never repeats the critique call (only the improvement step), and both
acceptance criteria -- Greedy always accepting even a worse candidate,
and Hill Climbing's avg-reward/step accept/reject
comparison, its "parent stays" behavior on rejection, the rejection note
appended to the next attempt's prompt, and rejected candidates still being
fully persisted). `core/program_tree.py`'s `build_program_tree` is covered
separately (`tests/test_program_tree.py`): a linear chain for Direct
Greedy, edge type + critique text for Critique-Guided, and -- the case
that actually needs a tree rather than a list -- a Hill Climbing rejection
producing two children (the rejected node and the next accepted attempt)
on the same parent, plus MCTS's live search stats surfacing correctly onto
the same reconstructed nodes.

`core/mcts.py`'s MCTS search has its own suite (`tests/test_mcts.py`):
pure unit tests of the selection/widening/backpropagation math against
hand-built node trees (self-value accumulating correctly across repeated
evaluations, subtree-value equaling the true max reachable below a node
regardless of which child was just backpropped through, backprop
incrementing visit counts only along the selected path and never a
sibling's, a node with existing children still able to select itself and
grow another one, progressive widening's expand/re-evaluate threshold at
several visit/children-count combinations including the zero-visit
starting case, and -- the two rules most likely to get quietly swapped --
child selection reading a child's subtree value while self-selection reads
the node's own self value, verified in isolation for each), plus
integration tests running a real (mocked-LLM) search end to end for both
the direct and critique edge types (checking every reconstructed node
actually stays reachable from the root -- the regression this caught
during development: a wrong metadata key silently disconnected every
child from the reconstructed tree while the root alone still looked
correct) and confirming the returned program is whichever node scored
highest on its own performance, not the root and not the most-visited node
(the root sits on every selection path, so it's always at least tied for
most-visited, yet must never be favored over an actually-better child).
