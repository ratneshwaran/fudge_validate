# Phase-conditioned DAG prompts — DRAFT for review

**Why this exists:** the current `prompts.yaml` (prompts 1–4) is phase-blind. It asks for a
generic *"mental health/therapy grounding"* flow with no mention of which PE phase is being
modelled, so `v1` produces byte-identical prompts for P5/P6/P7 and `v2/v3` only get phase signal
from the Prompt-5 data merge. Result: the generated DAGs don't encode a specific phase, which
breaks the per-phase discrimination design in TODO 8.

**What this draft changes:** inject the phase into prompts 1–4 via template slots, and tighten the
structural rules the models kept violating (acyclicity, connectivity, strict bot/user alternation).
Prompt 5 keeps its data-merge slots and gains phase context.

**Status:** draft only — nothing is wired into `generate_llm_dags.py` yet. The phase descriptions
below are my best reconstruction of the PE protocol from the project notes; they should get a
clinical sanity check (Francesca) before regeneration. Review, edit, then I'll port the agreed
version into `prompts.yaml` + add the `{{phase_*}}` substitution.

---

## Template slots

Filled at runtime, per phase, before the call:

| Slot | Example value |
|---|---|
| `{{phase_id}}` | `P6` |
| `{{phase_name}}` | `SUDS Monitoring` |
| `{{phase_description}}` | the one-paragraph description from the map below |
| `{{phase_moves}}` | optional bullet list of canonical therapist moves for the phase (steer, not constrain) |
| `{{design_guidelines}}` | (existing) Prompt-5 merge guidance |
| `{{thousand_voices_data}}` | (existing) sampled TV training conversations for the phase |

---

## Phase-description map

PE = Prolonged Exposure therapy. Phases below mirror the TV corpus segments. **P5/P6/P7 are the
phases currently generated**; P8/P10/P11 are included for completeness (pending labelling/top-up).

> ⚠️ Clinical-accuracy check needed before use — these are drafted from project notes, not a PE manual.

| `{{phase_id}}` | `{{phase_name}}` | `{{phase_description}}` (draft) |
|---|---|---|
| **P5** | Orientation | The therapist orients the client to the imaginal-exposure procedure: explains what will happen, gives the rationale (repeatedly revisiting the memory in a safe setting reduces distress and enables emotional processing), sets expectations, confirms readiness/consent, and instructs the client on the mechanics (eyes closed, recount in the first person and present tense). Focus is on *setting up* the exposure, not yet doing it. |
| **P6** | SUDS Monitoring | The therapist elicits Subjective Units of Distress Scale (SUDS, 0–100) ratings — a baseline and then periodically *throughout* the exposure — and uses the trajectory to pace the work (continue, slow down, or briefly ground if distress is too high). The defining move is the recurring, explicit distress check-in tied to the ongoing narrative. |
| **P7** | Reinforcing Exposure | The therapist keeps the client engaged with the memory: validates effort, encourages staying with it rather than avoiding, praises engagement, and provides supportive containment so the client tolerates the distress and remains in the exposure. Focus is *sustaining* exposure, not introducing new material. |
| **P8** | Eliciting Thoughts & Feelings | The therapist prompts the client to articulate the thoughts, beliefs, and emotions attached to the memory (e.g. "what went through your mind then?", "what does that say about you now?"), surfacing the cognitions that maintain the distress. |
| **P10** | Full Imaginal Exposure | The complete exposure protocol end to end: start the narrative, prompt to continue, ask for sensory and bodily detail, check SUDS, instruct grounding if distress peaks, loop through the memory repeatedly, then debrief. The richest, most structured phase. |
| **P11** | Processing | Post-exposure discussion: the therapist helps the client reflect on what surfaced, draw new perspectives, make meaning, and consolidate insight — moving from re-experiencing to cognitive integration. |

### Optional `{{phase_moves}}` (P6 example)
```
- greet_and_check_in
- ask_suds_baseline
- prompt_continue_narrative
- ask_suds_midexposure
- respond_to_high_distress (brief grounding / slow down)
- respond_to_decreasing_distress (continue / deepen)
- validate_and_reassure
- wrap_up_and_close
```

---

## Prompt 1 — generate  *(was phase-blind → now phase-conditioned)*

> **Changes:** opening reframed from generic "grounding" to the specific PE phase; added the phase
> description + (optional) canonical moves; kept all structural guidelines verbatim; added an explicit
> "stay within this phase — do not drift into generic support" instruction.

```
You are designing a dialogue flow for Prolonged Exposure (PE) therapy, specifically the
**{{phase_name}}** phase ({{phase_id}}).

Phase focus:
{{phase_description}}

Representative therapist moves in this phase (use as a guide, not a fixed list):
{{phase_moves}}

Design a directed acyclic dialogue flow for THIS PHASE ONLY, suitable for visualization with
mermaid.js. The flow should depict the nuances and potential branches of interactions between a
bot (therapist) and a user (client) as they occur during the {{phase_name}} phase. Stay within
the scope of this phase — do not drift into a generic greet/validate/coping-menu chatbot flow;
the phase-specific therapist actions above must be clearly represented and distinguishable.

Please adhere to the following guidelines:

Nodes Definition: Use distinct nodes to represent the bot ("B") and the user ("U").

High-Level Dialog Action: Each node should encapsulate that segment's core sentiment or function
in the conversation, relevant to the {{phase_name}} phase. It should be a label for the node
representing a high-level dialogue action and not just the dialog.

Flow & Directionality: Create directed connections between nodes to represent the progression of
the conversation. The dialogue should flow from one node to potentially multiple nodes, allowing
for various conversational turns.

Diverse Conversational Possibilities: Ensure that bot nodes can lead to multiple user nodes and
vice versa, accounting for various user responses or bot prompts within the {{phase_name}} phase.

Acyclic Structure: The dialog flow must not have loops or cyclic pathways. If a similar action or
sentiment arises later, introduce a NEW node rather than looping back to an earlier one.

Mermaid.js Compatibility: Ensure the flow adheres to mermaid.js graph notation for seamless
rendering.

Craft a dialogue flow focused on the {{phase_name}} phase of PE therapy. The bot always begins by
greeting/orienting the user appropriately for this phase. The graph must be connected. Bot and user
nodes should be in different colors. A bot node is only followed by user nodes and user nodes only
by bot nodes.
```

---

## Prompt 2 — critique  *(adds phase-fidelity criterion)*

> **Changes:** added a "Phase Fidelity" criterion so the critique pass actively flags drift away from
> the target phase; rest unchanged.

```
Based on the evaluation criteria below, suggest improvements and provide concise, actionable
feedback on the flow just generated for the **{{phase_name}}** ({{phase_id}}) phase:

Phase Fidelity: Does every part of the flow belong to the {{phase_name}} phase? Flag any node that
is generic supportive-chatbot filler (e.g. broad "offer coping menu", "safety check") rather than a
move specific to this phase. Are the defining actions of this phase ({{phase_moves}}) all present
and distinguishable?

Optimality: Check for redundancy. Ensure nodes aren't replicating the same or very similar dialog
actions at different points.

Clarity of High-Level Dialog Action: For every node, is the high-level dialog action clear and
meaningful? Could someone unfamiliar with the domain understand the flow? Avoid vague or overly
complex nodes.

Extensiveness: Does the flow cover the diverse conversational possibilities of this phase? Are all
nodes interconnected? Does it cover the major interactions within the {{phase_name}} phase?

Representativeness:
Bot Nodes (B): Do they represent clear, unambiguous therapist actions — neither too broad nor too
specific?
User Nodes (U): Do they capture an adequate range of client responses relevant to this phase?
```

---

## Prompt 3 — revise  *(adds "stay within phase" constraint)*

> **Changes:** added the phase-scope reminder; kept the structured diff output format verbatim.

```
Taking into account the feedback and the original design guidelines — keep a directed acyclic graph
structure, ensure all new components are labelled and connected, and keep the flow specific to the
**{{phase_name}}** ({{phase_id}}) phase (remove or merge any generic, non-phase-specific nodes the
critique identified) — revise the flow. Ensure your revision addresses the identified improvements
while still adhering to the primary construction rules. Account for all new nodes (including merged
nodes) and their labels/colors. All user nodes must connect to bot nodes, and bot nodes end the
conversation. Give your updates in the format below:

'split_nodes':
# 'NodeToSplit': ['NewNode1', 'NewNode2', ...],

'add_nodes':
# 'NodeToAdd': 'Label',

'remove_nodes':
# 'NodeToRemove1', 'NodeToRemove2', ...

'relable_nodes':
# 'NodeToRelabel': 'NewLabel',

'add_edges':
# ('Start Node', 'End Node'),

'remove_edges':
# ('Start Node', 'End Node'),
```

---

## Prompt 4 — finalize  *(stronger structural enforcement)*

> **Changes:** kept the cleanup intent; made the acyclicity / connectivity / alternation rules
> explicit and non-negotiable (these were the most-violated rules — 136 cycles in one DAG, several
> fragmented graphs); added a final "output one mermaid block" instruction so parsing is reliable.

```
Clean up the flow to produce the FINAL flow for the **{{phase_name}}** ({{phase_id}}) phase. Address
the remaining improvements while adhering to all construction rules. Specifically, the final flow MUST
satisfy ALL of the following — fix any violation before returning:

- Acyclic: no loops or cyclic pathways anywhere. If an action recurs, use a new node.
- Connected: every node is reachable from the start node; no isolated nodes and no disconnected
  fragments.
- Strict alternation: bot nodes are never connected directly to other bot nodes; user nodes are never
  connected directly to other user nodes.
- No hanging user nodes (every user node has an outgoing edge); only bot nodes may end the conversation.
- Every node except the begin/end nodes has both an input and an output.
- No duplicate edges (one node points to another at most once).
- All bot nodes correctly colored; the flow stays specific to the {{phase_name}} phase.

Return ONLY the final flow as a single ```mermaid fenced code block.
```

---

## Prompt 5 — merge  *(adds phase context to the merge)*

> **Changes:** names the phase so the merge is anchored to it; keeps both template slots unchanged.

```
You are given two dialogue flows for the **{{phase_name}}** ({{phase_id}}) phase of PE therapy. One
flow is LLM-generated and the other is derived from real data examples. Merge all unique elements of
the two flows without duplicating similar elements, keeping the result specific to the {{phase_name}}
phase. Merge based on the following design guidelines:

< Design guidelines >
{{design_guidelines}}

< Thousand Voices Data >
{{thousand_voices_data}}

Return the merged flow as a single ```mermaid fenced code block, preserving the directed-acyclic,
connected, strictly bot/user-alternating structure.
```

---

## Notes / open items (not prompts)

1. **Post-parse validity guard (code, not prompt).** Even with stronger prompts, models will
   occasionally emit cyclic/fragmented graphs. Recommend a guard in `generate_llm_dags.py` that, after
   `parse_mermaid_dag`, checks acyclicity + single-component connectivity and either (a) auto-prunes
   (drop isolated nodes, break back-edges) or (b) flags the cell for a reroll. Keeps TODO 8 inputs clean.
2. **`{{phase_moves}}` is optional.** Including it steers the model strongly toward phase-specific
   content but risks the model just transcribing the list. Worth A/B-ing with and without on one cell.
3. **Variant behaviour is unchanged.** v1 = Prompt 1 alone (now phase-conditioned, so v1 finally
   becomes a real per-phase DAG). v2 = 1–5 fused. v3 = 1–5 sequential.
4. **Regeneration cost** is roughly the same as the original TODO 5 run; cached calls for unchanged
   prompts won't re-bill, but since the prompts change, expect mostly fresh calls.
