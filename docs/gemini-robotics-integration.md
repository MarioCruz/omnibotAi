# Gemini Robotics-ER Integration: Smart Escalation for OmniAI

*When YOLO can't find it, ask Gemini to reason about where to look*

---

## The Problem

Right now, when a kid says "find the kitchen" or "go find my teddy bear" and YOLO doesn't see the target in the current frame, the system has two options:

1. **LLM (Groq) guesses** — but it only gets text descriptions of what YOLO *did* detect. It has no visual understanding of the scene. It might say "search left" but it's basically guessing.
2. **Rule-based search pattern** — the robot runs a canned `search` or `patrol` pattern, rotating and moving forward blindly until YOLO happens to spot the target.

Neither approach reasons about what the robot is actually *seeing*. A hallway, a door, a kitchen counter — these visual cues could tell you which way to go, but YOLO only knows its 80 COCO classes and Groq never sees the image.

## The Idea

Add **Gemini Robotics-ER 1.6** as an escalation layer. The normal fast loop stays exactly as-is. Gemini only gets called when:

1. A task/mission is active (kid asked the robot to find something)
2. YOLO doesn't see the target in the current frame
3. The robot needs to decide where to explore next

Gemini receives the **actual camera frame** and reasons about the scene visually — "I see a hallway with a door on the left that looks like it leads to a kitchen" — then generates navigation commands.

It can also use **Google Search as a built-in tool** for questions like "find something healthy to eat" (search what's healthy, then look for those items).

---

## Architecture: Where Gemini Fits

```
                         Kid says: "Find the cat"
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │     IMX500 / YOLO         │
                    │   (every 500ms, 30fps)    │
                    │   80 COCO class detection  │
                    └─────────┬────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Target detected?  │
                    └─────────┬─────────┘
                         ╱         ╲
                       YES          NO
                       ╱              ╲
              ┌───────▼───────┐  ┌────▼──────────────────┐
              │  Groq (fast)  │  │  Gemini Robotics-ER   │
              │  ~100ms       │  │  (escalation)         │
              │  Text only    │  │  Sends camera frame   │
              │  "approach    │  │  + task context        │
              │   the cat"    │  │  "I don't see a cat.  │
              └───────┬───────┘  │   What's in this      │
                      │          │   scene? Where should  │
                      │          │   I explore?"          │
                      │          └────┬──────────────────┘
                      │               │
                      ▼               ▼
              ┌──────────────────────────────┐
              │   Robot Executor              │
              │   Audio tones + speech        │
              └──────────────────────────────┘
```

**Nothing changes in the existing fast loop.** Gemini is a new parallel path that only activates when the target isn't found.

---

## New Module: `gemini_reasoner.py`

A standalone module, similar in structure to `llm_command_generator.py`. Does not replace it — works alongside it.

### Class: `GeminiReasoner`

```python
class GeminiReasoner:
    """
    Escalation reasoning using Gemini Robotics-ER 1.6.
    Called when YOLO can't find the target object.
    Sends camera frames for visual scene understanding.
    """
```

### Responsibilities

| Method | What it does |
|--------|-------------|
| `reason_about_scene(frame, task, yolo_detections)` | Main entry point. Sends frame + context to Gemini, returns navigation commands |
| `find_object_in_scene(frame, target_name)` | Ask Gemini to look for a specific object that YOLO missed (Gemini can recognize far more than 80 classes) |
| `suggest_direction(frame, task)` | Ask Gemini which direction to explore based on visual scene cues |
| `plan_search_strategy(frame, task, history)` | Multi-step planning — where to go, what to look for, when to give up |

### What Gets Sent to Gemini

Each call sends:
- **The camera frame** as a JPEG image (640x480, ~30-50KB)
- **A text prompt** with the task, what YOLO detected (or didn't), and what the robot has tried so far
- **Thinking budget** tuned per call type (low for simple "which direction", higher for planning)

### What Comes Back

Gemini returns structured JSON:
```json
{
  "reasoning": "I see a hallway with two doors. The left door shows what looks like a tiled floor, suggesting a bathroom or kitchen. The right door leads to a darker room.",
  "commands": ["left", "forward", "forward"],
  "confidence": 0.7,
  "speak": "I think the kitchen might be to the left, let me check!"
}
```

The commands feed directly into the existing `RobotCommandExecutor`. The `speak` field is optional — gives the robot personality while searching.

---

## Trigger Logic in `dashboard.py`

The escalation happens inside `process_loop()`. Here's the decision flow:

```
Current behavior (unchanged):
  1. Get frame from camera
  2. Run YOLO detection
  3. If task is set AND detections exist → send to Groq → execute commands

New addition:
  3b. If task is set AND target NOT in detections → escalate to Gemini
      - Only if enough time has passed since last Gemini call (cooldown)
      - Send frame + task + "what YOLO sees" to Gemini
      - Execute Gemini's navigation commands
      - Show Gemini's reasoning in dashboard
```

### Cooldown / Rate Limiting

Gemini calls are expensive and slower than Groq. We don't call it every 500ms loop cycle.

| Setting | Value | Why |
|---------|-------|-----|
| `gemini_cooldown` | 5 seconds | Don't spam the API — give the robot time to move and get a new view |
| `gemini_max_retries` | 5 | After 5 Gemini calls without finding the target, switch to patrol pattern and announce "I can't find it" |
| `gemini_thinking_budget` | 1024 tokens (direction), 4096 tokens (planning) | Balance latency vs reasoning depth |

### Target Detection Logic

How do we know if the target is "found" or not? The task string needs to be matched against YOLO labels:

```python
def target_in_detections(task: str, detections: list) -> bool:
    """
    Check if any YOLO detection matches the task target.
    
    Examples:
      task="Find the cat"     → look for 'cat' in detection labels
      task="Go to the kitchen" → no YOLO label for 'kitchen', always escalate
      task="Find my shoes"    → look for 'shoe' in detection labels (not in COCO 80)
    """
```

Some targets will **always** escalate because YOLO's 80 classes don't include them:
- Rooms: kitchen, bathroom, bedroom, garage
- Specific items: shoes, keys, toys, teddy bear
- Abstract goals: "somewhere warm", "something to drink"

This is exactly where Gemini shines — it can visually recognize things YOLO can't.

---

## Gemini API Integration Details

### Authentication

- Requires a **Google AI API key** (free tier available at [Google AI Studio](https://aistudio.google.com))
- Stored in `.env` as `GEMINI_API_KEY`
- Added to `config.json` as an option to enable/disable

### SDK

Uses the `google-genai` Python SDK:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
```

Added to `requirements.txt`:
```
google-genai
```

### API Call Structure

```python
def reason_about_scene(self, frame, task, yolo_detections):
    # Encode frame as JPEG bytes
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image_bytes = buffer.tobytes()
    
    # Build prompt
    yolo_summary = format_detections(yolo_detections)  # what YOLO did see
    prompt = f"""You are controlling a Tomy Omnibot robot on wheels.
    
Task: {task}
What YOLO detected (80 COCO classes only): {yolo_summary if yolo_summary else 'Nothing detected'}
The target was NOT found by YOLO in this frame.

Look at the camera image. Based on what you see:
1. Describe the scene briefly
2. Suggest which direction to move to find the target
3. Return commands the robot can execute

Available commands: forward, backward, left, right, stop, search, patrol

Respond with JSON:
{{"reasoning": "...", "commands": ["cmd1", "cmd2"], "speak": "optional thing to say"}}"""

    response = self.client.models.generate_content(
        model="gemini-robotics-er-1.6-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=1024)
        )
    )
    
    return self._parse_response(response.text)
```

### Google Search Integration

For abstract tasks, Gemini can use its built-in Google Search tool:

```python
# "Find something healthy to eat"
response = self.client.models.generate_content(
    model="gemini-robotics-er-1.6-preview",
    contents=[image_part, prompt],
    config=types.GenerateContentConfig(
        temperature=0.5,
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
)
```

This lets the robot look up information mid-task — local recycling rules, what a specific toy looks like, etc.

---

## Config Changes

### `config.json` additions

```json
{
  "gemini_enabled": true,
  "gemini_model": "gemini-robotics-er-1.6-preview",
  "gemini_cooldown": 5,
  "gemini_max_retries": 5,
  "gemini_thinking_budget": 1024
}
```

### `.env` addition

```
GEMINI_API_KEY=your-key-here
```

---

## Dashboard UI Changes

### Gemini Status Indicator

Add a small indicator to the dashboard showing when Gemini is being consulted:

- 🟢 **YOLO** — target found, using normal Groq loop
- 🟡 **Gemini Thinking...** — escalated to Gemini, waiting for response
- 🔴 **Searching** — Gemini responded, robot is exploring
- ⚪ **Idle** — no task set

### Gemini Debug Panel

Similar to the existing LLM debug panel, show:
- The prompt sent to Gemini
- Gemini's reasoning text (the "I see a hallway..." part)
- Commands generated
- Latency of the call
- Number of escalations so far in this mission

### Kids Dashboard

On the kids dashboard (`/kids`), show a simpler version:
- Robot speech bubble: "Hmm, I don't see a cat... Let me look around!"
- Robot speech bubble: "I think the kitchen is this way!"
- Progress indicator: "Looking for: cat 🔍 (checked 3 directions)"

---

## Cost Estimate

Based on the escalation pattern (not every loop cycle):

| Scenario | Gemini Calls | Approx Cost |
|----------|-------------|-------------|
| "Find the cat" — cat is in the next room | 2-3 calls | ~$0.10 |
| "Go to the kitchen" — needs to navigate | 5-8 calls | ~$0.30 |
| "Find my shoes" — extensive search | 10-15 calls | ~$0.50 |
| Full day of play (20 missions) | ~100-150 calls | ~$5-7 |

Compare to running Gemini every 500ms: ~$250/hour. The escalation approach is roughly **1000x cheaper**.

---

## What Gemini Can Do That YOLO + Groq Can't

| Capability | YOLO + Groq (current) | + Gemini Escalation |
|---|---|---|
| Detect 80 COCO objects | ✅ | ✅ |
| Detect rooms, furniture context | ❌ | ✅ Sees "this looks like a kitchen" |
| Detect objects outside COCO 80 | ❌ | ✅ Sees shoes, keys, toys, etc. |
| Reason about spatial layout | ❌ Text descriptions only | ✅ Sees doors, hallways, paths |
| Use Google Search mid-task | ❌ | ✅ "What does a teddy bear look like?" |
| Understand abstract goals | Limited | ✅ "Find somewhere cozy" |
| Generate movement trajectories | ❌ | ✅ Point sequences for navigation |
| Read text in the environment | ❌ | ✅ Signs, labels, room names |

---

## Files Changed / Added

| File | Change |
|------|--------|
| `gemini_reasoner.py` | **NEW** — GeminiReasoner class |
| `dashboard.py` | Modified — add escalation logic in `process_loop()`, add Gemini debug panel, add status indicator |
| `config.json` | Modified — add `gemini_enabled`, `gemini_model`, `gemini_cooldown`, `gemini_max_retries`, `gemini_thinking_budget` |
| `.env` | Modified — add `GEMINI_API_KEY` |
| `requirements.txt` | Modified — add `google-genai` |

### Files NOT Changed

| File | Why |
|------|-----|
| `llm_command_generator.py` | Groq fast loop stays exactly as-is |
| `camera_capture.py` | No changes needed — already provides frames |
| `object_detector.py` | YOLO detection unchanged |
| `robot_executor.py` | Already handles all the commands Gemini would generate |
| `audio_commander.py` | No changes needed |
| `eye_display.py` | Could optionally add a "thinking" expression, but not required |

---

## Sequence Diagram: "Find the Cat"

```
Kid                Dashboard           YOLO/IMX500        Groq           Gemini
 │                    │                    │                │               │
 │  "Find the cat"   │                    │                │               │
 │───────────────────>│                    │                │               │
 │                    │  task = "find cat" │                │               │
 │                    │                    │                │               │
 │                    │  Loop cycle 1      │                │               │
 │                    │───────────────────>│                │               │
 │                    │  detections: [chair, bottle]        │               │
 │                    │<───────────────────│                │               │
 │                    │                    │                │               │
 │                    │  No "cat" in detections             │               │
 │                    │  Escalate!         │                │               │
 │                    │────────────────────────────────────>│               │
 │                    │                    │                │  frame + task │
 │                    │                    │                │──────────────>│
 │                    │                    │                │               │
 │                    │                    │                │  "I see a     │
 │                    │                    │                │   hallway.    │
 │                    │                    │                │   Turn left   │
 │                    │                    │                │   toward the  │
 │                    │                    │                │   bedroom."   │
 │                    │                    │                │<──────────────│
 │                    │  commands: [left, forward]          │               │
 │                    │  speak: "Let me check this way!"   │               │
 │                    │                    │                │               │
 │                    │  Execute commands  │                │               │
 │                    │  (audio tones)     │                │               │
 │                    │                    │                │               │
 │                    │  ... 5 sec cooldown ...             │               │
 │                    │                    │                │               │
 │                    │  Loop cycle N      │                │               │
 │                    │───────────────────>│                │               │
 │                    │  detections: [cat] │                │               │
 │                    │<───────────────────│                │               │
 │                    │                    │                │               │
 │                    │  "cat" found!      │                │               │
 │                    │──────────────────>│                │               │
 │                    │  Groq: "approach"  │                │               │
 │                    │<─────────────────│                │               │
 │                    │                    │                │               │
 │  "Found the cat!" │                    │                │               │
 │<───────────────────│                    │                │               │
```

---

## Open Questions

1. **Eye display during Gemini thinking** — Should we add a new expression like "thinking" (eyes looking up, maybe with a subtle animation) while waiting for Gemini's response? Would give visual feedback that the robot is "figuring it out."

2. **Speech during search** — How chatty should the robot be while searching? Options:
   - Silent search (just moves)
   - Occasional updates ("Hmm, not here... let me try this way")
   - Running commentary ("I see a hallway, I'll go left")

3. **Give-up behavior** — After `gemini_max_retries` (5 by default), what should happen?
   - Announce "I can't find it, can you help me?"
   - Switch to patrol mode and keep trying silently
   - Return to starting position

4. **Multiple Gemini models** — Should we support both ER 1.5 (faster/cheaper) and ER 1.6 (more capable)? Could use 1.5 for simple direction queries and 1.6 for complex planning.

5. **Search history** — Should we track which directions the robot has already looked and pass that to Gemini? ("I already checked left and straight ahead, what about right?") This would prevent going in circles.

6. **Offline fallback** — When there's no internet, Gemini won't work. The current search/patrol patterns would still run. Is that sufficient, or should we add smarter local fallback logic?

---

## Summary

This is a **targeted, low-cost integration** that adds a powerful new capability without touching the existing fast loop. Gemini Robotics-ER acts as a "smart navigator" that only gets called when the robot is stuck — when YOLO can't see the target and the robot needs to reason about its environment visually to decide where to explore next.

The key insight: **use the right tool for each job**.
- IMX500/YOLO → fast, free, real-time object detection (30fps)
- Groq → fast, cheap text reasoning for known objects (~100ms)
- Gemini Robotics-ER → slow, visual scene understanding for unknown targets (escalation only)
