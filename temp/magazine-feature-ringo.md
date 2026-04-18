---
title: "The Robot That Remembered How to Dream"
dek: "A $600 toy from 1984 was supposed to bring us the future. Forty years later, a Miami maker gave it a brain."
byline: "Mario Cruz"
publication: "Mock feature — Make: / Popular Mechanics style"
photos: "Mario Cruz, PLACITECH"
wordcount: ~1,900
---

# THE ROBOT THAT REMEMBERED HOW TO DREAM

**A $600 toy from 1984 was supposed to bring us the future. Forty years later, a Miami maker gave it a brain.**

*By Mario Cruz*

---

On a workbench at Moonlighter FabLab in Miami, a white plastic robot built during Reagan's first term is staring at a laptop. It's a Tomy Omnibot — 1984 vintage, about two feet tall, with tank-style wheels, a serving tray, and the kind of smoked-plastic head-dome that reads as "robot" to anyone who grew up watching *Lost in Space*. Mounted on its head, grafted on like a second brain, is a small 3D-printed tower with a single glowing cyan eye. The eye blinks.

On a monitor next to the bench, a green rectangle hovers over the laptop: `laptop 73%`. The Omnibot leans forward on its drive wheels. The rectangle moves a few pixels to the right. The eye tracks. Through a Bluetooth speaker buried inside the Omnibot's chest, a synthesized voice announces, *"I found it."*

The laptop is mine. I asked the robot to find it from across the room. It did, without being told where to look.

---

## A PROMISE FROM 1984

When Tomy released the Omnibot in 1984, it was marketed as the domestic robot of the future. It could move via remote control, play cassette tapes, record and replay messages, and — in the commercial, at least — bring you a can of soda on its built-in tray. It retailed for about $600 at the time, or around $1,800 today. Several hundred thousand were sold. Most didn't survive the decades. Leaking alkaline batteries ate the electronics. Brittle rubber wheels crumbled. The dream of a robot butler quietly slid back into the toy box.

I found mine on eBay. It arrived in a box of parts, its PCB spotted with the crystalline white bloom that every vintage-electronics restorer learns to recognize on sight: battery acid. The remote control was in the same shape. Several traces on the main board had been eaten clean through. It took hours with isopropyl alcohol, a toothbrush, and a soldering iron to coax the 40-year-old logic back to life.

When power finally went back in, the motors whirred. The head turned. The Omnibot took its first drive in probably thirty years. Ten seconds later the rubber wheels disintegrated into a fine black dust on the shop floor.

It was the happiest I'd been all week.

> *"The plastic shell was remarkably good. Decades of sitting idle had only taken their toll on what moved."*

---

## THE RESTORATION THAT WOULDN'T STOP

The original plan was simple: fix the batteries, print new wheels, clean up the cassette mechanism, call it done. But every restoration project has a moment where the scope quietly doubles. Mine came when I started reading about the original cassette deck — how the robot interpreted audio frequencies recorded on tape as motor commands. 1614 Hz meant forward. 2208 Hz meant left. 1422 Hz was a magic number that toggled an internal relay to route cassette audio to the onboard speaker. It was an entire command language, encoded in sine waves.

Which meant, in principle, I could drive the robot from *anywhere a speaker could play*. Bluetooth, for example.

I soldered a Bluetooth receiver board into the cassette input. I wrote a Raspberry Pi script that synthesized tones on the fly and streamed them. The Omnibot rolled across my kitchen, obeying my laptop. Then I recorded a wet synthesized "Hello, I am Omnibot" through the audio path, and it felt, briefly, like 1984 had never ended.

That would have been a satisfying ending. Except the Raspberry Pi I'd wired up had a spare $70 AI camera attached to it. And I kept wondering what would happen if I let it drive.

---

## MEET RING-O

![Omnibot meets Ring-O. The tiny Pi-powered companion that finally gives the 1984 robot a brain.](../docs/images/omnibot-ringo-hero.jpg)
*Omnibot meets Ring-O. The tiny Pi-powered companion that finally gives the 1984 robot a brain. (Photo: PLACITECH)*

For the physical build I called in PLACITECH, a Miami engineering shop that specializes in the mechanical side of projects like this. I needed an enclosure that could mount a Raspberry Pi 5, a camera, a cooling fan, a tiny OLED display, and a battery — and sit cleanly on top of the Omnibot without disfiguring it. What came back was a small companion robot named **Ring-O**. He has arms, a round body, and a single glowing eye that looks like a Ring doorbell crossed with WALL-E's goldfish.

Ring-O doesn't move under his own power. His job is to see, think, and talk. The Omnibot is still the body. Ring-O is the brain — and, increasingly, the personality. When you walk up to the robot now, you don't make eye contact with the Omnibot's vacant visor. You make eye contact with Ring-O's tracking pupil.

PLACITECH did the soldering and the enclosure. I did the software. What I didn't realize, until the unit came back assembled, was that my software was wrong.

---

## THE LLM THAT ALWAYS SAID "RIGHT"

The first architecture I built was fashionable: a loop that fed camera detections to a cloud-hosted large language model, which would reason about the scene and output a list of commands. The AI camera was fine. It was running YOLO — a neural network trained to recognize 80 everyday objects — directly on its own silicon, at about 30 frames per second, without using any of the Pi's CPU. Detections like `person 87%` and `chair 72%` arrived as structured metadata.

I piped those into Groq, a service that serves Llama-family language models with enviable speed. The prompt was, in essence, *"Here is what the robot sees. Here is the mission. What should it do next?"*

It didn't work. Not a little. Not at all.

No matter where the person was in the frame — dead center, far left, all the way at the right edge — the model's answer was always `{"commands": ["right"]}`. I tried the small model. I tried the big model. Same answer. The robot spun in place. Every decision took about fifteen seconds. By the time it committed to turning right, the person was usually behind it.

I deleted two hundred lines of prompt engineering and replaced them with about fifty lines of grade-school arithmetic:

- If the person's center pixel is left of frame center, turn left.
- If centered, go forward.
- If right, turn right.
- If the person fills more than 60% of the frame, stop — you're close enough.

Instant. Correct. Every time.

> *"LLMs are wonderful at language. They are terrible at 'the number 331 is greater than 224 and less than 416, therefore output forward.' That's what math is for."*

I kept the LLM code in the repo. It's good at a *different* job — turning a list of detection labels into a spoken sentence when a visitor asks, "What do you see?" But for real-time driving? Rules win, and it isn't close.

---

## THE WAR OF THE BOUNDING BOXES

The second great debugging adventure was smaller in scope and much more satisfying. Ring-O could detect objects, but the green boxes that were supposed to draw around them on the live video feed weren't showing up. Or they'd appear in impossible places — floating above a person's head where the keyboard was supposed to be.

The model's coordinate system was a maze of little lies. The intrinsics metadata claimed boxes were non-normalized, but on this particular post-processed YOLO variant they arrived already normalized; dividing them again produced values like 0.001, which rendered as invisible rectangles. The official Python helper for mapping model coordinates to screen coordinates returned numbers like `width: 0, height: 148,800`. The axis-swap flag was inverting every bounding box's X and Y. And none of it compensated for the fact that the camera captured at 640×480 while the model expected 640×640, which meant the ISP was quietly letterboxing the image with 80 pixels of padding top and bottom that every subsequent calculation had to know about.

The fix was a little math:

- Look at the actual values; if they're bigger than 2, normalize.
- Skip the library's axis swap.
- Compensate for the letterbox padding.
- Clamp everything to frame bounds.

Ten lines of code. Green boxes started tracking objects precisely. The keyboard box lived on the keyboard. The person box lived on the person.

---

## WHAT A KID SEES

I have a son. I built the first dashboard — the one with real-time logs and JSON detection panels — for me. I built the second dashboard for him.

The `/kids` route on the robot's local web server opens into a neon arcade: CRT scanlines, an animated synthwave grid, the `Press Start 2P` pixel font that anyone born after 1990 instantly recognizes as "game mode." Big chunky 3D buttons with color-coded borders: **Find Shoe.** **Find Human.** **Find Ball.** **Dance.** Each with an emoji large enough for a five-year-old who can't read yet. A red STOP button the size of a fist.

Tap **Find Shoe** and the robot starts looking for shoes. Tap **Dance** and it launches a 20-step choreography pre-recorded as a sequence of turns and moves. Tap **What See?** and Ring-O uses the LLM to describe the room out loud through its speaker, in under ten words, in something that sounds vaguely like a confused toddler.

The killer app, it turns out, is shoe-finding. Kids hide their shoes. The robot finds them. It's hide-and-seek, except one of the players has a neural network and a laser-printed face.

> *"The best technology is the kind that disappears. Kids don't care about inference rates. They care that the robot found their shoes."*

---

## PERSONALITY, BY DESIGN

Ring-O's eye is a 128×128-pixel color OLED, about the size of a thumb print. It has no effect on navigation. It runs in its own thread and does exactly one thing: act alive.

The eye dilates and smiles when a person is detected. It widens into surprise when a cat or a dog shows up. It tracks left when the robot turns left. Every three to seven seconds, it blinks. After thirty seconds of no activity, the eyelids slowly close and it goes to sleep. When I tap the Omnibot awake, the eye opens — a full animation, not a state change — and looks directly at me.

This is the least-necessary feature in the entire stack. It is also the one that makes the whole thing work. People don't react to a wheeled platform with a camera on it. They react to a thing with an eye. The eye is where the 1984 promise finally gets kept: not because the hardware is smarter than we expected, but because the *character* finally showed up.

---

## MAKING IT SURVIVE A SATURDAY

Making the robot work once is engineering. Making it work for eight hours at Maker Faire Miami, while strangers mash buttons and children point it at uncooperative dogs, is a whole second project.

The dashboard now runs as a `systemd` service with automatic restart. A pre-launch smoke test verifies every import, camera frame, and audio path before the web server will even start. A health endpoint returns HTTP 503 with structured reasons when any subsystem is degraded — stale camera, dropped Bluetooth, frozen eye thread. Bluetooth status is polled in the background so the user interface never blocks. And because kids *will* mash buttons, the robot now serializes commands and tells the UI "busy, try again" when it can't accept a new one, instead of silently dropping it.

None of this is glamorous. All of it is the difference between a bench demo and a robot that still works at 5 p.m. on the second day of a public festival.

---

## THE FUTURE, FORTY YEARS LATE

Tomy's 1984 brochure promised a domestic robot that would learn your schedule, recognize your voice, and bring you a drink. The $600 toy of the Reagan era couldn't deliver any of that. The technology simply wasn't there.

It's here now. A $70 AI camera, a $100 single-board computer, a Bluetooth speaker, a little companion robot named Ring-O, and an open-source stack you can read on GitHub. The original 1984 motor board — the same hand-routed PCB with through-hole components and discrete logic — still interprets the audio tones that tell the robot to roll forward. That part hasn't changed. It didn't need to.

The Omnibot will be on the floor at **Maker Faire Miami 2026**. It will be looking for you. When it finds you, its companion's pupil will dilate, the internal speaker relay will click, and a small synthesized voice will say, *"I found it."*

The dream didn't arrive late. It just needed a buddy to help out.

---

*Mario Cruz is a maker based in Miami. Full source code for the Omnibot / Ring-O stack is on GitHub at [github.com/MarioCruz/omnibotAi](https://github.com/MarioCruz/omnibotAi). His ongoing blog series on the restoration is at mariothemaker.com.*

---

### SIDEBAR — HOW RING-O DECIDES

```
  Camera (IMX500)      →  YOLO11 on-chip  →  [person 73% at x:223]
                                                    │
                                                    ▼
                                  ┌────────────────────────────┐
                                  │  Rule-based navigator       │
                                  │  left?  center?  right?     │
                                  │  fills > 60%?  → STOP       │
                                  └────────────────────────────┘
                                                    │
                                                    ▼
                                       step("forward")
                                                    │
                                                    ▼
                       sox → pw-play → Bluetooth → Omnibot motors
                                 (1614 Hz = forward)
```

### SIDEBAR — THE NUMBERS

| Spec | Value |
|---|---|
| Original Omnibot | Tomy, 1984, ~$600 retail |
| Add-on compute | Raspberry Pi 5, 8 GB RAM |
| Vision | Sony IMX500 AI Camera, YOLO11 nano, ~17 ms inference |
| Object classes | 80 (COCO) |
| Decision latency | ~0 ms (rule-based), down from ~15 s (LLM) |
| Eye display | 128×128 SSD1351 OLED |
| Command bandwidth | 1 audio tone per motion |
| Open source | MIT, github.com/MarioCruz/omnibotAi |
