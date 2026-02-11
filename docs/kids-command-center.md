# Building a Kids Command Center for an AI Robot

*How we turned a vintage Tomy Omnibot into an arcade-style experience for children*

---

## The Problem: AI Dashboards Aren't Built for Kids

Most robot control interfaces look like developer tools - sliders, JSON panels, debug logs, and tiny buttons. Great for engineers, terrible for a 7-year-old who just wants to tell a robot to find their shoes.

We needed something different. Something that made a kid feel like they were in a movie, commanding their own robot. Something that looked like it belonged in an arcade, not a lab.

So we built the **Omnibot Command Center**.

---

## The Design: Synthwave Meets Arcade Cabinet

The kids dashboard lives at `/kids` on the robot's web server. Open it on any tablet, phone, or laptop on the same WiFi network, and you're greeted with neon-soaked retro glory.

### The Aesthetic

Every design choice was intentional:

- **Press Start 2P font** - The same pixel font from classic arcade games. Kids instantly recognize it as "game mode"
- **Neon color palette** - Cyan, pink, yellow, and green glowing against deep purple. High contrast, easy to read, impossible to ignore
- **CRT scanlines** - Subtle horizontal lines overlay the entire screen, mimicking a vintage television
- **Animated grid floor** - A perspective grid scrolls infinitely at the bottom, straight out of an 80s music video
- **Corner brackets** - Neon pink brackets in each corner frame the interface like a heads-up display

The result feels less like "controlling a robot" and more like "playing a video game" - which is exactly the point.

### The Camera Feed

The robot's live camera stream sits at the top in a CRT-style frame:

- Dark metallic bezel with realistic highlights and shadows
- "VISUAL FEED" label in glowing green text
- Screen reflection overlay for that glass CRT feel
- Boosted contrast and saturation to make detections pop

When the AI detects objects, bounding boxes and labels appear directly on the stream. Kids can see the robot thinking in real time - "person 87%" floating over someone's head.

---

## Missions: Give the Robot a Job

Instead of technical task descriptions, kids pick from **arcade-style mission buttons**:

| Button | Mission | What Happens |
|--------|---------|--------------|
| **Find Shoe** | "Find and go to the shoe" | Robot searches for shoes, moves toward them |
| **Find Human** | "Find and greet any person you see" | Robot looks for people, says hello |
| **Find Ball** | "Find and approach the sports ball" | Robot hunts for balls |
| **Explore** | "Explore and describe what you see" | Robot wanders, narrates discoveries |
| **Dance** | Triggers dance pattern | Robot does a dance routine |
| **Speak** | Plays "Hello, I am Omnibot" | Robot introduces itself |
| **Quiet** | Speaker off | Silence (parents love this one) |

Each button is a chunky 3D arcade button with:
- Color-coded backgrounds (pink, cyan, yellow, orange, green, red)
- Neon glow strips underneath
- Big emoji icons (no reading required for younger kids)
- Physical press animation - buttons depress when tapped

**The magic**: Tapping a mission auto-starts the robot if it's not already running. One tap from "off" to "searching for your shoes." No configuration, no setup, no "did you remember to click Start first?"

Behind the scenes, the mission text gets sent to a cloud LLM (Llama 3.1 8B via Groq) which interprets camera detections and generates movement commands. But the kid just sees a robot on a mission.

---

## The D-Pad: Manual Control

Below the missions sits a classic directional pad - a 3x3 grid inspired by game controllers:

```
        [ ▲ ]
  [ ◄ ] [STOP] [ ► ]
        [ ▼ ]
```

Each direction sends audio frequency tones to the original Tomy Omnibot motors:
- **Forward**: 1614 Hz
- **Backward**: 2013 Hz
- **Left**: 2208 Hz
- **Right**: 1811 Hz

The STOP button in the center is bright red - unmistakable even for the youngest operators.

The D-pad buttons have the same satisfying 3D press effect as the mission buttons, with cyan glow on the directional arrows. It feels tactile even on a touchscreen.

---

## Status Panel: The LED Dashboard

Between the camera feed and controls sits a status bar designed to look like a hardware control panel:

```
[  POWER  ]    [  STATUS  ]    [  BLUETOOTH  ]
```

Three LED indicators, each with its own color:
- **Power LED** (green) - Glows when robot is activated
- **Status LED** (yellow) - Pulses when processing
- **Bluetooth LED** (blue) - Shows connection to robot's speaker

The LEDs are CSS-only with box-shadow glow effects. When active, they cast colored light like real indicator LEDs. The Bluetooth status polls every 5 seconds so kids can see if the robot is connected.

---

## The Power Button

A full-width button toggles the robot on and off:

```
[ ACTIVATE ROBOT ]    →    [ DEACTIVATE ROBOT ]
```

When off, it's green with an inviting glow. When on, it turns red as a clear "stop everything" option. The button has a 6-pixel 3D border that compresses on press, giving physical feedback.

---

## Current Mission Display

At the bottom, a dark panel with an orange "CURRENT MISSION" label shows what the robot is doing:

```
┌─ CURRENT MISSION ─────────────────┐
│         FIND HUMAN                 │
└────────────────────────────────────┘
```

When idle, it displays "AWAITING ORDERS..." in dim gray. When a mission is active, the text glows green. This gives kids a persistent reminder of what their robot is up to.

---

## Mobile First

The entire interface is responsive. On phones:
- Mission buttons stack to single column
- D-pad buttons shrink to 50px
- Font sizes scale down
- Everything stays tappable with kid-sized fingers

This matters because the most common use case is a kid walking around with a phone, following their robot as it searches for things. The interface needs to work one-handed, at arm's length, while walking.

---

## What Kids Actually Do With It

From real-world testing:

1. **"Find my shoes" is the killer app.** Kids hide their shoes and send the robot to find them. It's hide-and-seek with a robot.

2. **Dance mode is instant entertainment.** The robot does a pre-programmed sequence of turns and moves. Kids dance along.

3. **The camera feed is hypnotic.** Kids love seeing what the robot sees, especially when bounding boxes appear around familiar objects. "It found the cat!"

4. **Manual driving always wins eventually.** After running missions, every kid gravitates to the D-pad to drive the robot around like an RC car. The AI is cool, but steering is cooler.

5. **"Quiet" gets used more than expected.** Usually by parents.

---

## The Technical Stack (That Kids Never See)

Behind the friendly arcade buttons:

- **Flask** serves the dashboard over HTTPS on the local network
- **IMX500 AI Camera** runs YOLOv8 at 30fps for object detection
- **Groq cloud LLM** (Llama 3.1 8B) interprets detections and generates commands
- **Audio tones** sent via Bluetooth control the original Omnibot motors
- **WebSocket** pushes real-time detection updates
- **ST7735S TFT eye** displays emotions (happy when it sees kids, sleepy when idle)

The entire AI pipeline - camera → detection → LLM reasoning → motor commands - runs every 500ms. To a kid, the robot just... does what you asked.

---

## Design Principles

A few things we learned building this:

**1. One tap to action.** If a kid has to tap more than once to make the robot do something, the interface has failed. Every mission button auto-starts the system.

**2. Emoji > text.** The shoe emoji tells you more than "Object Detection Mission: Footwear" ever could. Most 5-year-olds can't read "autonomous navigation" but they can tap a shoe.

**3. Feedback everywhere.** LEDs glow, buttons press, mission text updates, the camera shows detections. Kids need constant confirmation that something is happening.

**4. Make the off switch obvious.** The red STOP button is always visible, always reachable. When a robot is driving toward the cat and the cat is not happy, you need to stop it NOW.

**5. It should look fun.** If the interface looks like homework, kids won't use it. If it looks like a video game, they'll fight over whose turn it is.

---

## What's Next: Voice Commands

The biggest missing piece? Letting kids just *talk* to the robot.

Right now, every interaction goes through the touchscreen. But the natural way a kid communicates with a robot is by talking to it: "Omnibot, find my shoes!" or "Go forward!" or "Dance!"

Voice commands would remove the last barrier between a child and their robot. No screen needed - just speak and the robot listens. Imagine a kid standing in the living room shouting "Omnibot, find the cat!" and watching it roll off on a mission.

The building blocks are already there:
- **Speech-to-text** on the Pi (Whisper or Vosk for offline, or cloud STT for accuracy)
- **The LLM** already understands natural language mission descriptions
- **Wake word detection** ("Hey Omnibot") to avoid false triggers
- **Confirmation feedback** - the robot's eye display and speech can confirm it heard you

This would turn the Command Center from a touchscreen app into a fully hands-free experience. The arcade buttons stay for when kids want to drive manually, but voice becomes the primary way to give missions.

---

## Try It Yourself

The kids dashboard is part of the open-source OmniBot AI project. Point any device on your network to:

```
https://<your-pi-ip>:8080/kids
```

All you need is a Raspberry Pi 5, an IMX500 AI Camera, and a vintage Tomy Omnibot. The future of robotics is retro.

---

*The best technology is the kind that disappears. Kids don't care about LLMs, neural networks, or inference rates. They care that the robot found their shoes.*
