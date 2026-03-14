# Finger Music

Finger Music is a real-time virtual guitar project that turns webcam hand tracking into playable string interactions. It uses computer vision, gesture detection, and procedural audio synthesis to simulate an air-guitar style instrument directly from hand movement.

This README is intentionally written in English to improve GitHub search visibility, broader discoverability, and monetization potential.

## Core Value

- Real-time webcam-controlled virtual guitar
- Hand tracking with fret-hand and pluck-hand separation
- Gesture-based string crossing detection
- Procedural guitar-like audio synthesis
- Lightweight Python prototype for interactive music tech demos

## Why This Project Is Searchable

This repository targets several high-intent search categories:

- virtual guitar
- air guitar
- hand tracking music app
- computer vision music project
- gesture-controlled instrument
- webcam instrument
- Python audio synthesis
- MediaPipe hand tracking demo
- OpenCV music interaction
- real-time interactive audio

## Features

- Tracks hands from a webcam feed in real time
- Separates fretting and plucking hands on screen
- Detects pluck events when fingertips cross strings
- Maps neck position to fret numbers
- Generates synthesized string tones without external sample packs
- Draws an interactive guitar scene with string and fret feedback
- Includes unit tests for core gesture and pluck logic

## Tech Stack

- Python
- OpenCV
- MediaPipe Tasks
- NumPy
- SoundDevice
- Real-time audio synthesis
- Computer vision
- Hand tracking
- Gesture recognition
- Human-computer interaction

## Technology Tags

`python` `opencv` `mediapipe` `numpy` `sounddevice` `computer-vision` `hand-tracking` `gesture-recognition` `audio-synthesis` `real-time-audio` `virtual-instrument` `air-guitar` `music-tech` `interactive-audio` `webcam` `hci`

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3
- Webcam
- `hand_landmarker.task` model file in the project root

## Run

```bash
python3 main.py
```

Show your hands to the camera and move your fingertips across the strings. Press `q` to quit.

## Tests

```bash
python3 -m unittest -v
```

## How It Works

1. MediaPipe detects both hands from the webcam stream.
2. The left side of the screen is treated as the fretting zone.
3. The right side of the screen is treated as the plucking zone.
4. Fingertip movement across string lines triggers pluck events.
5. Fret position changes pitch per string.
6. The audio engine synthesizes a smoother, more natural guitar-like tone in real time.

## Use Cases

- Interactive music-tech demos
- Computer vision portfolio projects
- Gesture control experiments
- AI music interface prototypes
- Educational examples for MediaPipe and OpenCV
- Rapid prototyping for webcam-based instruments

## SEO Keywords

virtual guitar, air guitar, finger music, hand tracking guitar, gesture guitar, webcam guitar, computer vision music, Python music project, MediaPipe hand tracking, OpenCV guitar, interactive audio, real-time sound synthesis, AI instrument, virtual instrument, gesture-controlled music app

## Fast Discovery Checklist

For fastest discovery on GitHub and search engines, make sure the repository also has:

- A strong one-line repository description
- Relevant GitHub Topics
- A custom social preview image
- A public release with notes and screenshots
- A demo GIF or short video in the README
- Consistent project naming across repo title, README, and release title
- Clear setup instructions and keywords matching user search intent

Recommended assets to add next:

- demo GIF of plucking and fretting
- screenshot of the interface
- short product-style tagline image for social sharing
- release notes for version `v0.1.0`

## Monetization Readiness

This repository is positioned well for future monetization paths such as:

- premium app packaging
- paid desktop/mobile adaptation
- API or SDK access
- course or tutorial sales
- sponsorship-driven open-source growth
- creator tools for musicians and streamers

Billing, sponsorship, and payment integration can be connected separately without changing the technical positioning of the project.

## Commercial Licensing

This repository is public for discovery, portfolio visibility, and product validation.

The source code is not released under a permissive open-source license. Public visibility does not grant commercial usage rights, resale rights, redistribution rights, or white-label rights.

Commercial access, product integration, private licensing, and derivative commercial use require a separate written agreement with the repository owner.

See [LICENSE.md](/Users/alexmou/fihngermusic/LICENSE.md) and [COMMERCIALIZATION.md](/Users/alexmou/fihngermusic/COMMERCIALIZATION.md).

## Repository Metadata

Use the ready-to-paste repository metadata from [REPO_METADATA.md](/Users/alexmou/fihngermusic/REPO_METADATA.md) for GitHub Description, Topics, search keywords, and launch checklist.

For product-facing positioning and sales preparation, see [COMMERCIALIZATION.md](/Users/alexmou/fihngermusic/COMMERCIALIZATION.md).
