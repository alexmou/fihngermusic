# Air Theremin Blueprint

## Goal

Build a new instrument on top of the current hand-tracking prototype, but remove fixed frets and discrete guitar-note logic.

The new version should feel closer to a theremin:

- continuous pitch instead of fixed fret positions
- continuous volume instead of pluck-only triggering
- expressive hand articulation instead of string crossing
- conductor-inspired release gestures for natural note endings

## Core Interaction Model

### Hand Roles

- Dominant hand: pitch and fine pitch expression
- Non-dominant hand: volume, note gating, articulation, and stop/release

This mirrors the theremin principle of splitting pitch and volume across two hands.

## Gesture Design

### 1. Pitch Hand

Main control:

- horizontal position controls coarse pitch
- finger openness controls micro-adjustment
- wrist oscillation controls vibrato

Recommended mapping:

- `x` position in the pitch zone maps continuously to frequency
- thumb-index distance adds fine pitch offset
- wrist angular velocity adds vibrato depth when intentional

Pitch hand states:

- neutral: open relaxed hand, stable pitch
- precision: thumb and index closer together for fine tuning
- expressive: controlled wrist oscillation for vibrato
- glide: whole-hand movement for portamento

### 2. Volume Hand

Main control:

- height controls loudness
- palm openness controls brightness or softness
- conductor-style release gesture stops the sound

Recommended mapping:

- hand height maps to amplitude
- palm openness maps to filter brightness or bow-like pressure
- short downward release with a clear stop point mutes the tone

Volume hand states:

- sustain: hand held open and stable
- crescendo: smooth upward rise
- decrescendo: smooth downward fall
- mute: release gesture with a clear endpoint
- hold: suspended still hand for fermata-like sustain

## Conductor-Inspired Stop Gestures

We should not use a harsh "fist punch" mute as the default stop gesture.

Instead, use a release model inspired by conductors:

- soft cutoff:
  open palm, slight circular motion, then a small downward stop
- precise cutoff:
  compact wrist-led motion with a visible endpoint
- fermata release:
  hold the hand still, then perform the cutoff only when the note should end
- comma release:
  stop and reopen quickly for short phrase separation
- full-stop release:
  stop fully, wait a beat, then reopen for the next phrase

## Finger and Wrist Patterns

### Pitch Hand Patterns

- open hand: stable lyrical line
- thumb-index pinch narrowing: fine pitch correction
- gradual finger closing: approach target note
- small wrist oscillation: vibrato
- fast wrist shake should be ignored as noise unless sustained

### Volume Hand Patterns

- open relaxed palm: natural sustain
- fingers soft-close while hand descends: natural release
- palm rising with opening fingers: stronger entrance
- flat still palm: fermata hold
- clenched fist should be treated as an optional hard-cut effect, not the default musical stop

## Anti-False-Trigger Rules

- require gesture stability for 80-150 ms before mode switching
- separate expressive modulation from control switching
- ignore rapid accidental finger jitter
- smooth all continuous controls with low-pass filtering
- freeze pitch briefly during mute gesture so note endings sound intentional

## Sound Design Direction

The instrument should not sound like a raw sine wave.

Recommended sound character:

- voice-like or bowed-string-like base tone
- continuous sustain
- expressive vibrato
- soft attack options
- conductor-like release envelope
- low-pass filter tied to palm openness or speed

Suggested synthesis stack:

- base oscillator: sine + soft saw blend
- secondary harmonic layer for vocal presence
- slow envelope for legato
- faster envelope for articulated entries
- release envelope shaped by conductor-style mute gesture

## Tracking Strategy

Use MediaPipe landmarks already present in the project.

Track:

- wrist
- index fingertip
- thumb tip
- middle fingertip
- palm center
- hand openness
- wrist motion energy

Derived features:

- hand centroid
- thumb-index distance
- finger spread
- palm normal estimate
- vertical and horizontal velocity
- gesture hold duration

## Gesture State Machine

### Pitch Hand

- `idle`
- `tracking_pitch`
- `fine_tune`
- `vibrato`
- `glide`

### Volume Hand

- `silent`
- `attack`
- `sustain`
- `release`
- `hold`
- `retrigger`

## Recommended MVP

Phase 1:

- continuous pitch from pitch-hand horizontal position
- continuous volume from volume-hand height
- release gesture for mute
- vibrato from wrist motion

Phase 2:

- fine pitch control from thumb-index distance
- timbre control from palm openness
- fermata hold mode
- phrase articulation modes

Phase 3:

- user calibration
- scale snapping as optional assist only
- visual pitch guide
- record and playback

## Implementation Plan

1. Fork the current audio engine into a continuous-voice synthesizer.
2. Remove string and fret logic from gesture mapping.
3. Add per-hand feature extraction helpers.
4. Build gesture-state detection for pitch and volume hands.
5. Add conductor-style release detection and envelope shaping.
6. Add calibration for hand range and comfort zone.
7. Add optional overlays for pitch lane, volume lane, and release feedback.

## What To Build First

The best first playable version is:

- one continuous monophonic voice
- right hand controls pitch
- left hand controls volume
- left-hand open-palm downward release mutes
- right-hand wrist vibrato adds expression

That version is simple, musical, and much more promising than trying to classify many symbolic hand signs too early.
