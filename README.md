# Helmax

Real-time PPE compliance detection for mining site entrances. A mobile app captures live camera frames, runs them through a YOLOv8 backend, and displays bounding boxes with compliance status in real time.

## How it works

Mobile captures a frame every 500ms and sends it to the FastAPI backend. YOLOv8n runs inference and returns detections with bounding boxes and confidence scores. The app overlays results on the live feed and tracks compliant vs violation counts.

Gallery uploads are also supported. Pick an image and get a full detection result with an annotated summary screen.

## Structure

```
helmax/
    backend/    FastAPI inference server
    mobile/     Expo React Native app
    model/      YOLOv8 weights and training artifacts
```

## Backend

```bash
cd backend
uv sync
uv run fastapi dev main.py
```

Deployed on DigitalOcean App Platform via Docker.

## Mobile

```bash
cd mobile
npm install
cp .env.example .env.local
# set EXPO_PUBLIC_API_URL in .env.local
npx expo start
```

### Build

```bash
eas build --profile preview --platform android
```

### Environment

| Variable | Description |
|---|---|
| EXPO_PUBLIC_API_URL | Backend base URL |

## Detection Classes

Compliant: Hardhat, Safety Vest

Violations: No Hardhat, No Safety Vest
