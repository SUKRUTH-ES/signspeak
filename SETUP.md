# SignSpeak: setup on a fresh Windows laptop

This guide installs everything needed to **run**, **edit** and **retrain** SignSpeak.
It takes about 10 minutes and assumes plain Windows 10/11 with nothing extra installed.

---

## 1. Install the tools

Open **PowerShell** (Start menu → type `PowerShell` → Enter) and run these one at a time.
`winget` comes with Windows 10/11. Accept the prompts it shows.

```powershell
winget install -e --id Python.Python.3.12
```
```powershell
winget install -e --id Microsoft.VisualStudioCode
```
```powershell
winget install -e --id Git.Git
```

| Tool | Why | Required? |
|---|---|---|
| **Python 3.12** | Runs the local web server and the training scripts | Yes |
| **VS Code** | Editing the code | Recommended |
| **Git** | Version control / sharing changes | Optional |
| **Browser** | Microsoft Edge is already installed; Chrome also works | Already there |

> **Close PowerShell and open a new one** after installing, so it picks up the new commands.

Check that it worked:
```powershell
python --version
```
It should print `Python 3.12.x`.

If `winget` isn't available, download Python from https://www.python.org/downloads/ instead.
In the installer, **tick "Add python.exe to PATH"** before you click Install.

---

## 2. Get the project

```powershell
git clone https://github.com/SUKRUTH-ES/signspeak.git
```
```powershell
cd signspeak
```
Open that folder in VS Code (**File → Open Folder**), then open the built-in terminal (**Ctrl + `**).

---

## 3. Install the Python packages

These are only needed to retrain the model, but it's best to set them up now.
Run this in the project folder:

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\Activate.ps1
```
```powershell
pip install -r requirements.txt
```

If activation fails with *"running scripts is disabled on this system"*, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 4. Run the website

From the project folder:

```powershell
python -m http.server 8080
```

Open **http://localhost:8080** in Edge or Chrome and allow camera access when asked.
Press **Ctrl + C** in the terminal to stop the server.

> Opening `index.html` by double-clicking **won't work**. Browsers only allow the camera and
> model loading through `http://localhost`, so always use the server.

**Internet is needed** while using the site: the MediaPipe hand-tracking library and model load from a CDN.

### Using it
1. **Start camera** (button in the header). **Stop camera** turns it off again.
2. Open the **ASL Alphabet** folder → **Load** to fingerspell A–Z with one hand.
3. Open **My Signs** → add a sign → **Record** and hold it for 3 s (record 2–3 times) to teach your own words.
4. Signed letters and words build a sentence and are spoken aloud.

---

## 5. Retrain the ASL model (optional)

| File | What it is | In the repo? |
|---|---|---|
| `data/landmarks_raw.json` | Hand keypoints already extracted from 180 photos per letter | Yes |
| `asl-model.json` | The trained model the website loads | Yes |
| `data/asl.parquet` | The original dataset (Hugging Face `Marxulia/asl_sign_languages_alphabets_v03`, 10,873 photos, A–Z) | No, download it (below) |
| `data/images/` | The photos unpacked into one folder per letter | No, made by `unpack_dataset.py` |

**Just retrain** (uses the extracted keypoints, takes seconds):
```powershell
python tools/train_asl.py
```

**Full rebuild from the photos:**
1. Download the dataset (~71 MB) and unpack the photos:
   ```powershell
   python tools/download_dataset.py
   ```
   ```powershell
   python tools/unpack_dataset.py
   ```
2. Start the builder server:
   ```powershell
   python tools/build_server.py
   ```
3. Open http://localhost:8081/tools/build.html?per=180 and keep the tab in front until it says *Done* (about 15 minutes). Change `per=` to use more or fewer photos per letter.
4. Stop the server (Ctrl + C), then train:
   ```powershell
   python tools/train_asl.py
   ```

The keypoints are extracted in the browser on purpose: it uses the exact MediaPipe model the website uses,
so training data and live webcam input are processed the same way.

---

## 6. Project structure

```
signspeak/
├── index.html        page layout
├── style.css         frosted-glass design
├── app.js            camera, hand tracking, classifiers, sentence + speech
├── features.js       keypoint → feature vectors (shared by app and builder)
├── glass.js          generated liquid artwork + glass folder component
├── ui.js             wires the folders to the app
├── asl-model.json    trained ASL neural network (273 → 128 → 26)
├── requirements.txt  Python packages for training
├── data/             extracted keypoints (dataset downloads here)
└── tools/            dataset unpacking, keypoint extraction, training
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` opens the Microsoft Store or "not recognized" | Reinstall Python with **Add to PATH** ticked, then open a new terminal. Or turn off *Settings → Apps → Advanced app settings → App execution aliases → python.exe* |
| Camera doesn't start | Use `http://localhost:8080` (not a file path), allow camera in the browser's address-bar icon, and close other apps using the webcam (Zoom, Teams, Camera) |
| `Address already in use` | Another server is running. Close it, or use a different port: `python -m http.server 8090` |
| Page stuck on "Loading model…" | Check your internet connection (the MediaPipe library comes from a CDN) |
| Letters are wrong a lot | Good lighting, plain background, whole hand in frame, palm facing the camera; hold each letter still |
