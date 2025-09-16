# Eye & Head Controlled Mouse (Prototype)

## Overview
A Python-based prototype that enables hands-free mouse control using head movements and blink detection. Designed for accessibility experiments and human-computer interaction studies.  

### Features
- **Head tracking:** Move the cursor by turning your head.  
- **Blink detection:** Click by closing your left eye briefly.  
- **Toggle control:** Press `h` to enable or disable head movement control.
- **Sensitivity Slider:** Update mouse sensitivity using an on-screen slider.  

## Technologies
- **Languages & Libraries:** Python, OpenCV, Mediapipe, PyAutoGUI, pynput, NumPy  
- **Concepts:** Computer vision, facial landmark detection, head pose estimation

## How To Run
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Create and activate a virtual environment:

    Windows: 
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    Mac/Linux: 
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the program:
    ```bash
    python src/main.py
    ```