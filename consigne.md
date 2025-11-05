# Deep Learning Project: TinyGrad MNIST Classifier 🚀

## Project Goal

The objective is to take a deep learning model from training in Python (using **tinygrad**) all the way to a working, interactive web application using **WebGPU**. You will train models to recognize handwritten digits (MNIST), export them, and build a single-page web app where users can draw a digit and get a real-time prediction with visualized confidence.

**To Pass the Project, a final score of $\ge 10$ is expected for each section.**

-----

## 1\. Model Development and Training (Python/tinygrad)

| Weight: 35% (Difficulty: High) |
| :--- |

This section focuses on using Python and the **tinygrad** framework to build, train, and optimize your models.

### 1.1 Model Requirements & Training

1.  **MLP (Multi-Layer Perceptron):** Start with the provided `mnist_mlp.py`.
2.  **CNN (Convolutional Neural Network):** Create a new file (e.g., `mnist_convnet.py`) and implement a simple CNN.
3.  **Export:** Both scripts must use the export logic to generate the WebGPU files (`.js` and `.safetensors`).

**Key Commands for Training & Export:**

To train and export your MLP model for 100 steps with JIT enabled:

```bash
STEPS=100 JIT=1 python mnist_mlp.py
```

*Note: This will create a directory named `mnist_mlp/` containing the exported files.*

### 1.2 Hyperparameter Exploration & Accuracy

  * **Documentation:** Create a markdown file named **`HYPERPARAMETERS.md`**. Log the configurations you tried (e.g., Learning Rate, batch size `BS`, number of layers, optimizer, activation function, ...etc) and the resulting test accuracy.
  * **Experimentation:** Test at least **eight** distinct configurations for *each model* (bonus if you script this exploration).
  * **Accuracy Target:** Aim for the highest possible accuracy (Target: **$\ge 95\%$** for MLP; **$\ge 98\%$** for CNN).

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No models are implemented or exported. |
| **10 (Minimum)** | Two models are implemented and trained. Minimal documentation (only 1-2 attempts logged). Low accuracy ($\le 90\%$). |
| **15 (Good)** | Two models are implemented, documented exploration for both models (**3 settings each**), and final models meet the target accuracy ($\ge 95\%$/$\ge 98\%$). |
| **20 (Very Good)** | Meets "Good" criteria. The `HYPERPARAMETERS.md` shows exceptional analysis of why certain parameters work best (e.g., analyzing convergence speed or loss stability). |

-----

## 2\. Web Application Development: Functionality (HTML/JS)

| Weight: 30% (Difficulty: Medium) |
| :--- |

This section covers the core logic of the web application, ensuring the model runs and the user interaction works.

### 2.1 Core Functionality

1.  **Model Selection:** Dropdown or buttons to easily switch between MLP and CNN models.
2.  **Drawing:** A canvas with working **Pen**, **Eraser**, and **Clear** tools.
3.  **Real-Time Classification:** When drawing stops, the JavaScript must:
      * **Resize** and **Normalize** the image to $28 \times 28$ grayscale.
      * Run the prediction using the WebGPU model.
      * Display the **best guess** and the **inference time** (in ms, target 60 FPS Desktop).

### 2.2 Visualization (Probability Bar Plot)

  * **Mandatory Visualization:** Display the model's confidence ($\text{softmax}$ output) for all 10 digits (0-9) as a **bar chart** immediately after inference.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No functional web page, or model loading/inference fails. |
| **10 (Minimum)** | Single model loadable. **Best guess** prediction is displayed, but the bar chart is missing or non-functional. Only basic drawing works. |
| **15 (Good)** | Meets "Minimum" criteria. Two models are loadable/switchable. **Probability bar chart is fully functional** and correctly reflects the output. All drawing tools work. |
| **20 (Very Good)** | Meets "Good" criteria. JavaScript code is clean, modular, and optimized for performance (e.g., efficient canvas manipulation and WebGPU calls). |

-----

## 3\. UI/UX Design and Deployment

| Weight: 25% (Difficulty: Medium) |
| :--- |

This section focuses on the presentation, usability, and required version control/deployment.

### 3.1 User Experience (UI/UX)

  * **Design:** The application must look professional and be easy to use.
  * **Responsiveness:** The app must be fully **desktop and mobile friendly**.
  * **Single Page:** All functionality must be contained within a single `index.html` file.

### 3.2 GitHub Pages Deployment (Mandatory)

  * The final app must be hosted on your **GitHub Pages** URL and accessible to everyone.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | App runs but breaks on different screen sizes, or is not deployed on GitHub Pages. |
| **10 (Minimum)** | The app runs locally and is deployed. Design is basic, but the app is generally usable. Breaks appear on mobile devices. |
| **15 (Good)** | The app is fully responsive and uses a modern styling approach (e.g., Tailwind CSS). The UI is logical and attractive. **Successfully deployed on GitHub Pages and link is on the README.** |
| **20 (Very Good)** | Meets "Good" criteria. Exceptional, polished design and smooth, intuitive user flow. The deployment is robust and fast-loading. |

-----

## 4\. Documentation: Writing a Clean `README.md`

| Weight: 10% (Difficulty: Low) |
| :--- |

A clear and complete `README.md` is essential for any engineering project.

### Suggested `README.md` Structure

  * **Project Title & Screenshot:** Clear heading and a visual of your running app.
  * **Live Demo Link:** **CRUCIAL:** A single, easy-to-find link to your GitHub Pages URL.
  * **Overview:** A paragraph explaining the project, mentioning **tinygrad** and **WebGPU**.
  * **Features:** A bulleted list of all implemented features.
  * **Model Summary:** A small table showing the architecture and accuracy of your best MLP and CNN.
  * **Setup/Local Run:** Instructions for local setup and testing.
  * **Link to Hyperparameter Log:** Explicitly link to the `HYPERPARAMETERS.md` file.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No `README.md` file is present. |
| **10 (Minimum)** | A basic `README.md` with only a project title, a brief description, and the required links (Repo and Pages). |
| **15 (Good)** | Meets "Minimum" criteria. The `README.md` includes all suggested sections and is well-formatted. |
| **20 (Very Good)** | Meets "Good" criteria. The `README.md` is exceptionally clean, professional, and includes a brief *Project Retrospective* on technical challenges or insights. |

-----

## Summary of Weighting and Passing Criteria

| Section | Description | Weight | Passing Score (Minimum) |
| :--- | :--- | :--- | :--- |
| **1** | Model Development & Accuracy (`HYPERPARAMETERS.md`) | 35% | $\ge 10/20$ |
| **2** | Web App Functionality & Visualization | 30% | $\ge 10/20$ |
| **3** | UI/UX Design & Deployment (GitHub Pages) | 25% | $\ge 10/20$ |
| **4** | Documentation (`README.md`) | 10% | $\ge 10/20$ |
| **TOTAL** | | **100%** | **Expected Pass: $\ge 10/20$ on all sections** |

**Remember: A passing grade requires a minimum score of 10/20 in all four sections.**

### Appendix: Technical Setup

#### A. WebGPU Installation 🛠️

To run the tinygrad export script, you need the **Dawn** WebGPU library installed locally.

| OS | Installation Command | Notes |
| :--- | :--- | :--- |
| **macOS (Apple Silicon)** | `brew tap wpmed92/dawn && brew install dawn` | Requires the Homebrew package manager. |
| **Linux (x86\_64)** | `sudo curl -L https://github.com/wpmed92/pydawn/releases/download/v0.3.0/libwebgpu_dawn_x86_64.so -o /usr/lib/libwebgpu_dawn.so` | Downloads a pre-compiled shared library to the system path. |
| **Windows** | `pip install dawn-python` | Installs the Python package with the necessary WebGPU dependencies. |

#### B. Local Server for Testing 💻

WebGPU security requires files to be served via HTTP.

1.  Navigate to your project root folder in the terminal.
2.  Run the command:
    ```bash
    python -m http.server
    ```
3.  Open your browser to `http://localhost:8000`.

#### C. Bibliography & Resources 📚

| Resource | Description | Link |
| :--- | :--- | :--- |
| **tinygrad Repository** | Official framework repo. | [https://github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) |
| **WebGPU Specification** | The official standard for modern web graphics/compute. | [https://www.w3.org/TR/webgpu/](https://www.w3.org/TR/webgpu/) |
| **Tailwind CSS** | Utility-first CSS framework for fast, responsive UI development. | [https://tailwindcss.com/](https://tailwindcss.com/) |
