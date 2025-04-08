# ComfyUI Deployment with Modal

This repository contains the necessary files to deploy the ComfyUI web interface using Modal.

## Functionality

It sets up a Modal application that builds a container image with ComfyUI, necessary dependencies, custom nodes, and models defined in `config.toml`. It then launches the ComfyUI web server accessible via a Modal web endpoint.

## Demo

[![ComfyUI Modal Deployment Demo Video](https://img.youtube.com/vi/gMfVRiC6ymI/hqdefault.jpg)](https://www.youtube.com/watch?v=gMfVRiC6ymI)

## Setup

1.  **Clone the Repository:**
    Clone this repository to your local machine:
    ```bash
    git clone https://github.com/AINxtGen/comfy_modal.git
    cd comfy_modal
    ```

2.  **Install Local Dependencies:**
    Install the Modal and the `toml` library:
    ```bash
    pip install modal toml
    ```

3.  **Set up Modal Account:**
    - Create an account on [Modal](https://modal.com/).
    - Authenticate with Modal using the CLI:
      ```bash
      modal token new
      ```
    - Alternatively, create a token manually at [https://modal.com/settings/tokens](https://modal.com/settings/tokens).

4.  **Set up Hugging Face Token:**
    - Create a Hugging Face account and generate an access token with at least `read` permissions on the [Hugging Face website](https://huggingface.co/settings/tokens).
    - Store your token as a Modal secret named `huggingface-token`:
      ```bash
      modal secret create huggingface-token HF_TOKEN="your_hf_token_here"
      ```
    *(Replace `"your_hf_token_here"` with your actual Hugging Face token)*.

## Deployment

Once the local setup is complete, you can deploy the ComfyUI interface using the following command:

```bash
modal deploy comfy_ui.py
```

This will build the image (if not already built) and start the web server on Modal. The command output will provide the URL to access the ComfyUI interface.

## Configuration

-   **Modal Settings:** Adjust concurrency, timeouts, and GPU type in `config.toml` under `[modal_settings]`.
-   **Models:** Define models to download in `config.toml` under `[[models]]`.
-   **Custom Nodes:** List custom node Git repository URLs in `config.toml` under `[nodes]`.
-   **Extra Dependencies:** Add extra pip packages in `config.toml` under `[extra_dependencies]`.
