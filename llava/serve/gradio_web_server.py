import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib

from .nii_gz_gradio import NiftiViewer

import subprocess
import numpy as np

cssb = """
.icon-button {
    position: relative;
    border: none;
    background: none;
    cursor: pointer;
    padding: 3px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.icon-button svg {
    transform: scale(0.9); /* Scales down to 80% of the original size */
}

.material-symbols-outlined {
    font-variation-settings:
        'FILL' 0,
        'wght' 300,
        'GRAD' 0,
        'opsz' 24;
    font-size: 16px;
    vertical-align: middle;
}

.tooltip-text {
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.3s ease-in-out, visibility 0s linear 0.3s;
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
    background-color: #333;
    color: #fff;
    padding: 5px;
    border-radius: 3px;
    font-size: 12px;
    margin-top: 5px;
    z-index: 1;
}

.icon-button:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
    transition-delay: 0.5s; /* Delay before tooltip appears */
}

.icon-button:hover .tooltip-text {
    transition-delay: 0s;
}

"""

def update_dropdown(textbox):
    # Determine dropdown value based on the content of the textbox
    if "report generation" in textbox.lower():
        return "Report Generation"
    elif "long answer" in textbox.lower():
        return "Long Answer"
    elif "multiple choice" in textbox.lower():
        return "Multiple Choice"
    elif "short answer" in textbox.lower():
        return "Short Answer"
    else:
        return "Vanilla"

# Define the check_if_encoded function
def check_if_encoded(nifti_file, slope, intercept):
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(nifti_file.name))[0]
    file_name = file_name.split(".")[0]
    print(file_name)
    # Check if the corresponding .npz file exists in the ./embeddings/ directory
    npz_file_path = os.path.join('./embeddings', f'{file_name}.npz')
    if os.path.exists(npz_file_path):
        print("ENCODED !!")
        return [
            gr.update(visible=True, value="Volume is already encoded, delete if you want to re-encode."),
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(value=slope), 
            gr.update(value=intercept) # Hide the button if .npz file exists
        ]
    else:
        print("--NOT ENCODED--")
        return [
            gr.update(visible=False), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(value=slope), 
            gr.update(value=intercept) # Show the button if .npz file does not exist
        ]
# Update the encode_volume function to accept the new parameters
def encode_volume(nifti_file, slope_shown, intercept_shown, xy_spacing, z_spacing):
    try:
        # Placeholder for the actual command with the new parameters
        command = (
            f"python encode_script.py "
            f"--path {nifti_file} --slope {slope_shown} --intercept {intercept_shown} "
            f"--xy_spacing {xy_spacing} --z_spacing {z_spacing}"
        )  # Replace with your actual script command
        print("SLOPE: ", slope_shown, "\nINTERCEPT: ", intercept_shown)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return gr.update(value="Encoding complete: Success!", visible=True), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(value=f"Encoding failed: {result.stderr}", visible=True), gr.update(visible=True), gr.update(visible=True)

    except Exception as e:
        return gr.update(value=f"An error occurred: {str(e)}", visible=True), gr.update(visible=True), gr.update(visible=True)


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "CT-CHAT Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}
"""
def display_slice(nifti_file, index):
    viewer = NiftiViewer()
    data = viewer.load_nifti(nifti_file.name)
    return viewer.update_slice(index=index)  # Display the slice corresponding to the given index
"""

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, conversation_type, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    """

    if not isinstance(prev_human_msg[1], tuple):

        # Append the appropriate token based on the conversation type
        if conversation_type == "Multiple Choice":
            prev_human_msg[1] += " <multiple_choice>"
        elif conversation_type == "Long Answer":
            prev_human_msg[1] += " <long_answer>"
        elif conversation_type == "Short Answer":
            prev_human_msg[1] += " <short_answer>"
        elif conversation_type == "Report Generation":
            prev_human_msg[1] += " <report_generation>"
    else:
        a, b = prev_human_msg
        if conversation_type == "Multiple Choice":
            b[1] += " <multiple_choice>"
        elif conversation_type == "Long Answer":
            b[1] += " <long_answer>"
        elif conversation_type == "Short Answer":
            b[1] += " <short_answer>"
        elif conversation_type == "Report Generation":
            b[1] += " <report_generation>"
        print(b)
        prev_human_msg = (a , b)
    """
    print(prev_human_msg)
    print("test regenerate")


    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, conversation_type, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image.name is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5
    #text = text[:1536]  # Hard cut-off

    html_content = """\n<div id="html-content">
        <button id="regenerate-button" class="icon-button">
            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e8eaed"><path d="M160-160v-80h110l-16-14q-52-46-73-105t-21-119q0-111 66.5-197.5T400-790v84q-72 26-116 88.5T240-478q0 45 17 87.5t53 78.5l10 10v-98h80v240H160Zm400-10v-84q72-26 116-88.5T720-482q0-45-17-87.5T650-648l-10-10v98h-80v-240h240v80H690l16 14q49 49 71.5 106.5T800-482q0 111-66.5 197.5T560-170Z"/></svg>
            <span class="tooltip-text">Regenerate</span>
        </button>
        <button id="clear-button" class="icon-button">
            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e8eaed"><path d="M280-120q-33 0-56.5-23.5T200-200v-520h-40v-80h200v-40h240v40h200v80h-40v520q0 33-23.5 56.5T680-120H280Zm400-600H280v520h400v-520ZM360-280h80v-360h-80v360Zm160 0h80v-360h-80v360ZM280-720v520-520Z"/></svg>
            <span class="tooltip-text">Clear</span>
        </button>
    </div>
    """
    print(image)
    print("test image")
    if image is not None:
        #text = text[:1200]  # Hard cut-off for images
        if '<image>' not in state.get_prompt():
            text =  "<image>\n <provided>" +text
        if conversation_type == "Multiple Choice":
            text += " <multiple_choice>"
        elif conversation_type == "Long Answer":
            text += " <long_answer>"
        elif conversation_type == "Short Answer":
            text += " <short_answer>"
        elif conversation_type == "Report Generation":
            text += " <report_generation>"

        text = (text, image, image_process_mode)
        #state = default_conversation.copy()
    else:
        if conversation_type == "Multiple Choice":
            text += " <multiple_choice>"
        elif conversation_type == "Long Answer":
            text += " <long_answer>"
        elif conversation_type == "Short Answer":
            text += " <short_answer>"
        elif conversation_type == "Report Generation":
            text += " <report_generation>"
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)

    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def http_bot(nifti_name, state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")

    start_tstamp = time.time()
    model_name = model_selector

    html_content = """\n<div id="html-content">
        <button id="regenerate-button" class="icon-button">
            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e8eaed"><path d="M160-160v-80h110l-16-14q-52-46-73-105t-21-119q0-111 66.5-197.5T400-790v84q-72 26-116 88.5T240-478q0 45 17 87.5t53 78.5l10 10v-98h80v240H160Zm400-10v-84q72-26 116-88.5T720-482q0-45-17-87.5T650-648l-10-10v98h-80v-240h240v80H690l16 14q49 49 71.5 106.5T800-482q0 111-66.5 197.5T560-170Z"/></svg>
            <span class="tooltip-text">Regenerate</span>
        </button>
        <button id="clear-button" class="icon-button">
            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e8eaed"><path d="M280-120q-33 0-56.5-23.5T200-200v-520h-40v-80h200v-40h240v40h200v80h-40v520q0 33-23.5 56.5T680-120H280Zm400-600H280v520h400v-520ZM360-280h80v-360h-80v360Zm160 0h80v-360h-80v360ZM280-720v520-520Z"/></svg>
            <span class="tooltip-text">Clear</span>
        </button>
    </div>
    """
    for message in state.messages:
        if message[-1]:
            if not isinstance(message[-1], tuple):
                message[-1] = message[-1].replace(html_content, "")

    if state.skip_next:
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        template_name = "llama3"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    controller_url = args.controller_url

    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]

    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    prompt = state.get_prompt()
    #prompt = prompt.replace(html_content,"")
    images = state.get_images(return_pil=True)

    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": images,
    }
    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    # Include the icons in the generated message
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn) #(enable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg

        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return
    #state.messages[-1][-1] = state.messages[-1][-1]
    state.messages[-1][-1] = state.messages[-1][-1] + html_content
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# CT-CHAT: A vision-language foundational chat model for 3D chest CT volumes
[[📚Paper](https://arxiv.org/abs/2403.17834)] [[💻Code](https://github.com/ibrahimethemhamamci/CT-CLIP)] [[📈Model & Data](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)]]
""")

intended_markdown = ("""
### Intended use
CT-CHAT is a vision-language chat model designed for 3D chest CT volumes, currently in research preview. It is intended for academic and research purposes only, focusing on the exploration of natural language interaction with chest CT images. The model is not suitable for clinical use or medical decision-making. Instead, it serves to advance research in medical imaging and AI, with the goal of refining its capabilities for future applications. Users are encouraged to provide feedback to improve the model.
""")

tos_markdown = ("""
### Terms of use
By accessing and using the CT-RATE dataset and related models CT-CLIP and CT-CHAT, you agree to these terms and conditions. The dataset and the models are intended solely for academic, research, and educational purposes, and any commercial use without permission is prohibited. Users must comply with all relevant laws, including data privacy and protection regulations such as GDPR and HIPAA. Sensitive information within the dataset must be kept confidential, and re-identification of individuals is forbidden. Proper citation and acknowledgment of the dataset and model providers in publications are required, and no claims of ownership or exclusive rights over the dataset or derivatives are allowed. Redistribution of the dataset and models is not permitted, and any sharing of derived data must respect privacy terms. The dataset and the models are provided “as is” without warranty, and the providers are not liable for any damages arising from its use. Violation of these terms may result in access revocation. Continued use of the dataset and the models implies acceptance of any updates to these terms, which are governed by the laws of the dataset and model providers’ location (Türkiye and Switzerland).""")


learn_more_markdown = ("""
### License
We are committed to fostering innovation and collaboration in the research community. To this end, all elements of the CT-RATE dataset and related models are released under a Creative Commons Attribution (CC-BY-NC-SA) license. This licensing framework ensures that our contributions can be freely used for non-commercial research purposes, while also encouraging contributions and modifications, provided that the original work is properly cited and any derivative works are shared under similar terms.
""")

citation_markdown = ("""
### Citing us
When using the CT-RATE dataset and related models (including CT-CHAT and CT-CLIP), please consider citing our related works:

1. @article{hamamci2024foundation,
  title={A foundation model utilizing chest CT volumes and radiology reports for supervised-level zero-shot detection of abnormalities},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Dogan, Irem and Dasdelen, Muhammed Furkan and Wittmann, Bastian and Simsar, Enis and Simsar, Mehmet and others},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}

2. @article{hamamci2023generatect,
  title={GenerateCT: Text-Conditional Generation of 3D Chest CT Volumes},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Sekuboyina, Anjany and Simsar, Enis and Tezcan, Alperen and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Almas, Furkan and Dogan, Irem and Dasdelen, Muhammed Furkan and others},
  journal={arXiv preprint arXiv:2305.16037},
  year={2023}
}

3. @article{hamamci2024ct2rep,
  title={Ct2rep: Automated radiology report generation for 3d medical imaging},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Menze, Bjoern},
  journal={arXiv preprint arXiv:2403.06801},
  year={2024}
}
""")


js_code = """
(() => {
    console.log('working');

    const observer = new MutationObserver(() => {
        // Get the elements for the Regenerate and Clear buttons
        const frontendButton = document.querySelector('#regenerate-button');
        const clearButton = document.querySelector('#clear-button');

        if (frontendButton) {
            console.log('Front end present', frontendButton);

            // Check if the Regenerate button already has the event listener to prevent adding it multiple times
            if (!frontendButton._hasClickListener) {
                frontendButton.onclick = () => {
                    console.log('Regenerate Button clicked');
                    document.getElementById('regenerate-btn').click();
                };
                frontendButton._hasClickListener = true;  // Mark that the listener has been added
            }
        }

        if (clearButton) {
            console.log('Clear button present', clearButton);

            // Check if the Clear button already has the event listener to prevent adding it multiple times
            if (!clearButton._hasClickListener) {
                clearButton.onclick = () => {
                    console.log('Clear Button clicked');
                    document.getElementById('clear-btn').click();
                };
                clearButton._hasClickListener = true;  // Mark that the listener has been added
            }
        }
    });

    // Start observing the document for changes (including subtree changes)
    observer.observe(document.body, { childList: true, subtree: true });

    return 'done';  // This line ensures the code returns something that can be awaited.
})

"""

def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(label="Prompt", show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="CT-CHAT", theme='aliabid94/test2', css=cssb, js=js_code) as demo:
        
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=30, min_width=330):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)


                # Global variable to store the NiftiViewer instance
                viewer = None
                # Define the load_image function
                def load_image(file, slope, intercept):
                    global viewer
                    viewer = NiftiViewer()
                    result = viewer.load_nifti(file.name, slope, intercept)
                    if result:
                        image = result["image"]
                        max_index = result["max_index"]
                        print("IMAGE LOADED!")
                        return [
                            gr.Group(visible=True), 
                            gr.update(value=image, visible=True), 
                            gr.update(maximum=max_index, value=0, visible=True), 
                            gr.update(visible=False), 
                            gr.update(visible=True, filepath=None), 
                            gr.Accordion(open=False, visible=True)
                        ]
                    print("--NO IMAGE--")
                    return [
                        gr.Group(visible=False), 
                        gr.update(visible=False), 
                        gr.update(maximum=100, value=0, visible=False), 
                        gr.update(visible=False), 
                        gr.update(visible=False), 
                        gr.Accordion(open=True, visible=True)
                    ]
                def update_slope_intercept(slope, intercept):
                    return gr.update(value=slope), gr.update(value=intercept)
                
                # Define the display_slice function
                def display_slice(index, window="full"):
                    global viewer
                    if viewer is not None:
                        current_slice = viewer.update_slice(index, window)
                        return current_slice
                    return None
                
                # Accordion for file upload
                with gr.Accordion("Upload NIfTI File", open=True) as nifti_accordion:
                    with gr.Group() as slope_intercept_group:
                        with gr.Row() as slope_intercept_texts:
                            with gr.Column(scale=1, min_width=100):
                                slope = gr.Textbox(label="Slope", value="1.0", interactive=True)
                            with gr.Column(scale=1, min_width=100):
                                intercept = gr.Textbox(label="Intercept", value="-1024.0", interactive=True)
                        with gr.Row() as slope_intercept_button:
                            set_slope_intercept = gr.Button(value="Set", elem_classes="button")
                    with gr.Row() as nifti_row:
                        nifti_file = gr.File(label="Upload a NIfTI file", type="filepath", visible=True, file_types=[".gz"])
                
                # Slope & intercept, window selector, slice slider and image output outside the accordion
                with gr.Group(visible=False) as image_group:
                    window_selector = gr.Radio(["full", "lung", "mediastinum"], label="Window", interactive=True, value="full")
                    slice_slider = gr.Slider(0, 100, step=1, label="Slice Index", interactive=True)
                    image_output = gr.Image(label="Shown slice")

                with gr.Group(visible=False) as params_group:
                    with gr.Row():
                        # Add these sliders/text inputs for the new parameters
                        slope_shown = gr.Textbox(label="Slope", value="1", interactive=False)
                        intercept_shown = gr.Textbox(label="Intercept", value="-1024", interactive=False)
                        xy_spacing = gr.Textbox(label="XY Spacing", value="1.0", interactive=True)
                        z_spacing = gr.Textbox(label="Z Spacing", value="1.0", interactive=True)
                    with gr.Row():
                        encode_button = gr.Button(value="Encode 3D CT Volume", visible=False)
                with gr.Row():
                    textbox_status= gr.Textbox(visible=False, label="Status")
                
                # Function to load the image only if nifti_file is provided
                def conditional_load_image(nifti_file, slope, intercept):
                    if nifti_file and nifti_file != "":
                        return load_image(nifti_file, slope, intercept)
                    else:
                        # Return updates to reset the UI when the file is cleared
                        return [
                            gr.update(visible=False),  # Hide image group
                            gr.update(value=None, visible=False),  # Clear and hide image output
                            gr.update(value=0, visible=False),  # Reset and hide slice slider
                            gr.update(value="", visible=False),  # Clear and hide textbox status
                            gr.update(value=None),  # Clear the nifti_file component
                            gr.Accordion(open=True, visible=True)  # Close the accordion
                        ]

                # Function to check if encoded only if nifti_file is provided
                def conditional_check_if_encoded(nifti_file, slope, intercept):
                    if nifti_file and nifti_file != "":
                        return check_if_encoded(nifti_file, slope, intercept)
                    else:
                        # Return updates to reset the encode button and related UI components
                        return [
                            gr.update(visible=False),  # Hide the status textbox
                            gr.update(visible=False),  # Hide the encode button
                            gr.update(visible=False),  # Hide the params group
                            gr.update(value=slope, visible=False),  # Reset slope_shown
                            gr.update(value=intercept, visible=False)  # Reset intercept_shown
                        ]
                def on_button_click():
                    print("Button was clicked!")
                # Enter pressed while focused on slope or intercept textbox
                slope.submit(
                    fn=update_slope_intercept,
                    inputs=[slope, intercept],
                    outputs=[slope_shown, intercept_shown]
                ).then(
                    fn=conditional_load_image,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[image_group, image_output, slice_slider, textbox_status, nifti_file, nifti_accordion]
                ).then(
                    fn=conditional_check_if_encoded,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[textbox_status, encode_button, params_group, slope_shown, intercept_shown]
                )
                intercept.submit(
                    fn=update_slope_intercept,
                    inputs=[slope, intercept],
                    outputs=[slope_shown, intercept_shown]
                ).then(
                    fn=conditional_load_image,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[image_group, image_output, slice_slider, textbox_status, nifti_file, nifti_accordion]
                ).then(
                    fn=conditional_check_if_encoded,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[textbox_status, encode_button, params_group, slope_shown, intercept_shown]
                )
                # Update upload button visibility on whether the slope intercept is submitted
                set_slope_intercept.click(
                    fn=update_slope_intercept,
                    inputs=[slope, intercept],
                    outputs=[slope_shown, intercept_shown]
                ).then(
                    fn=conditional_load_image,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[image_group, image_output, slice_slider, textbox_status, nifti_file, nifti_accordion]
                ).then(
                    fn=conditional_check_if_encoded,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[textbox_status, encode_button, params_group, slope_shown, intercept_shown]
                )
                
                # Load the NIfTI file and initialize the viewer state
                nifti_file.change(
                    fn=conditional_load_image,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[image_group, image_output, slice_slider, textbox_status, nifti_file, nifti_accordion]
                ).then(
                    fn=conditional_check_if_encoded,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[textbox_status, encode_button, params_group, slope_shown, intercept_shown]
                )
                def uploaded(slope, intercept):
                    return [
                        gr.update(visible=True),  # Show encode button
                        gr.update(visible=True),  # Show params group
                        gr.Accordion(open=False, visible=True),  # Ensure the accordion is closed but visible
                        gr.update(value=slope),  # Set slope value
                        gr.update(value=intercept)  # Set intercept value
                    ]
                nifti_file.upload(
                    fn=uploaded,
                    #lambda slope=slope, intercept=intercept: [gr.update(visible=True)] * 2 + [gr.Accordion(open=False), gr.update(value=slope), gr.update(value=intercept)],  # Show the parameters row and the button
                    inputs=[slope, intercept],
                    outputs=[encode_button, params_group, nifti_accordion, slope_shown, intercept_shown]
                ).then(
                    fn=conditional_load_image,
                    inputs=[nifti_file, slope, intercept],
                    outputs=[image_group, image_output, slice_slider, textbox_status, nifti_file, nifti_accordion]
                )
                def clear_nifti_file():
                    return [
                        gr.update(value=None),            # Clear the nifti_file input
                        gr.update(value=None, visible=False),  # Clear and hide the image output
                        gr.update(value=0, visible=False),     # Reset and hide the slice slider
                        gr.update(value="", visible=False),    # Clear and hide the textbox status
                        gr.update(open=True, visible=True),    # Open Accordion for new upload
                        gr.update(visible=False),         # Hide the encode button
                        gr.update(visible=False)          # Hide the params group
                    ]
                # Clear the NIfTI file input and reset UI components
                nifti_file.clear(
                    fn=clear_nifti_file,
                    inputs=[],
                    outputs=[nifti_file, image_output, slice_slider, textbox_status, nifti_accordion, encode_button, params_group]
                )
                # Update the image slice based on the slider value
                slice_slider.change(
                    fn=display_slice,
                    inputs=[slice_slider, window_selector],
                    outputs=image_output,
                    show_progress=False
                )
                window_selector.change(
                    fn=display_slice,
                    inputs=[slice_slider, window_selector],
                    outputs=image_output,
                )
                # Update the encode_button.click function to pass the new parameters
                encode_button.click(
                    fn=encode_volume,
                    inputs=[nifti_file, slope_shown, intercept_shown, xy_spacing, z_spacing],
                    outputs=[textbox_status, encode_button, params_group]
                )

                # Organize the layout
                #imagebox = gr.Row([slice_slider, image_output])
                imagebox = image_output

                conversation_type_selector = gr.Dropdown(
                    choices=["Vanilla", "Multiple Choice", "Long Answer", "Short Answer", "Report Generation"],
                    value="Vanilla",
                    label="Select Conversation Type",
                    interactive=True,
                    show_label=True
                )

                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=120):
                with gr.Row():
                    '''
                    with gr.Column(visible=False):
                        
                        gr.HTML(value="""
                        <button id="frontend-button">Click me</button>
                        """)
                    '''
                    with gr.Column():
                        chatbot = gr.Chatbot(
                            elem_id="chatbot",
                            label="CT-CHAT",
                            height=650,
                            layout="panel",
                        )
                with gr.Row():
                    with gr.Column(scale=95):
                        textbox.render()
                    with gr.Column(scale=5, min_width=30):
                        submit_btn = gr.Button(value="➤", interactive=True, variant="primary")

                # Embed regenerate and clear buttons under the textbox
                with gr.Row():
                    regenerate_btn = gr.Button(value="🔄 Regenerate", interactive=True, variant="secondary", visible=False, elem_id="regenerate-btn")
                    clear_btn = gr.Button(value="🗑️ Clear", interactive=True, variant="secondary", visible=False, elem_id="clear-btn")


                with gr.Row():
                    check_example=gr.Textbox(label="Conversation Type", visible=False)

                    examples = gr.Examples(
                        examples=[
                            [f"{cur_dir}/examples/example_ct.nii.gz", "Can you generate a report for this patient?", "report generation", "1", "-1024"],
                            [f"{cur_dir}/examples/example_ct.nii.gz", "What are the findings of this patient?", "long answer", "1", "-1024"],
                            [f"{cur_dir}/examples/example_ct.nii.gz", "Which of the following is a finding in the mediastinum and hilar regions on the Chest CT image? (a) Lymph nodes up to 15 mm (b) Atherosclerosis (c) Diffuse degenerative changes in bone structures (d) Tapering of the vertebral corpus endplates", "multiple choice", "1", "-1024"],
                            [f"{cur_dir}/examples/example_ct.nii.gz", "What types of anomalies are visible in the scan?", "short answer", "1", "-1024"]
                        ], 
                        inputs=[nifti_file, textbox, check_example, slope, intercept]
                    )
                    check_example.change(
                        fn=update_dropdown,
                        inputs=[check_example],
                        outputs=[conversation_type_selector]
                    )
                """
                with gr.Row(elem_id="buttons") as button_row:
                    #upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    #downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    #flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
                """

        if not embed_mode:
            gr.Markdown(intended_markdown)
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
            gr.Markdown(citation_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [regenerate_btn, clear_btn]
        """
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        """
        regenerate_btn.click(
            regenerate,
            [state, image_process_mode, conversation_type_selector],
            [state, chatbot, textbox] + btn_list
        ).then(
            http_bot,
            [nifti_file, state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, nifti_file, image_process_mode, conversation_type_selector],
            [state, chatbot, textbox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [nifti_file, state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        submit_btn.click(
            add_text,
            [state, textbox, nifti_file, image_process_mode, conversation_type_selector],
            [state, chatbot, textbox] + btn_list
        ).then(
            http_bot,
            [nifti_file, state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
