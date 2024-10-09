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

# Keep track of the process for the running model
model_process = None


available_models = [
    "/shares/menze.dqbm.uzh/ihamam/ct-llava-codebase/LLaVa-llama3.1/LLaVA/checkpoints_new_halfdata/finetuned_pretrained_newdataset/llava-llama3_1_8B_ctclip-finetune_256-lora_2gpus/checkpoint-114000/",
]

# Function to run the selected model
def run_model(model_selector, request: gr.Request):
    global model_process
    model_path = model_selector
    command = [
        "python", "-m", "llava.serve.model_worker",
        "--host", "0.0.0.0",
        "--controller", args.controller_url,
        "--port", "40000",  # Adjust this if necessary
        "--worker", "http://localhost:40000",
        "--model-path", model_path,
        "--model-base", model_path
    ]

    try:
        # Start the model in a subprocess
        model_process = subprocess.Popen(command)
        logger.info(f"Started model {model_path} with PID {model_process.pid}")
        return None
    except Exception as e:
        logger.error(f"Failed to start model {model_path}: {str(e)}")
        return None

# Function to disconnect the model
def disconnect_model(request: gr.Request):
    global model_process
    if model_process and model_process.poll() is None:  # Check if the process is still running
        model_process.terminate()
        model_process.wait()  # Wait for the process to terminate
        logger.info(f"Disconnected model with PID {model_process.pid}")
        model_process = None
        return None
    else:
        return None




logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

def display_slice(nifti_file, index):
    viewer = NiftiViewer()
    data = viewer.load_nifti(nifti_file.name)
    return viewer.update_slice(index=index)  # Display the slice corresponding to the given index


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
    #models = available_models
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


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        template_name = "llama3"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
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

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
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
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# CT-CHAT: A vision-language foundational chat model for 3D chest CT volumes
[📚 [Paper](https://arxiv.org/abs/2403.17834)] [[Code](https://github.com/ibrahimethemhamamci/CT-CLIP)] [[Model & Data](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)]]
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

```
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
```

""")
block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):

                with gr.Row(elem_id="run_models"):
                    run_models = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0] if len(available_models) > 0 else "",
                        interactive=True,
                        show_label=True,
                        label="Select Model",
                        #container=False,
                        allow_custom_value=True
                    )

                    run_model_button = gr.Button(value="Run Model", variant="primary")
                    disconnect_model_button = gr.Button(value="Disconnect Model", variant="secondary")

                with gr.Row(elem_id="model_selector_row"):

                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)


                # Run Model button click event
                run_model_button.click(
                    fn=run_model,
                    inputs=[run_models],
                    outputs=[textbox],  # Output could be any widget to display the result
                    show_progress=True
                )

                # Disconnect Model button click event
                disconnect_model_button.click(
                    fn=disconnect_model,
                    inputs=None,
                    outputs=[textbox],  # Output could be any widget to display the result
                    show_progress=True
                )

                def load_image(file):
                    viewer = NiftiViewer()
                    result = viewer.load_nifti(file.name)
                    if result:
                        image = result["image"]
                        max_index = result["max_index"]
                        return image, gr.update(maximum=max_index, value=0), viewer
                    return None, gr.update(maximum=100, value=0), None

                def display_slice(viewer, index):
                    if viewer is not None:
                        current_slice = viewer.update_slice(index)
                        return current_slice
                    return None

                # Accordion for file upload
                with gr.Accordion("Upload NIfTI File", open=False):
                    nifti_file = gr.File(label="Upload a NIfTI file", type="filepath")

                # Slice slider and image output outside the accordion
                slice_slider = gr.Slider(0, 100, step=1, label="Slice Index", interactive=True)
                image_output = gr.Image(label="Current Slice")

                # Initialize an empty state for the NiftiViewer instance
                viewer_state = gr.State(None)

                # Load the NIfTI file and initialize the viewer state
                nifti_file.change(
                    fn=load_image,
                    inputs=nifti_file,
                    outputs=[image_output, slice_slider, viewer_state]
                )


                # Update the image slice based on the slider value
                slice_slider.change(
                    fn=display_slice,
                    inputs=[viewer_state, slice_slider],
                    outputs=image_output,
                    show_progress=False
                )

                # Organize the layout
                imagebox = gr.Row([slice_slider, image_output])
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))
                """
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                    [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
                ], inputs=[imagebox, textbox])
                """
                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="CT-CHAT",
                    height=650,
                    layout="panel",
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    #upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    #downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    #flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

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
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
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
    parser.add_argument("--model-list-mode", type=str, default="reload",
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
