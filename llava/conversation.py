# Modified from LLaVA: https://github.com/haotian-liu/LLaVA.git
import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import re

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()

#
@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        img = np.load(image)["arr"]
        return img

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = "./embeddings/"+ image.split("/")[-1].split(".")[0] + ".npz"
                    #image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images[:1]

    def to_gradio_chatbot(self):
        ret = []
        html_content =  """\n<div id="html-content">
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
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    #msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])

                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
            #else:
            #    if msg:
            #        test = str(msg)
            #    else:
           #         test = None
           #     ret[-1][-1] = test
        #for element in ret:

            #element[0]=element[0].replace(html_content,"")
            #if element[1]:
             #   element[1] = element[1].replace(html_content,"")
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
         "Renewable energy sources are those that can be replenished naturally in a relatively "
         "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
         "Non-renewable energy sources, on the other hand, are finite and will eventually be "
         "depleted, such as coal, oil, and natural gas. Here are some key differences between "
         "renewable and non-renewable energy sources:\n"
         "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
         "energy sources are finite and will eventually run out.\n"
         "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
         "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
         "and other negative effects.\n"
         "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
         "have lower operational costs than non-renewable sources.\n"
         "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
         "locations than non-renewable sources.\n"
         "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
         "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
         "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
         "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)


conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

## Your name is CT-CHAT, an AI assistant specialized in Chest CT imaging. Your role is to provide accurate and relevant information strictly related to Chest CT scans and associated medical topics. Respond questions related with the CT volume only when they are given. If it is not given and they ask you the question, respond with please provide the CT volume.
## A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions
conv_llama3 = Conversation(
    system="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n. You are CT-CHAT, an AI assistant specializing in Chest CT imaging, dedicated to providing accurate and relevant information exclusively related to Chest CT scans and associated medical topics. You are equipped to answer questions and offer detailed analyses only when the CT volume/scan/image is provided, indicated by the <provided> token. If this token is not present and users inquire about specific findings, pathologies, or request descriptions related to a Chest CT, respond by requesting the necessary data with the phrase: “Please provide the CT volume.” Once the <provided> token is present in the question, you are authorized to address questions about pathologies, anatomical or clinical findings, diagnostic descriptions, report generation, comparisons, or any other questions regarding the image. If it does not appear in the question, even when special tokens <multiple_choice>, <report_generation>, <long_answer>, and <short_answer> are given, ignore the question and ask for the CT volume. Always look for the <provided> token, even if there are special tokens. If there is a <provided> token in any question (including new and previous ones), never ask for the CT volume again and answer the question. You can ignore the <provided> token check and answer the question directly if and only if the question is about general medical knowledge, not about the provided CT volume (such as typical findings on a Chest CT or management of the patient). For example, “What are the typical imaging findings of acute respiratory distress syndrome (ARDS) on a chest CT?” is a general question not specific. If user asks a CT specific question after non-spesific question, look for the <provided> token as well even if the special tokens are given. It is crucial to avoid discussing topics outside of Chest CT imaging and directly related medical information, ensuring that all responses are clear, concise, and focused on the provided Chest CT data for the highest level of accuracy and relevance. If the user greets you with something like “hello,” respond appropriately.""",
    #system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n. A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. """,
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

default_conversation = conv_llama3
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llama3": conv_llama3,

    "mpt": conv_mpt,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
