import contextlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = [
    {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "prompt_input": (
            "You are supposed to follow an instruction, and then the input to generate proper response.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "You are supposed to follow an instruction to generate proper response."
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "prompt_input": (
            "Please follow the instruction and input to give a response.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Please follow the instruction to give a response."
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "prompt_input": (
            "You are an expert, please listen to human instruction and input to generate the response.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "You are an expert, please listen to human instruction to generate the response.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "prompt_input": (
            "Let's follow the instruction to respond to an input.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Let's follow the instruction to generate a response.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "prompt_input": (
            "The instruction is a description of the task. You need to follow that and respond to the paired input.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "The instruction is a description of the task. You need to follow that and respond.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "Instruction:\n{instruction}\n\nResponse:"
        ),
    },
    {
        "prompt_input": (
            "You are supposed to follow an instruction, and then the input to generate proper response.\n\n"
            "#Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"
        ),
        "prompt_no_input": (
            "You are supposed to follow an instruction to generate proper response."
            "Instruction:\n{instruction}\n\nResponse:"
        ),
    },
    {
        "prompt_input": (
            "Please follow the instruction and input to give a response.\n\n"
            "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"
        ),
        "prompt_no_input": (
            "Please follow the instruction to give a response."
            "Instruction:\n{instruction}\n\nResponse:"
        ),
    },
    {
        "prompt_input": (
            "You are an expert, please listen to human instruction and input to generate the response.\n\n"
            "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"
        ),
        "prompt_no_input": (
            "You are an expert, please listen to human instruction to generate the response.\n\n"
            "Instruction:\n{instruction}\n\nResponse:"
        ),
    },
    {
        "prompt_input": (
            "Let's follow the instruction to respond to an input.\n\n"
            "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"
        ),
        "prompt_no_input": (
            "Let's follow the instruction to generate a response.\n\n"
            "Instruction:\n{instruction}\n\nResponse:"
        ),
    },
    {
        "prompt_input": (
            "The instruction is a description of the task. You need to follow that and respond to the paired input.\n\n"
            "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"
        ),
        "prompt_no_input": (
            "The instruction is a description of the task. You need to follow that and respond.\n\n"
            "Instruction:\n{instruction}\n\nResponse:"
        ),
    },
]

PROMPT_TEMPLATE_SINGLE = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class PromptTooLong(ValueError):
    """A prompt does not fit the context window of the model (prompts are never cut)."""


def encode_prompt(
    task,
    template,
    train_samples,
    eval_sample,
    tokenizer,
    context_length,
    sfc=False,
    icl_sfc=False,
    generation=False,
    generation_with_gold=False,
    max_new_tokens=None,
):
    """
    Encode prompts for eval_sample
    Input:
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, context_length: tokenizer and the context window of the model; a prompt above it
      (minus `max_new_tokens` for generation tasks) raises `PromptTooLong`, it is never truncated
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [
        template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples
    ]
    train_prompts = task.train_sep.join(train_prompts).strip()

    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc
        verbalize_fn = template.verbalize_sfc
    else:
        encode_fn = template.encode
        verbalize_fn = template.verbalize

    unverbalized_eval_prompt = encode_fn(eval_sample).strip(" ")
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [
            verbalize_fn(eval_sample, cand).strip(" ") for cand in eval_sample.candidates
        ]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [
            (len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length)
            for verbalized_eval_prompt in verbalized_eval_prompts
        ]

        if sfc:
            # Without demonstrations
            final_prompts = verbalized_eval_prompts
        else:
            # With demonstrations
            final_prompts = [
                (train_prompts + task.train_sep + eval_prompt).lstrip().strip(" ")
                for eval_prompt in verbalized_eval_prompts
            ]
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [
                (len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length)
                for verbalized_eval_prompt in verbalized_eval_prompts
            ]
            final_prompts = [
                (train_prompts + task.train_sep + eval_prompt).lstrip().strip(" ")
                for eval_prompt in verbalized_eval_prompts
            ]
        else:
            option_lens = [0]
            final_prompts = [
                (train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(" ")
            ]

    # Tokenize
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Fail, never truncate: the caller drops the example (training) or stops (evaluation).
    budget = context_length - ((max_new_tokens or 0) if generation else 0)
    longest = max(len(encoding) for encoding in encodings)
    if longest > budget:
        raise PromptTooLong(
            f"prompt of {longest} tokens does not fit the context window of {context_length} "
            f"tokens (room for {budget})"
        )

    return encodings, option_lens


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
