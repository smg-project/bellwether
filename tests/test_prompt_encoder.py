"""The vendor-code oracle over a checkpoint that ships a prompt encoder in place of a chat template."""

import hashlib
import json
import pathlib
import shutil

import pytest

from bellwether.inputs import oracle_inputs, ships_a_template, vendor_encoder
from bellwether.record import vendor
from bellwether.record.reference import HfTemplateOracle
from bellwether.record.roundtrip import RoundtripOracle

# A toy encoder in the shape of the DeepSeek checkpoints': ``encode_messages`` renders the whole chat format, a
# trailing assistant turn closed, the assistant's header after any other turn; it marks its import, as vendor code
# that must never run outside the sandbox would.
ENCODER = """
import json
import pathlib

pathlib.Path(__file__).with_name("imported").write_text("yes")

REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}
IMAGE_PLACEHOLDER = "<image>"


def encode_messages(messages, thinking_mode, context=None, drop_thinking=True, reasoning_effort=None):
    assert thinking_mode in ("chat", "thinking"), thinking_mode
    text = "<|im_start|>"
    if thinking_mode == "thinking" and reasoning_effort is not None:
        text += f"effort={reasoning_effort}\\n"
    last_user = max((index for index, message in enumerate(messages) if message["role"] == "user"), default=-1)
    for index, message in enumerate(messages):
        if message.get("tools"):
            text += "tools=" + ",".join(tool["function"]["name"] for tool in message["tools"]) + "\\n"
        content = message.get("content") or ""
        if message["role"] == "system" and not content:
            continue
        text += f"{message['role']}:"
        # As V3.2's encoder: in thinking mode the turns after the last user turn keep their reasoning, the earlier
        # ones only when nothing is dropped.
        if message["role"] == "assistant" and thinking_mode == "thinking" and (index > last_user or not drop_thinking):
            text += f"<think>{message.get('reasoning_content') or ''}</think>"
        text += content
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]  # the vendor's convention: the JSON string, decoded here
            text += f"<call {call['function']['name']} {json.dumps(json.loads(arguments), ensure_ascii=False)}>"
        text += "<|im_end|>\\n"
    if messages[-1]["role"] != "assistant":
        text += "assistant:" + ("<think>" if thinking_mode == "thinking" else "")
    return text
"""

V32, V4, V41 = "DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"
HI = {"messages": [{"role": "user", "content": "Hi"}]}
THINKING = "<|im_start|>effort={}\nuser:Hi<|im_end|>\nassistant:<think>"
CHAT = "<|im_start|>user:Hi<|im_end|>\nassistant:"
REQUEST = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


@pytest.fixture
def encoder_checkpoint(tmp_path, tiny_model, monkeypatch) -> pathlib.Path:
    """A checkpoint that ships no chat template and a prompt encoder under ``encoding/``, of an architecture the
    engine's server renders with its ``deepseek_v32`` tokenizer mode; the sandbox mark set."""
    directory = tmp_path / "encoder-chat"
    shutil.copytree(tiny_model, directory)
    config = json.loads((directory / "tokenizer_config.json").read_text())
    del config["chat_template"]
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
    with_architecture(directory, V32)
    (directory / "encoding").mkdir()
    (directory / "encoding" / "encoding_toy.py").write_text(ENCODER)
    (directory / "encoding" / "test_encoding_toy.py").write_text("raise SystemExit('never run')\n")
    (directory / "encoding" / "README.md").write_text("How to call the encoder.\n")
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    return directory


def with_architecture(checkpoint: pathlib.Path, architecture: str) -> None:
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "toy", "architectures": [architecture]}))


def test_the_encoder_module_is_an_oracle_input_of_a_checkpoint_without_a_template(encoder_checkpoint):
    assert vendor_encoder(encoder_checkpoint) == "encoding/encoding_toy.py"
    inputs = oracle_inputs(str(encoder_checkpoint), "local")
    assert list(inputs) == [
        "config.json",
        "encoding/encoding_toy.py",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    module = encoder_checkpoint / "encoding" / "encoding_toy.py"
    assert inputs["encoding/encoding_toy.py"] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert not (encoder_checkpoint / "encoding" / "imported").exists(), "listing the inputs ran the vendor's code"


def test_a_checkpoint_with_a_template_keeps_the_template_unless_the_engine_renders_its_architecture_with_the_encoder(
    encoder_checkpoint, tiny_model
):
    shutil.copyfile(tiny_model / "tokenizer_config.json", encoder_checkpoint / "tokenizer_config.json")
    assert ships_a_template(encoder_checkpoint)
    # The engine renders V3.2's architecture with the encoder whatever the checkpoint ships: the encoder stays.
    assert vendor_encoder(encoder_checkpoint) == "encoding/encoding_toy.py"
    with_architecture(encoder_checkpoint, "SomeOtherForCausalLM")
    assert vendor_encoder(encoder_checkpoint) is None
    assert "encoding/encoding_toy.py" not in oracle_inputs(str(encoder_checkpoint), "local")


def test_a_template_file_counts_as_a_template(encoder_checkpoint):
    with_architecture(encoder_checkpoint, "SomeOtherForCausalLM")
    assert vendor_encoder(encoder_checkpoint) == "encoding/encoding_toy.py"
    (encoder_checkpoint / "chat_template.jinja").write_text("{{ messages }}")
    assert vendor_encoder(encoder_checkpoint) is None


def test_in_the_sandbox_the_encoder_renders_a_checkpoint_that_ships_a_template_too(encoder_checkpoint, tiny_model):
    shutil.copyfile(tiny_model / "tokenizer_config.json", encoder_checkpoint / "tokenizer_config.json")
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    assert oracle.render(REQUEST).text.endswith("<|im_end|>\nassistant:")
    assert "chat_template_sha256" not in oracle.provenance()


def test_in_the_sandbox_the_encoder_renders_as_the_engines_deepseek_v32_mode_calls_it(encoder_checkpoint):
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    rendered = oracle.render(REQUEST)
    assert rendered.text == "<|im_start|>user:What is the capital of France?<|im_end|>\nassistant:"
    assert rendered.input_ids == oracle.tokenizer.encode(rendered.text, add_special_tokens=False)
    assert rendered.input_ids[0] == 1 and rendered.template_source == "encoder"
    assert rendered.content_format == "string" and rendered.developer_role == "as-written"
    provenance = oracle.provenance()
    assert provenance["oracle"] == vendor.SOURCE and provenance["engine_renderer"] == "deepseek_v32"
    assert provenance["encoder_module"] == "encoding/encoding_toy.py"
    module = encoder_checkpoint / "encoding" / "encoding_toy.py"
    assert provenance["encoder_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert "chat_template_sha256" not in provenance


@pytest.mark.parametrize(
    ("request_fields", "ending"),
    [
        ({"chat_template_kwargs": {"enable_thinking": True}}, "assistant:<think>"),
        ({"chat_template_kwargs": {"thinking": True}}, "assistant:<think>"),
        ({"reasoning_effort": "high"}, "assistant:<think>"),  # the top-level effort switches thinking on
        ({"reasoning_effort": "none"}, "assistant:"),
        ({"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high"}, "assistant:"),
        ({"add_generation_prompt": False}, "assistant:"),  # the mode reads no generation-prompt flag
    ],
)
def test_the_deepseek_v32_mode_reads_the_thinking_kwargs_as_the_engine_folds_them(
    encoder_checkpoint, request_fields, ending
):
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    assert oracle.render({**REQUEST, **request_fields}).text.endswith(ending)


def test_the_deepseek_v32_mode_puts_the_tools_on_a_new_leading_system_message(encoder_checkpoint):
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    request = {"messages": [{"role": "system", "content": "Be brief."}, *REQUEST["messages"]], "tools": [TOOL]}
    rendered = oracle.render(request)
    assert rendered.text.startswith("<|im_start|>tools=get_weather\nsystem:Be brief.<|im_end|>\n")
    assert request["messages"][0] == {"role": "system", "content": "Be brief."}, "the request was written to"


@pytest.mark.parametrize(
    ("request_fields", "text"),
    [
        ({}, THINKING.format("high")),
        ({"chat_template_kwargs": {"reasoning_effort": "minimal"}}, THINKING.format("low")),
        ({"chat_template_kwargs": {"reasoning_effort": "max"}}, THINKING.format("max")),
        ({"reasoning_effort": "xhigh"}, THINKING.format("high")),
        ({"reasoning_effort": "none"}, CHAT),
        ({"chat_template_kwargs": {"enable_thinking": False}}, CHAT),
        ({"chat_template_kwargs": {"thinking": False, "reasoning_effort": "max"}}, CHAT),
    ],
)
def test_the_deepseek_v4_mode_defaults_to_thinking_and_folds_the_effort(encoder_checkpoint, request_fields, text):
    with_architecture(encoder_checkpoint, V4)
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    assert oracle.provenance()["engine_renderer"] == "deepseek_v4"
    assert oracle.render({**HI, **request_fields}).text == text


def test_the_deepseek_v4_mode_puts_the_tools_on_the_first_system_message(encoder_checkpoint):
    with_architecture(encoder_checkpoint, V4)
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    with_system = {"messages": [{"role": "system", "content": "Be brief."}, *REQUEST["messages"]], "tools": [TOOL]}
    assert oracle.render(with_system).text.startswith("<|im_start|>effort=high\ntools=get_weather\nsystem:Be brief.")
    without = {**REQUEST, "tools": [TOOL]}
    assert oracle.render(without).text.startswith("<|im_start|>effort=high\ntools=get_weather\nuser:")


def test_the_deepseek_v32_mode_gets_the_messages_as_its_renderer_hands_them(encoder_checkpoint):
    """The engine's V3.2 renderer parses the request in the string content format before its mode renders: the text
    parts of a list content joined with a newline, a part of another kind written as the model's placeholder (refused
    here), an assistant's call arguments the JSON text of the object they decode to, an empty tool_calls dropped."""
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}]}
    assert oracle.render(parts).text == "<|im_start|>user:A\nB<|im_end|>\nassistant:"
    assert isinstance(parts["messages"][0]["content"], list), "the request was written to"
    image = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}
    with pytest.raises(ValueError, match="placeholder"):
        oracle.render(image)
    call = {"id": "call0", "type": "function", "function": {"name": "get_weather", "arguments": '{ "city" : "Paris" }'}}
    history = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call0", "content": "18"},
        {"role": "user", "content": "Thanks"},
    ]
    spaced = oracle.render({"messages": history, "tools": [TOOL]}).text
    compact = dict(call, function={**call["function"], "arguments": '{"city": "Paris"}'})
    history[1] = {**history[1], "tool_calls": [compact]}
    assert spaced == oracle.render({"messages": history, "tools": [TOOL]}).text
    assert '<call get_weather {"city": "Paris"}>' in spaced
    shaped = vendor.as_the_renderer_hands_them([{"role": "assistant", "content": "ok", "tool_calls": []}], "string")
    assert shaped == [{"role": "assistant", "content": "ok"}]


def test_the_deepseek_v4_mode_gets_the_messages_as_its_renderer_hands_them(encoder_checkpoint):
    """The engine's V4 renderer parses the request in the openai content format: a list content stays a list for the
    vendor's module; the arguments shaping is the same, with an argument string that is not a JSON object coerced to
    ``{}`` as the engine coerces it."""
    with_architecture(encoder_checkpoint, V4)
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    broken = {"id": "call0", "type": "function", "function": {"name": "get_weather", "arguments": "not json"}}
    history = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [broken]},
        {"role": "tool", "tool_call_id": "call0", "content": "18"},
        {"role": "user", "content": "Thanks"},
    ]
    assert "<call get_weather {}>" in oracle.render({"messages": history, "tools": [TOOL]}).text
    assert history[1]["tool_calls"][0]["function"]["arguments"] == "not json", "the request was written to"
    parts = [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
    shaped = vendor.as_the_renderer_hands_them([{"role": "user", "content": parts}], "openai")
    assert shaped == [{"role": "user", "content": parts}]
    kept = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}],
    }
    assert vendor.as_the_renderer_hands_them([kept], "openai") == [kept]


def test_the_deepseek_v41_mode_normalizes_the_messages_and_checks_the_effort(encoder_checkpoint):
    with_architecture(encoder_checkpoint, V41)
    oracle = HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    parts = [{"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": "x"}}]
    rendered = oracle.render({"messages": [{"role": "user", "content": parts}], "reasoning_effort": "low"})
    assert rendered.text == "<|im_start|>effort=low\nuser:Describe\n\n<image><|im_end|>\nassistant:<think>"
    assert oracle.render({**REQUEST, "chat_template_kwargs": {"reasoning_effort": 42}}).text.startswith(
        "<|im_start|>effort=42\n"
    )
    assert oracle.render({**REQUEST, "reasoning_effort": "none"}).text.endswith("assistant:")
    with pytest.raises(ValueError, match="reasoning_effort must be one of high, low, max"):
        oracle.render({**REQUEST, "chat_template_kwargs": {"reasoning_effort": "xhigh"}})
    with pytest.raises(ValueError, match="refuses the role"):
        oracle.render({"messages": [{"role": "latest_reminder", "content": "x"}, *REQUEST["messages"]]})


def test_outside_the_sandbox_the_encoder_is_refused_before_its_module_is_read(encoder_checkpoint, monkeypatch):
    monkeypatch.delenv(vendor.SANDBOX_ENV, raising=False)
    with pytest.raises(vendor.OutsideTheSandbox):
        HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    assert not (encoder_checkpoint / "encoding" / "imported").exists()


def test_without_the_vendor_flag_a_checkpoint_with_an_encoder_is_refused(encoder_checkpoint, tiny_model):
    with pytest.raises(ValueError, match="ships a prompt encoder the engine's server renders with"):
        HfTemplateOracle(str(encoder_checkpoint), "local")
    shutil.copyfile(tiny_model / "tokenizer_config.json", encoder_checkpoint / "tokenizer_config.json")
    with pytest.raises(ValueError, match="ships a prompt encoder the engine's server renders with"):
        HfTemplateOracle(str(encoder_checkpoint), "local")  # a template it ships too is not the reference
    assert not (encoder_checkpoint / "encoding" / "imported").exists()


def test_an_encoder_of_an_architecture_the_engine_has_no_mode_for_is_refused(encoder_checkpoint):
    with_architecture(encoder_checkpoint, "SomeOtherForCausalLM")
    with pytest.raises(ValueError, match="no tokenizer mode of its own"):
        HfTemplateOracle(str(encoder_checkpoint), "local", vendor_code=True)
    assert not (encoder_checkpoint / "encoding" / "imported").exists()


def test_in_the_sandbox_the_round_trip_renders_the_turn_through_the_encoder(encoder_checkpoint):
    oracle = RoundtripOracle(str(encoder_checkpoint), "local", vendor_code=True)
    output = oracle.render_output(REQUEST, {"role": "assistant", "content": "Paris."})
    assert output.text == "Paris." and output.end_of_turn["stop_id"] == 2  # the turn ends on <|im_end|>
    assert output.template_source == "encoder" and output.content_format == "string"
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    with_call = oracle.render_output({**REQUEST, "tools": [TOOL]}, {"role": "assistant", "tool_calls": [call]})
    assert with_call.text == '<call get_weather {"city": "Paris"}>' and with_call.finish_reason == "tool_calls"
    assert oracle.provenance()["tool_call_arguments"] == "string"


def test_in_the_sandbox_the_checks_render_their_variants_of_a_call_carrying_turn_through_the_encoder(
    encoder_checkpoint,
):
    """The checks' variant renders take the turn as the encoder takes it (each call's arguments the JSON string),
    like the case's own render: handed the object a template gets, the encoder refused every variant and every check
    skipped the call-carrying case, so a lossy encoder was recorded unchecked."""
    oracle = RoundtripOracle(str(encoder_checkpoint), "local", vendor_code=True)
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    message = {"role": "assistant", "tool_calls": [call]}
    rendered = oracle.render_variant(REQUEST["messages"], {"tools": [TOOL]}, message)
    assert rendered is not None and rendered.endswith('<call get_weather {"city": "Paris"}><|im_end|>\n')
    # An encoder that writes nothing for the arguments: the first check that changes them (adding one) sees the same
    # turn and refuses the case, where before the fix every check skipped it.
    encoder = oracle.renderer.prompt_encoder
    render = encoder.render

    def without_arguments(messages, tools, kwargs, last_role=None):
        stripped = [
            {**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": "{}"}} for c in m["tool_calls"]]}
            if m.get("tool_calls")
            else m
            for m in messages
        ]
        return render(stripped, tools, kwargs, last_role)

    encoder.render = without_arguments
    with pytest.raises(
        ValueError, match="does not render every tool call: adding an argument to call 0 \\(get_weather\\)"
    ):
        oracle.render_output({**REQUEST, "tools": [TOOL]}, message)


def test_the_round_trip_keeps_the_requests_last_role_for_the_appended_turn(encoder_checkpoint):
    """DeepSeek V3.2's mode drops earlier reasoning after a user turn: the appended assistant turn must not flip it,
    or the turn would not extend the prompt."""
    oracle = RoundtripOracle(str(encoder_checkpoint), "local", vendor_code=True)
    history = [
        {"role": "user", "content": "Is 91 prime?"},
        {"role": "assistant", "reasoning_content": "7 times 13.", "content": "No."},
        {"role": "user", "content": "And 97?"},
    ]
    request = {"messages": history, "chat_template_kwargs": {"thinking": True}}
    output = oracle.render_output(request, {"role": "assistant", "reasoning_content": "Prime.", "content": "Yes."})
    assert output.text == "Prime.</think>Yes." and output.end_of_turn["stop_id"] == 2


def test_a_reasoning_carrying_case_under_a_chat_mode_request_is_refused_through_the_encoder(encoder_checkpoint):
    """The ``deepseek_v32`` mode is thinking-off unless the request switches it on, and a chat-mode turn carries no
    reasoning: a case whose message states ``reasoning_content`` is refused as the reasoning probes are, a call-carrying
    one too (its variant renders go through the encoder since ``as_given``; handed the object a template gets, the
    encoder raised on them and the check was skipped, and four such cases were once recorded without their reasoning).
    The same message under the mode's switch records the reasoning and its closer, after the request's last role kept
    the prompt a prefix."""
    oracle = RoundtripOracle(str(encoder_checkpoint), "local", vendor_code=True)
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    message = {"role": "assistant", "reasoning_content": "The user asks about Paris.", "tool_calls": [call]}
    with pytest.raises(ValueError, match="does not render the message's reasoning"):
        oracle.render_output({**REQUEST, "tools": [TOOL]}, message)
    with pytest.raises(ValueError, match="does not render the message's reasoning"):
        oracle.render_output(REQUEST, {"role": "assistant", "reasoning_content": "Nothing to say.", "content": "Hi."})
    thinking = {**REQUEST, "tools": [TOOL], "chat_template_kwargs": {"thinking": True}}
    output = oracle.render_output(thinking, message)
    assert output.text == 'The user asks about Paris.</think><call get_weather {"city": "Paris"}>'
    assert output.finish_reason == "tool_calls" and output.end_of_turn["stop_id"] == 2
