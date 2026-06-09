def test_mlx_generator_loads_model_lazily_and_reuses_it():
    from darwin_rag_exp2.api.generator import MlxLmGenerator

    load_calls = []
    generate_calls = []
    sampler_calls = []

    def fake_load(model_name):
        load_calls.append(model_name)
        return "model", "tokenizer"

    def fake_generate(model, tokenizer, **kwargs):
        generate_calls.append(
            {
                "model": model,
                "tokenizer": tokenizer,
                **kwargs,
            }
        )
        return "  generated answer  "

    generator = MlxLmGenerator(
        "local/model",
        max_tokens=64,
        temperature=0.2,
        loader=fake_load,
        generate_fn=fake_generate,
        sampler_factory=lambda **kwargs: sampler_calls.append(kwargs) or "sampler",
    )

    assert load_calls == []
    assert generator.generate("prompt one") == "generated answer"
    assert generator.generate("prompt two") == "generated answer"

    assert load_calls == ["local/model"]
    assert [call["prompt"] for call in generate_calls] == ["prompt one", "prompt two"]
    assert generate_calls[0]["max_tokens"] == 64
    assert generate_calls[0]["sampler"] == "sampler"
    assert sampler_calls[0] == {"temp": 0.2}


def test_mlx_generator_can_enable_qwen_thinking_mode():
    from darwin_rag_exp2.api.generator import MlxLmGenerator

    prompts = []

    def fake_generate(model, tokenizer, **kwargs):
        prompt = kwargs["prompt"]
        prompts.append(prompt)
        return "answer"

    generator = MlxLmGenerator(
        "Qwen/Qwen3-4B-MLX-4bit",
        thinking_mode="think",
        loader=lambda model_name: ("model", "tokenizer"),
        generate_fn=fake_generate,
    )

    assert generator.generate("prompt") == "answer"
    assert prompts == ["prompt\n\n/think"]


def test_mlx_generator_passes_sampling_settings():
    from darwin_rag_exp2.api.generator import MlxLmGenerator

    sampler_calls = []
    logits_processor_calls = []
    generate_calls = []

    def fake_sampler_factory(**kwargs):
        sampler_calls.append(kwargs)
        return "sampler"

    def fake_logits_processors_factory(**kwargs):
        logits_processor_calls.append(kwargs)
        return ["processor"]

    def fake_generate(model, tokenizer, *, prompt, max_tokens, sampler, logits_processors):
        generate_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "sampler": sampler,
                "logits_processors": logits_processors,
            }
        )
        return "answer"

    generator = MlxLmGenerator(
        "Qwen/Qwen3-4B-MLX-4bit",
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.05,
        presence_penalty=0.2,
        thinking_mode="think",
        sampler_factory=fake_sampler_factory,
        logits_processors_factory=fake_logits_processors_factory,
        loader=lambda model_name: ("model", "tokenizer"),
        generate_fn=fake_generate,
    )

    assert generator.generate("prompt") == "answer"
    assert sampler_calls == [
        {"temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    ]
    assert logits_processor_calls == [
        {"repetition_penalty": 1.05, "presence_penalty": 0.2}
    ]
    assert generate_calls == [
        {
            "prompt": "prompt\n\n/think",
            "max_tokens": 2048,
            "sampler": "sampler",
            "logits_processors": ["processor"],
        }
    ]


def test_clean_generated_answer_strips_thinking_block_and_repeated_tail():
    from darwin_rag_exp2.api.generator import clean_generated_answer

    raw_answer = (
        "<think>근거를 검토한다.</think>\n\n"
        "검색된 근거에는 교내근로 신청 기간이 없습니다.\n\n"
        "최종 답변: 확인이 필요합니다. 교내근로 신청 기간은 근거에 없습니다. "
        "--- (최종 답변은 근거에 기반하여 작성되었으며, 근거 없이 추측한 내용은 포함하지 않음) "
        "--- (최종 답변은 근거에 기반하여 작성되었으며, 근거 없이 추측한 내용은 포함하지 않음)"
    )

    assert (
        clean_generated_answer(raw_answer)
        == "확인이 필요합니다. 교내근로 신청 기간은 근거에 없습니다."
    )


def test_vllm_generator_can_disable_qwen_thinking_mode():
    from types import SimpleNamespace

    from darwin_rag_exp2.api.generator import VllmGenerator

    prompts = []

    class FakeLlm:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts_arg, sampling_params):
            prompts.extend(prompts_arg)
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="answer"),
                    ],
                ),
            ]

    generator = VllmGenerator(
        "Qwen/Qwen3-4B",
        thinking_mode="no_think",
        llm_factory=FakeLlm,
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert generator.generate("prompt") == "answer"
    assert prompts == ["prompt\n\n/no_think"]


def test_vllm_generator_passes_sampling_settings():
    from types import SimpleNamespace

    from darwin_rag_exp2.api.generator import VllmGenerator

    sampling_calls = []

    class FakeLlm:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling_params):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="answer"),
                    ],
                ),
            ]

    def fake_sampling_params(**kwargs):
        sampling_calls.append(kwargs)
        return {"sampling": kwargs}

    generator = VllmGenerator(
        "Qwen/Qwen3-4B",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.05,
        presence_penalty=0.2,
        llm_factory=FakeLlm,
        sampling_params_factory=fake_sampling_params,
    )

    assert generator.generate("prompt") == "answer"
    assert sampling_calls == [
        {
            "temperature": 0.6,
            "max_tokens": 512,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.2,
        }
    ]


def test_vllm_generator_ignores_sampling_settings_unsupported_by_runtime():
    from types import SimpleNamespace

    from darwin_rag_exp2.api.generator import VllmGenerator

    sampling_calls = []

    class FakeLlm:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling_params):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="answer"),
                    ],
                ),
            ]

    def limited_sampling_params(*, temperature, max_tokens, top_p):
        sampling_calls.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
        )
        return {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}

    generator = VllmGenerator(
        "Qwen/Qwen3-4B",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.05,
        presence_penalty=0.2,
        llm_factory=FakeLlm,
        sampling_params_factory=limited_sampling_params,
    )

    assert generator.generate("prompt") == "answer"
    assert sampling_calls == [
        {
            "temperature": 0.6,
            "max_tokens": 512,
            "top_p": 0.95,
        }
    ]


def test_vllm_generator_loads_engine_lazily_and_reuses_it():
    from types import SimpleNamespace

    from darwin_rag_exp2.api.generator import VllmGenerator

    engine_calls = []
    sampling_calls = []
    generate_calls = []

    class FakeLlm:
        def __init__(self, **kwargs):
            engine_calls.append(kwargs)

        def generate(self, prompts, sampling_params):
            generate_calls.append(
                {
                    "prompts": prompts,
                    "sampling_params": sampling_params,
                }
            )
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="  vllm generated answer  "),
                    ],
                ),
            ]

    def fake_sampling_params(**kwargs):
        sampling_calls.append(kwargs)
        return {"sampling": kwargs}

    generator = VllmGenerator(
        "local/vllm-model",
        max_tokens=96,
        temperature=0.3,
        llm_factory=FakeLlm,
        sampling_params_factory=fake_sampling_params,
    )

    assert engine_calls == []
    assert generator.generate("prompt one") == "vllm generated answer"
    assert generator.generate("prompt two") == "vllm generated answer"

    assert engine_calls == [{"model": "local/vllm-model"}]
    assert [call["prompts"] for call in generate_calls] == [["prompt one"], ["prompt two"]]
    assert sampling_calls == [
        {"temperature": 0.3, "max_tokens": 96},
        {"temperature": 0.3, "max_tokens": 96},
    ]


def test_vllm_generator_warm_start_loads_engine_once():
    from types import SimpleNamespace

    from darwin_rag_exp2.api.generator import VllmGenerator

    engine_calls = []

    class FakeLlm:
        def __init__(self, **kwargs):
            engine_calls.append(kwargs)

        def generate(self, prompts, sampling_params):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="answer"),
                    ],
                ),
            ]

    generator = VllmGenerator(
        "local/vllm-model",
        llm_factory=FakeLlm,
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    generator.warm_start()

    assert generator.generate("prompt") == "answer"
    assert engine_calls == [{"model": "local/vllm-model"}]


def test_vllm_generator_passes_optional_engine_settings():
    from types import SimpleNamespace

    from darwin_rag_exp2.api.generator import VllmGenerator

    engine_calls = []

    class FakeLlm:
        def __init__(self, **kwargs):
            engine_calls.append(kwargs)

        def generate(self, prompts, sampling_params):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="answer"),
                    ],
                ),
            ]

    generator = VllmGenerator(
        "local/vllm-model",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        tokenizer="local/tokenizer",
        hf_config_path="local/config",
        llm_factory=FakeLlm,
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert generator.generate("prompt") == "answer"
    assert engine_calls == [
        {
            "model": "local/vllm-model",
            "tokenizer": "local/tokenizer",
            "hf_config_path": "local/config",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
        }
    ]
