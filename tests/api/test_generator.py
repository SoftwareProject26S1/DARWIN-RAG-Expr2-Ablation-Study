def test_mlx_generator_loads_model_lazily_and_reuses_it():
    from darwin_rag_exp2.api.generator import MlxLmGenerator

    load_calls = []
    generate_calls = []

    def fake_load(model_name):
        load_calls.append(model_name)
        return "model", "tokenizer"

    def fake_generate(model, tokenizer, *, prompt, max_tokens, temp):
        generate_calls.append(
            {
                "model": model,
                "tokenizer": tokenizer,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temp": temp,
            }
        )
        return "  generated answer  "

    generator = MlxLmGenerator(
        "local/model",
        max_tokens=64,
        temperature=0.2,
        loader=fake_load,
        generate_fn=fake_generate,
    )

    assert load_calls == []
    assert generator.generate("prompt one") == "generated answer"
    assert generator.generate("prompt two") == "generated answer"

    assert load_calls == ["local/model"]
    assert [call["prompt"] for call in generate_calls] == ["prompt one", "prompt two"]
    assert generate_calls[0]["max_tokens"] == 64
    assert generate_calls[0]["temp"] == 0.2


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
