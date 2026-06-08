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
