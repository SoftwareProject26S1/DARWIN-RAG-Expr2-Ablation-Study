import json

from darwin_rag_exp2.retrieval.settings import (
    build_lambda_by_category,
    load_primary_run_settings,
    write_primary_run_settings,
)


def test_build_lambda_by_category_recomputes_from_mu_sigma_and_parameters() -> None:
    rows = [
        {
            "category": "학사",
            "mu_confidence": 0.9,
            "sigma_confidence": 0.1,
        },
        {
            "category": "장학",
            "mu_confidence": 0.7,
            "sigma_confidence": 0.4,
        },
    ]

    lambdas = build_lambda_by_category(rows, alpha=8.0, rho=4.0, tau=0.5)

    assert lambdas == {
        "학사": 0.942676,
        "장학": 0.5,
    }


def test_primary_run_settings_round_trip_yaml(tmp_path) -> None:
    settings_path = tmp_path / "frozen.yaml"

    write_primary_run_settings(
        settings_path,
        candidate_k_per_partition=50,
        report_top_k=10,
        generation_context_top_n=5,
        theta_route=0.6,
        lambda_fixed=0.5,
        lambda_by_category={"학사": 0.9},
        tuning_metadata={"dev_metric": "ndcg@10"},
    )

    settings = load_primary_run_settings(settings_path)
    payload = settings_path.read_text()

    assert settings.candidate_k_per_partition == 50
    assert settings.report_top_k == 10
    assert settings.generation_context_top_n == 5
    assert settings.theta_route == 0.6
    assert settings.lambda_fixed == 0.5
    assert settings.lambda_by_category == {"학사": 0.9}
    assert "dev_metric" in payload


def test_load_primary_settings_can_derive_lambdas_from_category_stats(tmp_path) -> None:
    settings_path = tmp_path / "frozen.yaml"
    category_stats_path = tmp_path / "category_stats.json"
    settings_path.write_text(
        "\n".join(
            [
                "candidate_k_per_partition: 50",
                "report_top_k: 10",
                "generation_context_top_n: 5",
                "theta_route: 0.6",
                "lambda_fixed: 0.5",
                "adaptive_lambda:",
                "  alpha: 8.0",
                "  rho: 4.0",
                "  tau: 0.5",
            ]
        ),
        encoding="utf-8",
    )
    category_stats_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "category": "학사",
                        "mu_confidence": 0.9,
                        "sigma_confidence": 0.1,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    settings = load_primary_run_settings(
        settings_path,
        category_stats_path=category_stats_path,
    )

    assert settings.lambda_by_category == {"학사": 0.942676}
