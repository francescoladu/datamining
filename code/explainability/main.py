import argparse
import sys
from pathlib import Path


module_dir = Path(__file__).resolve().parent
code_dir = module_dir.parent

if str(code_dir) not in sys.path:
    sys.path.insert(
        0,
        str(code_dir),
    )


from explainability import config
from shared.modeling import (
    load_development_and_test,
    rebuild_final_model,
)


def parse_arguments() -> argparse.Namespace:
    """Select exactly one experiment to explain."""

    parser = argparse.ArgumentParser(
        description=(
            "Run explainability analysis "
            "for one Data Mining experiment."
        )
    )

    parser.add_argument(
        "experiment",
        choices=[
            "1",
            "2",
            "3",
        ],
        help=(
            "Experiment to explain: "
            "1 = raw data, "
            "2 = exact deduplication, "
            "3 = weighted deduplication."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    experiment = args.experiment

    # IMPORTANT:
    # Configure all paths BEFORE importing the modules
    # that read those paths from explainability.config.
    config.configure_experiment(
        experiment
    )

    from explainability.permutation_importance import (
        run_permutation_importance,
    )

    from explainability.plots import (
        run_permutation_importance_plot,
    )

    from explainability.shap_force import (
        run_shap_force,
    )

    config.create_output_directories()

    (
        X_train,
        y_train,
        w_train,
        X_test,
        y_test,
        w_test,
    ) = load_development_and_test(
        config.TRAIN_DATA_PATH,
        config.TEST_DATA_PATH,
    )

    (
        model,
        model_name,
        dev_score,
        parameters,
    ) = rebuild_final_model(
        X_train=X_train,
        y_train=y_train,
        sample_weight=w_train,
        parameters_path=(
            config.FINAL_BEST_PARAMETERS_PATH
        ),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"EXPLAINABILITY - EXPERIMENT {experiment}"
    )

    print(
        "=" * 70
    )

    print(
        f"Run: {config.RUN_NAME}"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Test rows: {len(X_test)}"
    )

    print(
        f"Training observation mass: "
        f"{w_train.sum():.0f}"
    )

    print(
        f"Test observation mass: "
        f"{w_test.sum():.0f}"
    )

    print(
        f"Input features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"SHAP sample position: "
        f"{config.SAMPLE_POSITION}"
    )

    # ---------------------------------------------------------
    # Global explainability:
    # nested-CV permutation importance
    # ---------------------------------------------------------

    importance_df = (
        run_permutation_importance(
            model_name
        )
    )

    run_permutation_importance_plot(
        importance_df
    )

    # ---------------------------------------------------------
    # Local explainability:
    # SHAP on one held-out test sample
    # ---------------------------------------------------------

    run_shap_force(
        model=model,
        X_background=X_train,
        X_test=X_test,
        sample_position=(
            config.SAMPLE_POSITION
        ),
        class_to_explain=(
            config.CLASS_TO_EXPLAIN
        ),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"EXPLAINABILITY EXPERIMENT "
        f"{experiment} COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()