"""Section 4.4: Table 2, architecture, head and prior ablations."""

from experiments import errors, predictions
from experiments.latex import row, write_tabular

MODEL_GROUPS = {
    "Architectural components": ["w/o QFE", "w/o column attention", "w/o PMA"],
    "Head formulations": ["Ordinal", "Regression", "Focal Loss", "Classification"],
}
PRIOR_GROUPS = {
    "Prior volume": {
        "25% of prior": r"25\% of synthetic prior",
        "50% of prior": r"50\% of synthetic prior",
        "75% of prior": r"75\% of synthetic prior",
    },
    "Leave-one-generator-out": {
        "w/o networked": r"w/o networked \cite{jahn2024densired}",
        "w/o concentric shells": "w/o concentric shells",
        "w/o elliptical": r"w/o elliptical \cite{zellinger2025natural}",
        "w/o Gaussians": r"w/o Gaussians \cite{rodriguez2019clustering}",
        "w/o elongated": r"w/o elongated \cite{fachada2023generating}",
    },
}
INDENT = r"\hspace{4mm} "


def main() -> None:
    """Write Table 2 with MAE changes relative to TabK."""
    frame = predictions()
    variants = [v for group in MODEL_GROUPS.values() for v in group]
    variants += [v for group in PRIOR_GROUPS.values() for v in group]
    mae = errors(frame, ["TabK", *variants]).mean()

    def cells(variant: str | None, label: str | None = None) -> list[str]:
        if variant is None:
            return ["", "", ""]
        delta = mae[variant] - mae["TabK"]
        return [INDENT + (label or variant), f"{mae[variant]:.2f}", f"{delta:+.2f}"]

    body = []
    for i, (model_group, prior_group) in enumerate(zip(MODEL_GROUPS, PRIOR_GROUPS)):
        if i:
            body.append(r"\cmidrule(r){1-3} \cmidrule(l){4-6}")
        body.append(
            rf"\multicolumn{{3}}{{l}}{{\textit{{{model_group}}}}} & "
            rf"\multicolumn{{3}}{{l}}{{\textit{{{prior_group}}}}} \\"
        )
        models = MODEL_GROUPS[model_group]
        priors = list(PRIOR_GROUPS[prior_group].items())
        for j in range(max(len(models), len(priors))):
            left = cells(models[j]) if j < len(models) else cells(None)
            right = cells(*priors[j]) if j < len(priors) else cells(None)
            body.append(row([*left, *right]))
    header = [
        r"\textbf{Model variation} & \textbf{MAE $\downarrow$} & \textbf{$\Delta$} & "
        r"\textbf{Prior variation} & \textbf{MAE $\downarrow$} & \textbf{$\Delta$} \\"
    ]
    write_tabular("table2_ablations", r"l c c @{\hspace{10pt}} l c c", header, body)


if __name__ == "__main__":
    main()
