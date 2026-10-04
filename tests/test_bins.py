from mafia_lie_detector.bins import write_bins


def row(model, family, label, text, verified=True):
    return {"model": model, "family": family, "label": label, "text": text, "model_verified": verified}


def test_three_tiers_and_unidentified_box(tmp_path):
    rows = [
        row("m1", "acme", "lying", "I am  totally  town."),
        row("m1", "acme", "truth", "Vote the quiet one."),
        row("m2", "acme", "lying", "Trust me on this."),
        row("x1", "other", "truth", "No kill tonight, interesting."),
        row("z2", "custom", "lying", "Homemade AI lies here.", verified=False),
    ]
    manifest = write_bins(rows, tmp_path)["lines_per_file"]
    read = lambda p: (tmp_path / p).read_text().splitlines()  # noqa: E731
    assert read("all/deceiving.txt") == ["I am totally town.", "Trust me on this."]  # Z2 excluded, whitespace fixed
    assert read("all/truthful.txt") == ["Vote the quiet one.", "No kill tonight, interesting."]
    assert read("families/acme/deceiving.txt") == ["I am totally town.", "Trust me on this."]
    assert read("families/acme/m1/deceiving.txt") == ["I am totally town."]
    assert read("families/acme/m2/deceiving.txt") == ["Trust me on this."]
    assert read("families/_unidentified/z2/deceiving.txt") == ["Homemade AI lies here."]
    assert not (tmp_path / "families/acme/m2/truthful.txt").exists()  # no empty files
    assert manifest["all/deceiving.txt"] == 2 and manifest["families/_unidentified/deceiving.txt"] == 1
