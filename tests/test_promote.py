from platform_core.promote import should_promote


def test_cold_start_promotes():
    assert should_promote(None, {"accuracy": 0.40})


def test_challenger_wins():
    assert should_promote({"accuracy": 0.51}, {"accuracy": 0.52})


def test_champion_defends():
    assert not should_promote({"accuracy": 0.52}, {"accuracy": 0.51})


def test_tie_goes_to_challenger_at_zero_edge():
    # fresher data wins ties: retraining on newer data is worth it at equal skill
    assert should_promote({"accuracy": 0.52}, {"accuracy": 0.52}, min_edge=0.0)


def test_min_edge_blocks_marginal_win():
    assert not should_promote({"accuracy": 0.520}, {"accuracy": 0.521}, min_edge=0.005)
