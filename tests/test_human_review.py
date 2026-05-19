from app.rag.human_review import parse_selected_indexes


def test_parse_selected_indexes_accepts_comma_separated_numbers() -> None:
    assert parse_selected_indexes("1, 3,5", max_index=5) == [1, 3, 5]


def test_parse_selected_indexes_ignores_invalid_values() -> None:
    assert parse_selected_indexes("0, x, 2, 9", max_index=3) == [2]
