from tests.test_manager_demo_readiness import demo_df, demo_roles
from src.question_answering import answer_business_question
from src.report_generator import generate_pdf_report, _clean_items, _format_cell


def test_enhanced_pdf_report_is_information_rich():
    ctx = answer_business_question("Give me board-level summary", demo_df(), demo_roles())["context_results"]
    pdf = generate_pdf_report(ctx)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 9000


def test_pdf_text_helpers_clean_manager_output():
    items = _clean_items("Driver one.; Driver two; Driver one.")
    assert items == ["Driver one.", "Driver two."]
    assert _format_cell(-42.86, "Profit Margin Percent") == "-42.86%"
    assert "$" in _format_cell(-15525, "Profit")
