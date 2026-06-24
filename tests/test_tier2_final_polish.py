import pandas as pd

from src.column_detection import detect_columns
from src.dynamic_question_engine import answer_dynamic_question


def _sample_df():
    return pd.read_csv('sample_data/sample_sales_data.csv')


def test_dimension_specific_average_discount_routes_to_tier2_ranking():
    df = _sample_df()
    roles = detect_columns(df)
    ans = answer_dynamic_question('Which product has highest average discount?', df, roles)
    text = ans.get('markdown_answer', '')
    assert ans.get('handled') is True
    assert ans.get('intent') == 'dynamic_ranking'
    assert 'Avg Discount' in text
    assert 'Average Discount is' not in text
    assert '$' in text


def test_generic_correlation_sales_profit_and_quantity_profit():
    df = _sample_df()
    roles = detect_columns(df)
    for q in ['What is the correlation between sales and profit?', 'What is correlation between quantity and profit?']:
        ans = answer_dynamic_question(q, df, roles)
        text = ans.get('markdown_answer', '')
        assert ans.get('handled') is True
        assert ans.get('intent') == 'tier2_correlation'
        assert 'Correlation' in text
        assert 'Valid rows used' in text


def test_compare_and_trend_use_dollar_formatting_not_rupees():
    df = _sample_df()
    roles = detect_columns(df)
    for q in ['Compare Furniture and Technology', 'Show sales by month']:
        ans = answer_dynamic_question(q, df, roles)
        text = ans.get('markdown_answer', '')
        assert ans.get('handled') is True
        assert '$' in text
        assert '₹' not in text


def test_ship_mode_column_detection_and_unique_values():
    df = pd.DataFrame({
        'Order ID': ['A1', 'A2', 'A3'],
        'Ship Mode': ['First Class', 'Standard Class', 'First Class'],
        'Sales': [100, 200, 300],
        'Profit': [10, 20, 30],
    })
    roles = detect_columns(df)
    assert roles['roles'].get('ship_mode_column') == 'Ship Mode'
    ans = answer_dynamic_question('What ship modes are available?', df, roles)
    text = ans.get('markdown_answer', '')
    assert ans.get('handled') is True
    assert 'First Class' in text
    assert 'Standard Class' in text


def test_tier2_plural_wording_clean_for_category():
    df = _sample_df()
    roles = detect_columns(df)
    ans = answer_dynamic_question('Which category is the most profitable?', df, roles)
    text = ans.get('markdown_answer', '').lower()
    assert 'categorys' not in text
    assert 'subcategorys' not in text
    assert 'citys' not in text
