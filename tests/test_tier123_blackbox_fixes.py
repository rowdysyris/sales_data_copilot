import pandas as pd

from src.column_detection import detect_columns
from src.question_answering import answer_business_question


def _df():
    return pd.read_csv('sample_data/sample_sales_data.csv')


def _roles(df):
    return detect_columns(df)['roles']


def test_discount_band_loss_question_does_not_crash_and_uses_discount_bands():
    df = _df(); roles = _roles(df)
    ans = answer_business_question('Which discount band causes the most loss?', df, roles)
    text = ans['deterministic_answer']
    assert ans['intent'] in {'dynamic_root_cause', 'dynamic_ranking'}
    assert '20-30%' in text or '30-40%' in text or '40%+' in text
    assert '₹' not in text


def test_high_sales_low_profit_is_not_top_profit_ranking():
    df = _df(); roles = _roles(df)
    ans = answer_business_question('Which products have high sales but low profit?', df, roles)
    text = ans['deterministic_answer'].lower()
    assert ans['intent'] == 'tier2_high_sales_low_profit'
    assert 'high sales' in text and ('weak' in text or 'low' in text)


def test_missing_filter_value_is_not_silently_ignored_for_subset_percentage():
    df = _df(); roles = _roles(df)
    ans = answer_business_question('What percentage of Texas orders are unprofitable?', df, roles)
    text = ans['deterministic_answer'].lower()
    assert ans['intent'] == 'tier3_filter_value_not_found'
    assert 'texas' in text and 'not found' in text


def test_missing_named_customer_does_not_fallback_to_top_customer_discount_answer():
    df = _df(); roles = _roles(df)
    ans = answer_business_question('Should we give Sean Miller more discounts?', df, roles)
    text = ans['deterministic_answer'].lower()
    assert ans['intent'] == 'tier3_filter_value_not_found'
    assert 'sean miller' in text and 'not found' in text


def test_quick_profile_counts_are_not_formatted_as_currency():
    df = _df(); roles = _roles(df)
    ans = answer_business_question('Give me a quick data profile and overview of this dataset', df, roles)
    text = ans['deterministic_answer']
    assert '| Rows/Records | 5,015 |' in text
    assert '| Unique Orders | 5,000 |' in text
    assert '| Rows/Records | $' not in text
