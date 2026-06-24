import pandas as pd

from src.column_detection import detect_columns
from src.dynamic_question_engine import answer_dynamic_question, interpretIntent, checkAssumption, handleMultiIntent, handleFollowUp


def load_superstore():
    df = pd.read_csv('tests/fixtures/sample_superstore.csv', encoding='latin1')
    roles = detect_columns(df)
    return df, roles


def answer(q: str) -> str:
    df, roles = load_superstore()
    res = answer_dynamic_question(q, df, roles)
    assert res.get('handled'), res
    return res['markdown_answer']


def assert_tier4_shape(ans: str):
    for heading in ['INTERPRETATION', 'ANSWER', 'DATA', 'INSIGHT', 'RECOMMENDATION']:
        assert heading in ans
    assert '₹' not in ans


def test_tier4_public_api_helpers_never_return_null():
    assert interpretIntent('How are we doing?')['intent'] == 'business_health'
    assert checkAssumption('Our furniture business is doing well right?', pd.DataFrame(), {})['assumes_positive'] is True
    assert len(handleMultiIntent('Give me sales and also tell me what to fix')) >= 2
    assert handleFollowUp('What about the west?')['entity'] == 'west'


def test_q1_non_unique_interprets_both_meanings():
    ans = answer('How many non unique datasets are there?')
    assert_tier4_shape(ans)
    assert '0 fully duplicate rows' in ans
    assert '4,985 rows share Order IDs' in ans
    assert '5,009 unique orders' in ans


def test_q2_how_are_we_doing_business_health():
    ans = answer('How are we doing?')
    assert_tier4_shape(ans)
    assert '$2,297,200.86' in ans
    assert '$286,397.02' in ans
    assert '12.47%' in ans
    assert '18.72%' in ans
    assert 'Tables' in ans and 'Furniture' in ans


def test_q3_profit_killers():
    ans = answer('What is killing our profit?')
    assert_tier4_shape(ans)
    assert 'Tables' in ans and '-$17,725.48' in ans
    assert '40%+' in ans and '-$99,558.59' in ans
    assert 'Cubify CubeX' in ans


def test_q4_discounts_full_profile():
    ans = answer('Tell me about discounts')
    assert_tier4_shape(ans)
    assert '15.62%' in ans
    assert 'Discount-profit correlation' in ans
    assert '40%+' in ans and '-$99,558.59' in ans
    assert 'Most discounted categories' in ans


def test_q5_wrong_assumption_gets_pushback():
    ans = answer('Our furniture business is doing well right?')
    assert_tier4_shape(ans)
    assert 'Not fully' in ans
    assert '2.49%' in ans
    assert 'Tables' in ans and '-$17,725.48' in ans


def test_q6_follow_up_west_profile():
    ans = answer('What about the west?')
    assert_tier4_shape(ans)
    assert 'West generated $725,457.82' in ans
    assert '$108,418.45' in ans
    assert 'Region comparison' in ans
    assert 'Top products in West' in ans
    assert 'Top sub-categories in West' in ans


def test_q7_compare_technology_furniture_verdict():
    ans = answer('Which is better — technology or furniture?')
    assert_tier4_shape(ans)
    assert 'Technology is better' in ans
    assert '17.40%' in ans
    assert '2.49%' in ans
    assert 'Furniture' in ans


def test_q8_multi_intent_sales_and_fix():
    ans = answer('Give me sales and also tell me what to fix')
    assert_tier4_shape(ans)
    assert '$2,297,200.86' in ans
    assert 'Weakest sub-category: Tables' in ans
    assert '40%+' in ans
    assert 'Top loss-making products' in ans


def test_q9_loss_orders_root_causes():
    ans = answer('Why are we losing money in some orders?')
    assert_tier4_shape(ans)
    assert '1,871 order rows' in ans
    assert '-$156,131.29' in ans
    assert '40%+' in ans and 'Tables' in ans


def test_q10_executive_top3_exactly_three_actions():
    ans = answer('Summarize everything and tell me the 3 most important things I should act on right now')
    assert_tier4_shape(ans)
    assert '$2,297,200.86' in ans
    assert '$286,397.02' in ans
    assert ans.count('\n1.') == 1
    assert ans.count('\n2.') == 1
    assert ans.count('\n3.') == 1
    assert '40%+' in ans and 'Tables' in ans and 'Technology' in ans and 'West' in ans


def test_tier123_spot_checks_still_work():
    checks = {
        'How many rows and columns does this dataset have?': ['9,994', '21'],
        'Which category is the most profitable?': ['Technology', '$145,454.95'],
        'What is the profit margin by category?': ['17.40%', '2.49%'],
        'What percentage of our orders are unprofitable?': ['18.72%', '1,871'],
        'Is First Class shipping worth it vs Standard Class?': ['First Class', 'Standard Class', 'verdict'],
    }
    for q, expected in checks.items():
        ans = answer(q)
        assert '₹' not in ans
        for e in expected:
            assert e.lower() in ans.lower()
