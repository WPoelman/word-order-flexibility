import conllu
import pytest

from word_order_flexibility.flexibility import word_order_flexibility
from word_order_flexibility.ud import calculate_ud_metrics, surface_sentences


SAMPLE = """# sent_id = 1
1\tcats\tcat\tNOUN\t_\t_\t2\tnsubj\t_\t_
2\tchase\tchase\tVERB\t_\t_\t0\troot\t_\t_
3\tdogs\tdog\tNOUN\t_\t_\t2\tobj\t_\t_
4\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = 2
1\tchase\tchase\tVERB\t_\t_\t0\troot\t_\t_
2\tcats\tcat\tNOUN\t_\t_\t1\tnsubj\t_\t_
3\tdogs\tdog\tNOUN\t_\t_\t1\tobj\t_\t_

# sent_id = 3
1\tToday\ttoday\tADV\t_\t_\t4\tadvmod\t_\t_
2\tcats\tcat\tNOUN\t_\t_\t4\tnsubj\t_\t_
3\toften\toften\tADV\t_\t_\t4\tadvmod\t_\t_
4\tchase\tchase\tVERB\t_\t_\t0\troot\t_\t_
5\tdogs\tdog\tNOUN\t_\t_\t4\tobj\t_\t_

"""


def test_matches_original_metric_snapshot():
    sentences = conllu.parse(SAMPLE)
    ud = calculate_ud_metrics(sentences)
    flexibility = word_order_flexibility(surface_sentences(sentences), decay=0.5)
    assert ud["head_direction_entropy_weighted"] == pytest.approx(0.576851090830413)
    assert ud["relation_order_entropy_weighted"] == pytest.approx(1.5238095238095237)
    assert ud["head_direction_entropy_unweighted"] == pytest.approx(0.5127565251825893)
    assert ud["relation_order_entropy_unweighted"] == pytest.approx(1.1428571428571428)
    assert flexibility["flexibility"] == pytest.approx(0.26666666666666666)
    assert flexibility["qualifying_pairs"] == 3
    assert surface_sentences(sentences)[2][0] == "Today"


def test_descriptive_ud_names_distinguish_aggregations():
    ud = calculate_ud_metrics(conllu.parse(SAMPLE))
    assert "head_direction_entropy_weighted" in ud
    assert "relation_order_entropy_weighted" in ud
    assert "head_direction_entropy_unweighted" in ud
    assert "relation_order_entropy_unweighted" in ud
