import pytest
from src.doc2img.dataloader import function_sum

def test_basic():
    assert (function_sum(1,2) == 3)