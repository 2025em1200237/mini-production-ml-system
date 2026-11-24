import pytest
from pydantic import ValidationError

from serving.schemas import BatchRequest, TransactionRequest
from tests.conftest import VALID


def test_valid_payload_parses():
    r = TransactionRequest(**VALID)
    assert r.type == "TRANSFER" and r.amount == 1000.0


def test_type_is_normalised_to_upper():
    assert TransactionRequest(**{**VALID, "type": "cash_out"}).type == "CASH_OUT"


@pytest.mark.parametrize("field,value", [
    ("amount", -1.0), ("amount", "abc"), ("amount", float("nan")), ("amount", float("inf")),
    ("type", "CRYPTO"), ("step", 0), ("transaction_id", ""), ("oldbalanceOrg", -5.0),
])
def test_bad_values_rejected(field, value):
    with pytest.raises(ValidationError):
        TransactionRequest(**{**VALID, field: value})


def test_missing_required_field_rejected():
    payload = {k: v for k, v in VALID.items() if k != "amount"}
    with pytest.raises(ValidationError):
        TransactionRequest(**payload)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        TransactionRequest(**{**VALID, "isFraud": 1})


def test_optional_balances_reported_as_missing():
    r = TransactionRequest(**{**VALID, "oldbalanceDest": None, "newbalanceDest": None})
    # newbalanceDest is not a model input, so only oldbalanceDest is reported
    assert r.missing_fields() == ["oldbalanceDest"]


def test_batch_size_limits():
    with pytest.raises(ValidationError):
        BatchRequest(records=[])
    with pytest.raises(ValidationError):
        BatchRequest(records=[VALID] * 1001)
