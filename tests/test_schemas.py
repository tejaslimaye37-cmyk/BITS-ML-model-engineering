import pytest
from pydantic import ValidationError

from serving.schemas import CustomerRecord


def test_valid_record(record):
    assert CustomerRecord(**record).tenure == 1


def test_optional_fields_can_be_omitted():
    r = CustomerRecord(tenure=5, MonthlyCharges=50.0, Contract="One year")
    assert r.TotalCharges is None


@pytest.mark.parametrize("patch", [{"tenure": -3}, {"MonthlyCharges": -1.0}, {"tenure": "abc"},
                                   {"MonthlyCharges": float("inf")}, {"SeniorCitizen": 2}, {"Contract": ""},
                                   {"unknown_field": 1}])
def test_malformed_payloads_are_rejected(record, patch):
    with pytest.raises(ValidationError):
        CustomerRecord(**{**record, **patch})


def test_missing_required_field_is_rejected(record):
    record.pop("Contract")
    with pytest.raises(ValidationError):
        CustomerRecord(**record)
